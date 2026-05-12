---
tags:
  - 论文分析
  - llm-inference
  - simulation
  - hardware-software-co-design
arxiv: 2408.05499
authors:
  - Jaehong Cho
  - Minsu Kim
  - Hyunmin Choi
  - Guseul Heo
  - Jongse Park
institutions:
  - KAIST (School of Computing, Daejeon, South Korea)
created: 2026-05-11
rating: ⭐⭐⭐⭐
conference: IISWC 2024
doi: 10.1109/IISWC63097.2024.00012
github: https://github.com/casys-kaist/llmservingsim
---

# LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale

## 一、论文概览

### 1.1 基本信息

LLMServingSim 是由韩国 KAIST 计算学院的 Jongse Park 团队（Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park）开发的面向大规模 LLM 推理服务的**硬件-软件协同仿真框架**。论文发表于 **IISWC 2024**，代码开源在 GitHub。

**核心洞察**：现有的 AI 系统仿真器（如 ASTRA-sim）主要针对**训练场景**设计，其迭代计算是"相同且重复的"；而 LLM **推理服务**的每一次自回归迭代处理的 batch 构成、序列长度都**动态变化**，需要全新的仿真设计。

### 1.2 要解决的核心问题

| 问题 | 具体表现 |
|------|---------|
| **动态工作负载变化** | LLM 推理的自回归特性导致每次迭代的输入 batch、序列长度、KV cache 状态各不相同 |
| **重复仿真开销** | 现有仿真器对每个迭代从头编译和运行硬件仿真，未利用 LLM 架构中的计算冗余 |
| **异构仿真需求** | 现代 LLM 系统常采用 NPU+PIM 等异构加速器，缺少支持灵活插件式硬件仿真的系统级框架 |
| **仿真速度瓶颈** | mNPUsim 运行一次推理迭代需 ~10 小时，GeneSys 需 ~1.5 小时，根本无法用于系统级探索 |

### 1.3 主要贡献

1. **迭代级硬件-系统联合仿真**：将 LLM 推理服务的仿真拆解为逐迭代的调度 → 硬件仿真 → 系统仿真循环
2. **计算复用优化**：利用 decoder 架构冗余（编译一个 transformer block 并复制；分离 attention/non-attention 层缓存结果）
3. **异构加速器插件框架**：提供标准化接口，允许用户以插件形式接入任意加速器的编译器+仿真器栈
4. **KV cache 感知的精细内存建模**：引入 vLLM 式 demand paging 机制，模拟页分配、逐出和重载
5. **精度与速度验证**：与真实 GPU 系统误差 < 14.7%，仿真速度相比现有加速器仿真器加速 **34.7× ~ 491×**

---

## 二、技术方法详解

### 2.1 系统架构与工作流

LLMServingSim 建立在 **ASTRA-sim** 之上，但做了根本性的重新设计。其工作流由四个核心组件构成：

```
Scheduler (调度器)
    ↓ 生成 batch 配置
Execution Engine Stack (执行引擎栈)
    ├── NPU Engine (PolyMath Compiler + GeneSys Simulator)
    ├── PIM Engine (In-house PIM Simulator)
    └── 可扩展的其他 Engine
    ↓ 输出硬件仿真 trace
Graph Converter (图转换器)
    ↓ 构建 Chakra 执行图
ASTRA-sim (系统级仿真)
    ↓ 仿真结果反馈回 Scheduler
Scheduler 更新 timer → 下一迭代...
```

#### 1) Scheduler（调度器）
- 接收用户请求，根据到达时间与调度器 timer 决定可 batch 的请求
- 基于 operator mapping 策略将不同算子分配到不同加速器引擎
- 在异构环境下，将 memory-bound 算子（Score, Attend）映射到 PIM，compute-bound 算子（QKV, FFN）映射到 NPU
- 每个迭代结束后更新 timer，组装新 batch

#### 2) Execution Engine Stack（执行引擎栈）
- 每个异构加速器有独立的 **compiler + simulator** 组合
- 对 NPU 使用 **PolyMath Compiler + GeneSys Simulator**
- 对 PIM 使用 in-house PIM simulator
- 引擎栈内部的 **Operator Scheduler** 使用贪心启发式调度算子执行顺序，最大化异构硬件利用率

#### 3) Graph Converter（图转换器）
- 将硬件仿真 trace 转换为 **Chakra 执行图**
- 根据配置的并行策略（tensor/pipeline/hybrid）分发算子和插入通信原语（All-Reduce）
- 在异构池间插入数据传输算子

#### 4) ASTRA-sim（系统仿真器）
- 接收 Chakra 执行图，进行系统级网络互连、通信、同步仿真
- 将结果反馈给 Scheduler

### 2.2 迭代级调度（Iteration-Level Scheduling）

这是 LLMServingSim 最核心的设计理念：

```
预填充阶段 (Prefill/Initiation Phase):
  输入: 完整 prompt → GEMM 密集型 → 生成第一个 token + KV cache
生成阶段 (Generation/Autoregressive Phase):
  输入: 上一 token → GEMV 密集型 → 利用 KV cache 生成下一 token
```

与传统训练仿真（所有迭代相同）不同，LLM 推理的每次迭代：
- batch 构成不同（新请求加入、完成请求退出）
- 生成阶段每次处理一个 token，序列长度递增
- KV cache 动态增长

LLMServingSim 的设计：逐次执行 **Prompt Scheduling → Hardware Simulation (per accelerator) → System Simulation (ASTRA-sim) → 更新 Scheduler Timer**，循环往复。

### 2.3 快速仿真技术（核心创新）

#### 2.3.1 模型冗余复用（Model Redundancy Reuse）

LLM 的 decoder-only 架构包含**大量重复的 transformer block**：

```
[Embedding] → [Block 1] → [Block 2] → ... → [Block N] → [LM Head]
                ↓ 相同结构 ↓
```

**优化策略**：只编译 **一个** transformer block，然后对结果进行复制（replicate），大幅缩减编译时间。

#### 2.3.2 计算复用 / 结果缓存（Computation Reuse via Caching）

进一步利用 attention 层和 non-attention 层的特性差异：

| 层级类型 | 耗时 | 复用频率 |
|---------|------|---------|
| Non-attention 层（QKV, FFN） | 高 | 高（只要 batch size/seq len 不变） |
| Attention 层（Score, Attend） | 低 | 低（每次迭代都变） |

- 由于预填充和生成阶段的唯一区别在于 **attention 层**（是否有 KV cache），non-attention 层的仿真结果可以被**缓存复用**
- 具体做法：对 non-attention layers 只编译仿真一次，后续迭代仅替换 attention layers

#### 2.3.3 加速效果

| 优化 | 速度提升 |
|------|---------|
| 整体 vs mNPUsim | ~491× |
| 整体 vs GeneSys | ~34.7× |
| 整体 vs NeuPIMs | ~45× |
| 计算复用（开/关） | 6.4× ~ 12.2× |

### 2.4 异构系统仿真

#### 2.4.1 异构拓扑支持

支持两种异构拓扑（如图 5 所示）：

**(a) NPU-PIM 紧耦合**：NPU 和 PIM 直连，作为单个节点在系统层面出现
- Operator mapping 在 **Execution Engine 内部调度器**完成
- memory-bound 算子 → PIM module；compute-bound 算子 → NPU module

**(b) NPU 池 + PIM 池分离**：通过 CXL 等高带宽互联
- Operator mapping 在 **Scheduler + Graph Converter** 完成
- Scheduler 创建不同的仿真计划分发给 NPU/PIM 各自的 execution engine
- Graph Converter 在不同池之间插入数据传输算子

#### 2.4.2 Operator Mapping 完整流程（Algorithm 1）

```
1. Batch_Formatting(L_req, Mem_free, Time_cur)
2. Batch_Partitioning(Batch, Criteria) → L_sub_batch
3. for each sub_batch:
     a. Operator_Profiling(sub_batch) → L_ops
     b. Operator_Mapping(L_ops, L_dev) → L_ops_mapped
     c. for each (operator, device):
            Execution_Engine(operator, device) → Ops_sim
     d. Operator_Scheduling(L_sub_batch_sim) → Trace
4. Graph_Converter(Trace) → G_exec
```

### 2.5 KV Cache 感知的内存建模

- 在 ASTRA-sim 的内存模型基础上，增加了**容量约束**和**页式内存管理**
- 采用 vLLM 风格的 **demand paging** 机制
- Scheduler 在每个迭代检查：
  - 新请求需要多少 KV cache 页
  - 已生成 token 是否需要新页
  - 内存不足时：逐出最近添加请求的 KV cache 页到 host memory
  - 后续迭代有空闲内存时：从 host 重载逐出的页
- Graph Converter 在 Chakra 图中插入对应的 **page eviction / reload** 数据传输算子

### 2.6 并行策略支持

| 并行策略 | 实现方式 |
|---------|---------|
| Tensor Parallelism | 权重矩阵分片到多个节点，插入 All-Reduce 同步 |
| Pipeline Parallelism | decoder blocks 顺序分配到不同节点 |
| Hybrid Parallelism | 组内 tensor parallel + 组间 pipeline parallel |
| 选择性批处理 | Attention layers 分配唯一 ID，分布到不同 worker 并行处理 |

### 2.7 ONNX 兼容性

- 输入格式：**ONNX**（Open Neural Network Exchange）
- 可与 PyTorch、TensorFlow 等主流框架转换互操作
- 支持直接导入开源社区的 ONNX 格式 LLM 模型

---

## 三、实验评估

### 3.1 实验设置

| 项目 | 配置 |
|------|------|
| **真实基线系统** | 4× NVIDIA RTX 3090 (24GB) + Intel Xeon Gold 6326 |
| **LLM 服务框架** | vLLM |
| **NPU 配置** | 128×128 Systolic Array, 1GHz (模拟 RTX 3090 性能) |
| **PIM 配置** | 与 NeuPIMs 相同的规格 |
| **互联** | PCIe 4.0 ×16 (64GB/s, 100ns latency) |
| **仿真主机** | Intel Xeon Gold 6226R, 96GB DRAM |
| **模型** | GPT-3 (7B, 30B, 175B), LLaMA (7B, 30B) |
| **数据集** | ShareGPT, Alpaca |

### 3.2 仿真精度验证

#### 3.2.1 同构 NPU 系统（vs vLLM on 4× RTX 3090）

- **预填充阶段吞吐趋势**：LLMServingSim 与 GPU 系统高度吻合
  - 吞吐量受调度决策和 KV cache 容量共同影响
  - 验证了迭代级调度和精细内存建模的准确性
- **生成阶段吞吐趋势**：整体趋势一致，但存在一些偏差
  - 原因：NPU 架构难以精确匹配 GPU 性能
  - GPU 使用 FlashAttention 等 kernel 优化，LLMServingSim 尚未模拟这些优化
- **平均误差率**：< **14.7%**

#### 3.2.2 异构 NPU-PIM 系统（vs NeuPIMs）

- 使用 Alpaca 数据集，256 个请求
- 不同模型尺寸和并行方案下：
  - 各配置误差率均 < 20%
  - **几何平均误差率：8.88%**
- LLMServingSim 吞吐略低于 NeuPIMs：因为实现了更详细的系统级特性（链路通信、同步）

### 3.3 仿真速度对比

| 仿真器 | 单次迭代仿真时间 | 相对 LLMServingSim |
|--------|----------------|-------------------|
| mNPUsim | ~10 小时 | 491× 更慢 |
| GeneSys | ~1.5 小时 | 34.7× 更慢 |
| NeuPIMs | ~2 小时 | 45× 更慢 |
| **LLMServingSim** | **最快** | **基准** |

### 3.4 计算复用效果

| 条件 | 仿真时间 | 加速比 |
|------|---------|-------|
| 无复用优化 | 198.0 ~ 215.7 秒 | 1× |
| 启用复用优化 | 16.3 ~ 33.6 秒 | **6.4× ~ 12.2×** |

### 3.5 可扩展性

- 使用 GPT-3 175B + 2048 NPU（Tensor Parallelism）进行单次迭代仿真
- 仿真时间随 NPU 数量线性增长
- 即使扩展到 **2048 个 NPU**，LLMServingSim 仍能在 **4.13 小时**内完成仿真
- 主要瓶颈：ASTRA-sim 的系统协调和 Graph Converter 处理

### 3.6 并行策略开销

| 并行策略 | 仿真时间（开启复用） |
|---------|------------------|
| Tensor Parallelism (TP) | 最长（同步操作多） |
| Pipeline Parallelism (PP) | 最短 |
| Hybrid Parallelism | 居中 |

---

## 四、亮点与局限

### 4.1 亮点

| 维度 | 评价 |
|------|------|
| **设计创新** | 首次将 LLM 推理服务的"迭代级"动态特性纳入系统仿真设计，与传统训练的"相同迭代"仿真有本质区别 |
| **速度突破** | 34.7×~491× 的加速使原本不可行的系统级探索变得可行 |
| **异构支持** | 插件式架构设计优雅，不局限于 NPU/PIM，理论可接任何加速器 |
| **开源可用** | 代码完整开源，附有 artifact appendix，可复现性优秀 |
| **精度合理** | <15% 的误差率对于系统级仿真而言是可接受的"real2sim"差距 |
| **微架构洞察** | 利用 decoder block 冗余和阶段差异（prefill vs decode）做缓存优化的思路具有普遍借鉴意义 |

### 4.2 局限

| 局限 | 说明 |
|------|------|
| **缺乏 kernel 优化模拟** | 未模拟 FlashAttention 等 GPU kernel 优化，导致高负载下的生成阶段吞吐偏差 |
| **架构覆盖有限** | 当前主要支持传统 decoder-only 架构，对新架构（MoE、多模态、RAG）仅提及但未实现验证 |
| **硬件精度折衷** | 为达到速度优势，在硬件仿真精度上做了妥协，NPU 架构与真实 GPU 存在差异 |
| **单次迭代 vs 端到端** | 论文报告的加速比主要针对单次迭代，端到端服务仿真中迭代数会放大总时间 |
| **图 6 中的偏差** | 生成阶段的吞吐偏差在某些请求密集条件下较大，论文未量化最坏情况误差 |

---

## 五、个人评价

### 5.1 学术价值

LLMServingSim 填补了 LLM 推理服务领域**系统级仿真基础设施**的重要空白。该工作的核心贡献不在于提出新的硬件或软件算法，而在于**设计了一个实用的、可扩展的、足够快速的仿真工具**，让体系结构和系统研究人员能够在合理时间内探索 LLM 推理服务的软硬件设计空间。

与同期工作 **Vidur**（微软，arXiv: 2405.05465，基于 ML 预测和数学建模的 GPU 仿真）相比，LLMServingSim 走的是**更传统的 cycle-level 硬件仿真 + 系统仿真**路线，精度来源更为可信，但速度可能不及纯预测方法。

### 5.2 工程实践价值

对于工业界而言，LLMServingSim 的价值在于：
1. **硬件选型评估**：在部署前仿真不同 NPU/PIM 配比的系统性能
2. **调度策略探索**：测试不同 batching/scheduling 策略的效果
3. **并行策略比较**：对比 TP/PP/Hybrid 在不同模型规模下的表现
4. **容量规划**：评估 KV cache 需求，确定最优内存配置

### 5.3 未来方向

- 支持 **MoE（Mixture of Experts）** 架构的仿真
- 集成 **FlashAttention** 等 kernel 优化模型
- 支持 **Speculative Decoding**、**Continous Batching** 等更先进的服务技术
- 扩展到 **prefill-decode 分离**架构
- 与 **Vidur** 等轻量级仿真器形成互补（精度 vs 速度的帕累托前沿）

### 5.4 评分理由（⭐⭐⭐⭐）

- **创新性**：★★★★☆ — 迭代级仿真的设计巧妙，但对 ASTRA-sim 的依赖降低了独创性
- **完整性**：★★★★★ — 从 motivation 到设计到验证到 artifact 都非常完整
- **实用性**：★★★★☆ — 速度突破显著，但 ONNX 转译和配置复杂度可能限制部分用户
- **影响力**：★★★★☆ — 极有可能成为 LLM 推理系统硬件-软件协同设计领域的基准仿真工具

---

## 相关链接

- [[LLM推理系统深度综述]]
- [[Vidur大规模LLM推理仿真框架]]
- [[ASTRA-sim分布式AI系统仿真器]]
- [[NeuPIMs: NPU-PIM异构加速]]
- [[GeneSys端到端NPU仿真器]]
- [[vLLM: PagedAttention高效内存管理]]
