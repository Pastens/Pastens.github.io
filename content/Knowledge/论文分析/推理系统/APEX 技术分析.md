---
tags:
- 论文分析
- llm-inference
- simulation
- parallel-strategy
- llm-serving
arxiv: '2411.17651'
authors:
- Yi-Chien Lin
- Woosuk Kwon
- Ronald Pineda
- Fanny Nina Paravecino
institutions:
- University of Southern California (USC)
- University of California, Berkeley
- University of California, Los Angeles (UCLA)
- Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# APEX: 面向 LLM 服务的可扩展、感知动态性的自动化并行执行模拟器

> **APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving**
>
> 论文链接: https://arxiv.org/abs/2411.17651
> 代码仓库: https://github.com/microsoft/apex_plus

---

## 一、论文概览

| 项目 | 内容 |
|------|------|
| **标题** | APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving |
| **作者** | Yi-Chien Lin (USC), Woosuk Kwon (UC Berkeley), Ronald Pineda (UCLA), Fanny Nina Paravecino (Microsoft) |
| **机构** | 南加州大学、UC Berkeley、UCLA、微软 |
| **发表** | arXiv:2411.17651v2 (2024-11-26 初版, 2025-04-29 修订) |
| **领域** | cs.DC (分布式、并行与集群计算) |
| **代码** | https://github.com/microsoft/apex_plus |
| **评分** | ⭐⭐⭐⭐ (4/5) |

### 核心贡献

1. **自动化并行执行计划搜索**：APEX 是一个 LLM 服务模拟器，能够根据给定的 LLM、设备集群和输入请求，自动找出最优的并行执行方案（DP、PP、TP 的组合）。
2. **动态感知模拟**：APEX 模拟了 iteration-level batching（逐迭代批处理）的运行时行为，捕捉了真实 LLM 服务中动态变化的 batch 特性。
3. **高精度 & 高效**：平均相对误差仅 10.7%；发现的方案比启发式方案最高快 3.37×，比延迟最优方案节能最高 45%。
4. **广泛兼容 & 可扩展**：支持多种 LLM、并行策略、量化格式和设备集群（包括 GPU 和 ASIC/TPU），新增模型/设备/并行策略仅需 0-200 行代码。
5. **极致性价比**：在 CPU 上 15 分钟内找到最优方案，比实际 GPU 部署快 71×、成本低 1234×。
6. **综合评估指标**：报告 TPOT、TTFT、P95 延迟、MFU、MBU 等关键 Serving 指标，帮助服务提供商满足 SLO。

---

## 二、技术方法详解

### 2.1 背景与动机

LLM 推理包含两个阶段：
- **Prefill（预填充）阶段**：并行处理整个 prompt，计算密集型（compute-bound）。
- **Decode（解码）阶段**：逐 token 自回归生成，内存密集型（memory-bound）。

LLM 服务面临三个独特挑战：
1. **Iteration-level batching 的动态性**：请求可被持续加入当前正在处理中的 batch，导致 batch 大小和阶段混合情况动态变化，传统静态批处理建模方法不适用。
2. **指数级增长的搜索空间**：模型大小和设备数量的增加使搜索空间爆炸。
3. **快速演进系统的适配**：模型架构、硬件、量化方法、并行策略不断更新，模拟器需具备良好可扩展性。

### 2.2 APEX 系统架构

整体工作流（论文 Figure 2）：

```
用户输入：
  - LLM 配置
  - 设备集群配置
  - 请求 trace（到达时间、上下文长度、生成长度）
        ↓
并行执行计划生成器 (Section 3.2)
  - 转换 LLM 为 Transformer IR
  - 基于 Parallel Templates 生成多种并行方案
  - Device Mapper 将逻辑设备映射到物理设备
        ↓
批处理模块 (Section 3.3)
  - 模拟 iteration-level batching
  - 追踪内存使用和请求到达，动态维护活跃请求列表
        ↓
LLM 服务模拟器 (Section 3.4)
  - 根据活跃请求 + 预采集的 profiling 结果
  - 估算每轮迭代的执行时间和能耗
  - 利用 Transformer IR 的重复结构，只模拟单个 block 再外推
        ↓
输出：每个并行执行计划的模拟报告
  - E2E Latency, TTFT, TPOT, P95, Energy, MFU, MBU
```

### 2.3 Transformer IR（中间表示）

**核心思想**：将 LLM 抽象为规范化的中间表示，利用 Transformer 的重复结构大幅减少搜索空间。

- **Cell**：对应关键 Transformer 操作，如 MHA（Multi-Head Attention）、MLP（Multi-Layer Perceptron）、GQA（Group Query Attention）、SwiGLU 等。
- **Task**：Cell 内的子任务，如单个 attention head、MoE 中的 expert。
- **Block**：最小的非重复相邻 Cell 集合，通常就是一个完整的 Transformer layer。

通过 Transformer IR，一个 LLM 被统一表示为多个相同的 Block，每个 Block 由 Cell 链组成。模拟时只需运行一个 Block 的计算，然后外推到整个模型，大幅降低仿真开销。

### 2.4 并行执行计划生成器 (Section 3.2)

#### 2.4.1 Parallel Templates

APEX 为每种 Cell 类型预定义并行模板，指定该 Cell 如何在多设备间并行化。模板关键特性：

- **Cell 内并行**：Tensor Parallelism（TP）将 attention head / FFN 中间层切分到多设备；Expert Parallelism（EP）将 MoE expert 分布到多设备；Data Parallelism（DP）复制 Cell。
- **Cell 间通信**：自动插入 AllReduce、All-to-All、AllGather 等集合通信操作用于张量重分片（resharding）。

#### 2.4.2 分层搜索算法 (Algorithm 1)

采用层次化、自顶向下的搜索方式：

1. **Model-level DP**：选择模型级数据并行的 degree（1, 2, ..., n），等分设备集群。
2. **Pipeline Stages**：在每个 replica 内确定 pipeline stage 数量，等分设备。
3. **Cell-level DP**：对每个 Cell 选择 cell 级数据并行的 degree。
4. **Intra-layer parallelism**：若 Cell 分配的设备数 > DP degree，则应用 TP/EP。
5. **Insert collectives**：相邻 Cell 间插入重分片通信操作。

> 通过约束并行度只取设备总数的**约数**，有效控制搜索空间。

#### 2.4.3 Device Mapper（设备映射器）

采用**自底向上**的方式将逻辑设备映射到物理设备：
- 高通信需求的逻辑设备（如同 TP cell 内设备）优先映射到低层级、高带宽的物理连接（如 NVLink）。
- 低通信需求的设备映射到更高层级（如跨节点 InfiniBand）。
- 有效利用树状网络拓扑的层次带宽特性。

### 2.5 批处理模块 (Section 3.3)

模拟 **iteration-level batching**（即 continuous batching）：
- 维护活跃请求列表（active request list），记录每个请求的已生成 token 数。
- 在每个迭代中，检查是否有新请求到达且内存充足 → 加入 batch。
- 当请求达到生成长度 → 从 batch 移除，释放 KV cache 内存。
- **内存溢出处理**：贪心策略可能超出内存容量，此时最新加入的请求被临时移除，优先保证已完成请求。

### 2.6 LLM 服务模拟器 (Section 3.4)

- **Prefill**：用完整 context length 查询 profiling 结果。
- **Decode**：有效 context length = 1，若有 n 个 decode 请求则用 context length = n（所有 decode 请求并行处理）。
- **时间估算**：取所有 pipeline stage 的最大值（流水线中最慢的 stage 决定整体延迟）。
- **能耗估算**：对所有 stage 求和（所有设备都消耗能量）。
- **关键指标**：TTFT, TPOT, P95 Latency, MFU (Model FLOPs Utilization), MBU (Model Bandwidth Utilization)。

### 2.7 Profiling 机制 (Section 3.5)

- **Operation-level profiling**：测量关键 Transformer 操作（MHA、GEMM）在各种配置下的执行时间和能耗。
- **通信 profiling**：测量 AllReduce、ReduceScatter 等集合操作在不同数据量、设备数、节点数下的开销。
- **一次 profiling，多次复用**：迁移到新集群后只需 profiling 一次，后续可被所有模拟任务摊销。
- 缺失数据点通过**线性插值**估算。

---

## 三、实验评估

### 3.1 实验设置

| 项 | 详情 |
|----|------|
| **模型** | Llama-3.1-70B, Llama-3.1-405B, Mistral-Large-Instruct (123B), Mixtral 8x22B-Instruct (MoE) |
| **量化** | FP16 默认, FP8 (W8A8) |
| **数据集** | Summarization（长上下文短生成）、Creation（短上下文长生成）、Chat（实际对话） |
| **集群** | 单节点 8×H100 (80GB)、双节点 16×H100、单节点 8×H200 (141GB) |
| **对比系统** | vLLM v0.6.0、SGLang v0.4.5 |
| **基线方案** | 启发式：TP 在节点内、PP 跨节点 |

### 3.2 主要实验结果

#### 3.2.1 端到端延迟对比 (Table 2, 单节点 H100)

| 模型 | Trace | Arrival Rate | Baseline | Feasible Optimal | APEX Optimal |
|------|-------|:-----------:|:--------:|:----------------:|:------------:|
| Llama-3.1-70B | Summarization | 0.25 | 1998.82s (1×) | 1340.99s (1.49×) | **1175.09s (1.70×)** |
| Llama-3.1-70B | Creation | 0.25 | 4027.70s (1×) | 2301.54s (1.75×) | **1945.75s (2.07×)** |
| Llama-3.1-70B | Chat | 0.25 | 1622.48s (1×) | 1118.85s (1.45×) | **824.43s (1.97×)** |
| Mistral-Large | Creation | 0.25 | 3561.55s (1×) | 2268.50s (1.57×) | **1496.45s (2.38×)** |
| Mixtral-8x22B | Chat | 0.25 | 1938.26s (1×) | 1658.22s (1.17×) | **575.12s (3.37×)** |
| Mixtral-8x22B | Summarization | 0.25 | 2005.06s (1×) | 1753.79s (1.14×) | **859.83s (2.33×)** |

**关键发现**：
- Feasible Optimal（限于当前系统支持的并行策略）最高提升 1.75×。
- APEX Optimal（支持 Cell-level DP 等高级策略）最高提升 3.37×。
- 引入 DP（数据并行）经常带来显著收益——现有系统通常因内存开销而忽略 DP。
- Mixtral-8x22B (141B MoE) 的 Feasible Optimal 在 Creation trace 上提升有限，因为内存瓶颈占主导。

#### 3.2.2 多节点与 H200 集群 (Table 3)

| 模型 & 集群 | Trace | Baseline | APEX Optimal |
|------------|-------|:--------:|:------------:|
| Llama-3.1-405B (2节点 H100) | Creation | 2314.25s | **1229.14s (1.88×)** |
| Llama-3.1-70B (H200) | Chat | 1189.00s | **536.74s (2.22×)** |
| Mixtral-8x22B (H200) | Chat | 1380.76s | **500.32s (2.76×)** |

- 在 2 节点 16×H100 上部署 405B 模型，APEX 仍能稳定找到更优方案。
- H200（141GB vs 80GB HBM3）更大内存允许更灵活的策略选择，如 4-way DP 替代 2-way DP。

#### 3.2.3 能耗优化 (Table 4)

| Trace | 方案 | 频率 | 能耗 | TTFT | TPOT |
|-------|------|:---:|:----:|:----:|:----:|
| Summarization | 延迟最优 | 2.0 GHz | 4.095 KJ | 788.67ms | 32.06ms |
| Summarization | 能耗最优 | 2.0 GHz | **3.293 KJ (↓19%)** | 992.69ms | 26.89ms |
| Summarization | 能耗最优 | 0.8 GHz | **2.265 KJ (↓45%)** | 1485.94ms | 33.62ms |
| Creation | 能耗最优 | 0.8 GHz | **5.016 KJ (↓41%)** | 75.70ms | 34.59ms |

- 能耗最优方案比延迟最优方案节能最高 19%。
- 结合降低 GPU 频率（0.8 GHz），可节能 45%，而 TTFT/TPOT 仍在可接受范围内。

### 3.3 模拟保真度验证 (Section 4.3)

- **速度预测精度**：APEX 预测的 Feasible Optimal 加速比与实际加速比的**平均相对误差仅 10.7%**。
- **主要误差来源**：Mixtral-8x22B 上 APEX 预期 EP > TP，但实际 SGLang 中 TP 实现更优化。
- **可扩展性预测**：在 2→8 GPU 的 TPOT 趋势预测中，APEX 与实际结果高度一致。

### 3.4 效率评估 (Section 4.4)

| 指标 | APEX (CPU) | 实际 GPU 部署 | 提升 |
|------|:----------:|:-------------:|:----:|
| 执行所有方案 | < 2.5 小时 | ~160 GPU 小时 | **71× 更快** |
| 成本 | ~$7.20 | ~$8,889 | **1234× 更便宜** |
| Profiling（一次性） | — | ~40 GPU 小时 | 可摊销 |

- 从 32B 扩展到万亿参数模型，模拟开销基本保持不变（Figure 8），得益于 Transformer IR 的抽象。

### 3.5 可扩展性评估 (Section 4.5)

| 扩展类型 | 代码量 (LoC) | 实现时间 |
|----------|:-----------:|:--------:|
| 新 LLM（标准架构） | 0 | ~0 |
| 新 LLM（含未知 Cell） | 50-150 | 1-2 小时 |
| 新设备集群 | ~20 | 6-8 小时 |
| 新批处理机制 | ~100 | 1-2 小时 |
| 新并行策略 | 50-200 | 1-2 小时 |

### 3.6 超 SLO 优化 (Section 4.6)

APEX 可用于辅助满足 SLO。例如通过模拟不同 max batch size 约束下的 TPOT 变化，帮助服务提供商选择合适的 batch 上限：将 batch size 从 16 降到 8，可提升 TPOT 约 14-18%。

---

## 四、亮点与局限

### 亮点

1. **动态感知是核心创新**：首次在并行策略搜索中系统性地建模 iteration-level batching 的动态特性，这比以往仅考虑静态工作负载的工具有本质提升。
2. **Transformer IR + 重复结构利用**：通过规范化中间表示和 Block 抽象，在保证精度的同时大幅缩小搜索空间，使其能扩展到万亿参数模型。
3. **综合指标体系**：不仅报告延迟，还覆盖能源、TTFT、TPOT、P95、MFU/MBU 等全面指标，支持多目标优化。
4. **操作级 Profiling**：无需为每个模型单独 profiling，只需 profiling 一次关键算子即可适用于所有 Transformer 模型。
5. **极高的性价比**：CPU 上 15 分钟 vs GPU 集群上数天的成本差异极具说服力。
6. **设计良好的模块化架构**：支持轻松扩展新模型、新硬件、新策略，代码量极小。

### 局限

1. **搜索空间仍受限于模板**：Parallel Templates 是预定义的，可能遗漏一些非常规的并行化策略。
2. **能耗建模粒度粗**：能耗估算基于 operation-level profiling 的线性组合，未建模芯片级动态功耗行为（如时钟频率切换开销）。
3. **仅支持 Transformer 架构**：不适用于非 Transformer 模型（如状态空间模型 Mamba 等）。
4. **未覆盖推理优化技术**：未考虑 KV cache 压缩、speculative decoding、prompt cache 等高级优化对并行策略选择的影响。
5. **EP vs TP 预测偏差**：在 MoE 模型上 EP 的预测与实际有差距，反映实现层面的优化差异难以建模。
6. **不支持多目标自动权衡**：虽然报告中可查看多指标，但选择机制仍是单目标优化（如最小化延迟或最小化能耗）。

---

## 五、个人评价

### 学术贡献

APEX 填补了 LLM 推理服务系统中一个重要的空白：**如何自动化搜索最优并行策略**。与已有的 Calculon（训练场景）、Vidur（静态分析）、LLMServingSim（NPU 专用）等工作相比，APEX 在以下方面实现了明确进步：

1. **场景针对性**：专门针对 serving 场景的 dynamics（iteration-level batching）。
2. **综合度**：覆盖 DP + PP + TP + EP 的混合并行，支持量化、多节点、多种硬件。
3. **实用性**：15 分钟 CPU 搜索 vs 数天 GPU 部署的成本差异，使"先仿真再部署"成为可行实践。

### 实践价值

对于实际部署 LLM 服务的团队，APEX 提供了一个极具吸引力的 pipeline：
1. 在目标集群上做一次 operation-level profiling（~40 GPU 小时）。
2. 根据实际请求 trace，用 APEX 在 CPU 上快速仿真（~15 分钟）。
3. 选择最优并行方案，直接配置到 vLLM / SGLang / TensorRT-LLM 中。

这种方法比纯启发式（如"TP inside node + PP across node"）有数倍提升潜力，且成本极低。

### 与相关工作对比

| 工具 | 场景 | 动态感知 | 混合并行 | 能耗 | 可扩展性 | 代码量 |
|------|:----:|:--------:|:--------:|:----:|:--------:|:------:|
| **APEX** | Serving | ✅ | DP+PP+TP+EP | ✅ | ✅ 万亿级 | 开源 |
| Vidur [1] | Serving | ❌ | DP+PP+TP | ❌ | ❌ | 开源 |
| LLMServingSim [7] | Serving (NPU) | ❌ | PP+TP | ❌ | ❌ | 开源 |
| Calculon [20] | Training | ❌ | DP+PP+TP | ❌ | ✅ | — |
| DynamoLLM [42] | Serving | ❌ | 仅 TP | ✅ | ❌ | — |
| LLM-Pilot [25] | Serving | ❌ | 仅 TP | ❌ | ❌ | — |

### 改进建议

1. **引入学习型搜索**：当前是枚举+模拟的 brute-force 方式，可引入贝叶斯优化或强化学习加速搜索。
2. **支持计算-通信 overlap 建模**：当前模拟假设计算和通信串行，真实系统中常有 overlap。
3. **动态重配置支持**：扩展到在线场景，支持根据负载变化动态调整并行策略。
4. **多目标优化界面**：提供 Pareto 前沿可视化，帮助用户在延迟、吞吐、能效间做 trade-off。
5. **更细粒度的能耗模型**：集成 RAPL / NVML 级别的功率追踪数据。

---

## 相关链接

- [[Knowledge/论文分析/推理系统/LLM推理系统深度综述]]

---
*分析创建: 2026-05-11 | 论文版本: v2 (2025-04-29)*
