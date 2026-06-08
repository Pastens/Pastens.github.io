---
tags:
- 论文分析
- llm-inference
- simulation
- hardware-software-co-design
- heterogeneous
- disaggregated-serving
arxiv: 2602.23036
authors:
- Jaehong Cho
- Hyunmin Choi
- Guseul Heo
- Jongse Park
institutions:
- KAIST (Korea Advanced Institute of Science and Technology)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure

## 一、论文概览

### 1.1 基本信息

- **标题**: LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure
- **作者**: Jaehong Cho*, Hyunmin Choi*, Guseul Heo, Jongse Park（*共同第一作者）
- **机构**: KAIST (韩国科学技术院)
- **发表**: arXiv:2602.23036v2, 2026年3月
- **代码**: https://github.com/casys-kaist/LLMServingSim
- **链接**: [arXiv](https://arxiv.org/abs/2602.23036)

### 1.2 核心贡献

LLMServingSim 2.0 是一个统一的系统级 LLM 服务模拟器，专门设计用于对**异构硬件**（多种加速器、PIM、CXL 等）和**解聚化服务架构**（prefill-decode 分离、专家并行、前缀缓存等）进行联合建模。其主要贡献包括：

1. **交互感知建模（Interaction-awareness）**：将服务决策与硬件行为嵌入到单一运行时循环中，显式捕获 batching、routing、placement、offloading 等决策与硬件状态之间的反馈环路。
2. **异构与解聚化的统一建模**：在同一框架内统一表达异构加速器、多层内存系统和各种解聚化服务技术。
3. **运行时驱动的服务动态**：性能由动态请求流、资源竞争和互连效应自然涌现，而非静态配置。
4. **新兴硬件的可扩展性**：通过基于 profile 的算子建模，支持低代价集成新加速器和内存技术。
5. **功耗建模**：内置七组件功耗模型，支持性能和能效的联合评估。

### 1.3 关键数据

| 指标 | 数值 |
|------|------|
| 仿真误差（平均） | **0.95%** |
| GPU 端到端误差 | 0.99% (RTX A6000), 1.54% (H100) |
| 内存使用误差 | 0.93% (单实例), 0.41% (多实例) |
| 功耗误差 | 1.34% (总能耗) |
| TPU 聚合误差 | <0.3% |
| 典型仿真时间 | ~10 分钟（复杂配置） |
| 单算子 Profiling 时间 | ~2.1 小时 (Llama 3.1-70B on H100) |

---

## 二、技术方法详解

### 2.1 系统架构概览

LLMServingSim 2.0 由三大核心组件构成：

```
+-------------------+     +--------------------+     +------------------+
|  Execution        | --> |  Serving Engine    | --> |  输出 Metrics    |
|  Planner          |     |                    |     |                  |
|  (一次性初始化)    |     | +-------------+    |     | - TTFT, TPOT     |
|                   |     | | Request     |    |     | - 吞吐量         |
| - 读取 workload   |     | | Router      |    |     | - 内存使用       |
|   与 cluster 配置  |     | +------+------+    |     | - 功耗/能耗      |
| - 创建 MSG 实例   |     |        |           |     | - 排队延迟       |
| - 分配设备策略     |     | +------v------+    |     +------------------+
| - 初始化 System   | --> | | Model       |    |
|   Simulator       |     | | Serving     |    |
+-------------------+     | | Group (MSG)|    |
                          | | (每个模型)  |    |
                          | +------+------+    |
                          |        |           |
                          | +------v------+    |
                          | | System      |    |
                          | | Simulator   |    |
                          | | (ASTRA-sim  |    |
                          | |  扩展)      |    |
                          | +-------------+    |
                          +--------------------+
```

#### 输入配置

1. **Workload 配置**: 描述 LLM 模型和请求模式（到达率、per-request 执行 trace）
2. **Cluster 配置**: 节点类型和数量、CPU/内存/设备放置、服务策略（路由、并行策略、KV cache 驱逐、offloading 等）
3. **硬件 Profile**: 算子的延迟和功耗 profile

#### 输出指标

- **系统级**: 内存使用量、能耗、吞吐量
- **请求级**: TTFT (Time-to-First-Token)、TPOT (Time-Per-Output-Token)、排队延迟、端到端延迟

### 2.2 基于 Profile 的算子性能建模

这是 LLMServingSim 2.0 的核心效率创新：

- **Operator-level Profiler**：基于 PyTorch/HuggingFace 的 profiling API，只需单设备且几乎无需修改模型代码
- **一次性 profiling**：每个模型-设备对只需测量单个 decode block 的算子级延迟和功耗
- **耗时**: ~2.1 小时（Llama 3.1-70B on H100）
- **profile 可复用**: 收集后可在多次实验间重复使用
- **支持外部模拟器输入**: 可以从硬件模拟器（如 PIM 的行为模型）获取 profile，支持尚无物理实体的新兴硬件评估

### 2.3 Execution Planner（执行规划器）

负责一次性初始化：
- 配置 **Request Router**（请求路由器）
- 为每个模型创建 **Model Serving Group (MSG)**
- 分配设备形成可定制的设备池
- 安装服务策略（并行策略、计算/内存 offloading、KV cache 管理、内存共享）

### 2.4 Model Serving Group (MSG) — 核心执行单元

每个 MSG 是一个逻辑执行单元，负责服务一个 LLM 实例。其内部组件：

#### 设备池 (Device Pool)
- 可包含多种加速器和内存设备：GPU、NPU、CXL 设备、PIM 内存通道
- 支持异构硬件组合

#### 请求队列 (Request Queue)
- 跟踪请求从到达至完成的完整生命周期
- 累积排队延迟、TTFT、TPOT 等统计信息

#### 批调度器 (Batch Scheduler)
- 从队列中选择待处理请求形成可执行 batch
- 评估系统/设备内存容量、KV cache 占用、最大 batch 大小约束
- 考虑前缀缓存驻留和驱逐代价

#### 内存模型 (Memory Model)
- KV cache 管理：驱逐（eviction）和提升（promotion）决策
- 多层内存层级建模：设备内存 → 主机内存 → CXL 内存 → 存储
- 捕获缓存策略、容量约束和数据移动代价
- **显式建模内存共享**以反映集群级效应

#### 功耗模型 (Power Model)
七组件分解，覆盖现代服务器的大部分能耗：

| 组件 | 建模方式 |
|------|----------|
| Accelerators (GPU/NPU/TPU) | 三态模型 (idle/active/standby) |
| CPU | 恒功率 + 利用率相关 |
| DRAM | 与数据传输量成正比 |
| Interconnect Links (含交换机) | 与数据传输量成正比 |
| NIC (网卡) | 恒功率 |
| Storage | 恒功率 |
| Others (主板、散热等) | 恒功率 |

#### 算子映射器 (Operation Mapper)
- 根据并行策略和 offloading 规则将算子分配到设备
- 支持算子粒度的 offloading（例如 attention 放到 PIM，其他放到 GPU）
- 为每个分配添加延迟和功耗估计

#### 算子调度器 (Operation Scheduler)
- 构建执行 DAG（有向无环图）：编码数据依赖、顺序约束、并行策略、通信和内存操作
- DAG 传递给 System Simulator 执行

### 2.5 System Simulator（系统模拟器）

- 执行各 MSG 生成的算子 DAG，评估集群级端到端执行
- 基于 **ASTRA-sim** 和 **Chakra** 的扩展版本
  - 原框架针对训练设计，本工作扩展以支持：
    - 异构计算结构
    - 算子驱动的执行 DAG（捕获 LLM 推理动态）
    - 推理特定并行性（如 expert parallelism）
- 建模同步开销、网络竞争、设备间通信、内存访问延迟
- 增强的内存模型：设备内存 → 主机内存 → 存储 → CXL 内存 + PIM 操作原语

---

## 三、与 LLMServingSim 1.0 对比

| 维度 | LLMServingSim 1.0 | LLMServingSim 2.0 |
|------|-------------------|-------------------|
| **单/多实例** | 仅单实例 | 支持多实例、多模型 |
| **异构硬件** | ❌ 不支持 | ✅ 支持 GPU/NPU/TPU/PIM/CXL 混合 |
| **Prefill-Decode 分离** | ❌ | ✅ 任意 M:N 映射 |
| **Attention/FFN 解聚** | ❌ | ✅ |
| **Pipeline/Tensor 并行** | ✅ (有限) | ✅ 完整支持 |
| **Data Parallelism** | ❌ | ✅ |
| **Expert Parallelism** | ❌ | ✅ 含 MoE 路由策略 |
| **PagedAttention** | ✅ | ✅ |
| **Prefix Caching** | ❌ | ✅ 多层、跨实例共享 |
| **Expert Offloading** | ❌ | ✅ |
| **功耗建模** | ❌ | ✅ 七组件详细模型 |
| **仿真时间** | 较慢 | 大幅优化（~10分钟复杂配置） |
| **仿真误差** | 未提供具体数值 | **0.95%** |
| **ASTRA-sim 集成** | 基础集成 | 深度扩展支持推理 DAG |
| **内存模型** | 简化 | 多层显式建模+带宽竞争 |
| **TPU 支持** | ❌ | ✅ 案例验证 |
| **PIM 支持** | ❌ | ✅ 案例验证 |

### 核心设计差异

1. **从统计收集到预测仿真**：v1.0 主要聚焦于收集运行时统计信息（batch 演化、算子分解），而 v2.0 是完整的端到端性能预测模拟器。

2. **从同构到异构**：v1.0 不支持异构设备组合；v2.0 将设备池作为一等公民，支持任意加速器和内存设备的组合。

3. **从单实例到集群级**：v1.0 限制在单实例环境；v2.0 支持多 MSG、跨节点通信、全局共享缓存。

4. **从简化内存到多层内存**：v1.0 的内存模型较为简单；v2.0 显式建模设备内存、主机内存、CXL 内存、存储四层，并包含带宽竞争和迁移代价。

5. **新增 MoE 支持**：v1.0 不支持 MoE 模型；v2.0 支持完全算子级的 MoE 执行流建模，包括专家路由、并行和 offloading。

6. **新增功耗建模**：v1.0 无功耗建模；v2.0 集成七组件功耗模型，支持三态加速器功耗。

---

## 四、实验评估

### 4.1 实验设置

#### 平台

| 平台 | 硬件配置 | 服务框架 |
|------|----------|----------|
| On-premise GPU | 4× NVIDIA RTX A6000 (40GB, 936GB/s, PCIe 4.0×16) | vLLM |
| Cloud GPU | 8× NVIDIA H100-SXM-80GB (80GB, 3.35TB/s, NVLink) | vLLM |
| Cloud TPU | TPU-v6e-1 (32GB, 1.6TB/s, 800GB/s interconnect) | vLLM-TPU |

#### 模型

| 模型 | 类型 | 平台 | 并行度 |
|------|------|------|--------|
| Llama 3.1-8B | Dense | RTX A6000 / TPU | TP=1 |
| Llama 3.1-70B | Dense | H100 | TP=4 |
| Phi-mini MoE | MoE | RTX A6000 / TPU | - |
| Mixtral 8×7B | MoE | H100 | TP=4 |

#### 数据集
- ShareGPT (300 requests sampled)
- 请求到达：Poisson process, 10 requests/sec

### 4.2 性能验证

#### 吞吐量时间序列

LLMServingSim 2.0 在多种配置（多模型服务、前缀缓存、PD 解聚、MoE 服务）下紧密跟踪真实系统的时间序列吞吐量：

- RTX A6000: 平均逐时步误差 **5.14%**
- H100: 平均逐时步误差 **3.29%**
- **聚合性能误差**: 0.99% (A6000), 1.54% (H100)

#### 功耗验证

使用三脉冲负载（每 60 秒一批 10 请求），LLMServingSim 2.0 精确匹配真实系统的功耗时间序列：
- 总能耗平均误差 **1.34%**
- 正确捕获 active/standby/idle 三种状态转换

#### 内存验证

| 场景 | 误差 |
|------|------|
| 单实例 GPU 前缀缓存 | **0.93%** |
| 多实例 CPU 共享前缀缓存 (LMCache) | **0.41%** |

### 4.3 与其他模拟器的对比

| 模拟器 | 平均误差（简单配置） | 平均误差（复杂配置：PD/ MoE） | 仿真时间 |
|--------|-------------------|---------------------------|----------|
| **LLMServingSim 2.0** | **2.43%** | **1.81%** | ~10 min |
| Vidur | 高（仅部分指标准确） | 不支持复杂配置 | 快 |
| APEX | 高（仅部分指标准确） | 不支持复杂配置 | 中等 |
| TokenSim | 中等（简化建模） | 大误差/不支持 | 快 |
| LLMServingSim v1 | 有限精度 | 不支持 | 慢 |

### 4.4 新兴硬件案例研究

#### TPU-v6e 案例
- 逐时步吞吐量误差 **3.95%**
- 聚合性能误差 **<0.3%**
- 展示了 LLMServingSim 2.0 模拟非 GPU 加速器的能力
- 支持在 TPU 框架尚未实现的场景下进行假设分析（PD 解聚、前缀缓存）

#### PIM (Processing-in-Memory) 案例
- 对比 GPU-only vs GPU+PIM vs GPU+PIM+SBI（sub-batch interleaving）
- GPU+PIM 在 decode 阶段实现 **1.43×** 吞吐量提升
- GPU+PIM 每 token 能耗降低 **14.8%**
- SBI 仅在 batch size ≥256 时有效

---

## 五、亮点与局限

### 亮点

1. **精度卓越**：0.95% 的平均误差在系统级模拟器中极为罕见，几乎达到了实际测量的水平。

2. **覆盖面广**：在一个统一框架中同时支持异构硬件、解聚化服务、MoE、前缀缓存、功耗建模等，是目前功能最全面的 LLM 服务模拟器。

3. **可扩展性强**：通过 profile-based 建模，无需重新设计即可集成新硬件（TPU、PIM），为硬件-软件协同设计提供了理想的平台。

4. **运行时驱动**：区别于静态分析工具，真正捕获了 batching、排队、竞争等动态行为的时间演化效应。

5. **实用性好**：复杂配置也只需约 10 分钟仿真时间，远优于实际部署的成本和复杂度。

6. **代码开源**：代码已在 GitHub 公开，可复现性强。

### 局限

1. **仿真开销高于轻量级模拟器**：如 Vidur 和 TokenSim 更快，但这是精度和覆盖面的必然代价。

2. **Profiling 需要物理硬件**：虽然单次 profiling 可复用，但对于尚无物理实体的新兴硬件，需要使用外部模拟器提供 profile，增加了依赖链。

3. **TPU 验证有限**：由于 vLLM-TPU 框架的限制，TPU 验证仅覆盖单实例 dense serving，多实例/解聚场景尚未验证。

4. **网络建模依赖 ASTRA-sim 扩展**：虽然继承了大量已有工作，但通信模型的精度仍取决于底层 ASTRA-sim 的保真度。

5. **未覆盖训练**：专注于推理服务场景，不支持训练流程的模拟。

6. **功耗模型简化**：虽然三态加速器模型捕获了主要趋势，但可能无法精确建模更复杂的功耗管理机制（如 DVFS）。

---

## 六、个人评价

### 6.1 论文定位

LLMServingSim 2.0 是 LLM 服务系统模拟领域的里程碑式工作。如果说 Vidur 展示了"如何快速估算吞吐量"，APEX 展示了"如何自动搜索最优配置"，那么 LLMServingSim 2.0 回答的是更根本的问题——**"当异构硬件遇上解聚化服务，系统行为到底如何涌现？"**

这是一个典型的系统研究问题：组件之间交互的复杂性远超各部分的简单加和。作者敏锐地捕捉到，现代 LLM 服务基础设施正在经历从**同构、单体**到**异构、解聚**的根本转变，而现有工具无法应对这一转变带来的联合建模需求。

### 6.2 技术评价

**最令人印象深刻的是 0.95% 的误差**。这个数字在系统级模拟器中极其罕见。实现这种精度的关键是：

1. **算子级 profile 而非 kernel 级**：在保真度和开销之间找到了巧妙的平衡
2. **多层内存的显式建模**：KV cache 行为是 LLM 服务性能的核心决定因素，v2.0 对此的建模远优于前代和其他工具
3. **运行时循环而非静态分析**：避免了离线分析的累积误差

相较于 v1.0，这几乎是一个**从零开始的重写**——功能覆盖度从约 30% 提升到约 95% 的现代服务技术，同时仿真速度反而更快。这是真正的"做了更难的事，反而做得更好"。

### 6.3 影响力评估

该工作对以下领域具有重要价值：

1. **硬件架构师**：可以快速评估新加速器（如 NPU、PIM、CXL 设备）在真实服务负载下的收益
2. **系统研究者**：可以探索新的调度策略、缓存策略、解聚化方案而不需要部署大规模集群
3. **学术社区**：提供了一个可复用的基准平台，有望成为 LLM 服务系统研究的标准评估工具

### 6.4 关注点

一个值得关注的设计选择是最小化 profile 成本的策略——用一个 decode block 代表整个模型的性能特征。这在层内同构的模型中合理，但在层间异构（如不同 layer 使用不同 attention 变体）的模型中可能存在偏差。此外，异步执行和内存访问重叠等底层硬件优化未被显式建模，可能在某些场景下引入误差。

### 6.5 评分: ⭐⭐⭐⭐⭐

**理由**:
- **技术贡献扎实**（1 分）：从问题定义到解决方案到验证，完整闭环
- **实用性极高**（1 分）：开源 + 10 分钟仿真 + 精确到 1% 以内
- **填补空白**（1 分）：是首个统一建模异构硬件的解聚化 LLM 服务模拟器
- **可扩展性强**（1 分）：profile-based 设计天然支持新兴硬件
- **超越前作**（1 分）：v2.0 相对 v1.0 的改进幅度是革命性的

---

## 相关链接

- [[LLM推理系统深度综述]]
- [[LLMServingSim 1.0 论文分析]]（原始版本）
- [[Vidur 论文分析]]（同类系统模拟器）
- [[TokenSim 论文分析]]
- [[APEX 论文分析]]
- [[ASTRA-sim 2.0]]（本工作的底层通信模拟基础）
- [[NeuPIMs]]（本工作的 PIM 案例研究参考）
