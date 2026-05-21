---
tags:
- 论文分析
- llm-inference
- system-design
- pd-disaggregation
- kv-cache
- scheduling
- serving-system
arxiv: '2407.00079'
authors:
- Ruoyu Qin
- Zheming Li
- Weiran He
- Mingxing Zhang
- Yongwei Wu
- Weimin Zheng
- Xinran Xu
institutions:
- Moonshot AI (月之暗面)
- Tsinghua University (清华大学 MadSys Lab)
venue: FAST 2025 Best Paper
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
github: kvcache-ai/Mooncake (5294 stars)
project: Kimi (kimi.moonshot.cn)
permalink: mooncake
---

# Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving
## Mooncake：以 KVCache 为中心的大模型服务解构架构

## 一、论文概览

| 项目            | 内容                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **论文标题**      | Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving                                                     |
| **作者**        | Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu                                      |
| **机构**        | Moonshot AI（月之暗面）+ 清华大学 MadSys Lab                                                                                         |
| **发表会议**      | USENIX FAST 2025 **Best Paper Award**                                                                                      |
| **arXiv 版本**  | 2407.00079v4 (2025年9月3日更新)                                                                                                 |
| **GitHub 仓库** | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — ★ 5,294 stars / 731 forks                                  |
| **所属项目**      | Kimi Chat — 月之暗面旗下大模型对话服务 (kimi.moonshot.cn)                                                                               |
| **核心贡献**      | 以 KVCache 为中心的预填-解码分离解构架构，用于大规模 LLM 在线服务                                                                                   |
| **关键词**       | PD Disaggregation, KVCache Centeric Scheduling, Overload-oriented Scheduling, Chunked Pipeline Parallelism, Prefix Caching |
| **读后整体评价**    | ★★★★★ — 工业界生产级系统的经典之作，技术深度与工程实践兼备，FAST 2025 最佳论文实至名归                                                                       |

### 摘要

Mooncake 是月之暗面（Moonshot AI）为其大模型服务 Kimi 构建的生产级推理服务平台。其核心设计是一个 **以 KVCache 为中心的解构架构**：将预填（Prefill）和解码（Decoding）集群分离，并利用 GPU 集群中未被充分利用的 CPU、DRAM 和 SSD 资源构建分布式的 KVCache 缓存池。系统的中心是一个 KVCache 感知的全局调度器（Conductor），它在最大化有效吞吐量的同时保证延迟相关的服务等级目标（SLO）。与假设所有请求都会被处理的传统研究不同，Mooncake 需要应对 **严重过载（overloaded）** 的场景，为此论文提出了一种基于预测的提前拒绝策略。实验表明：在长上下文场景下，Mooncake 相比基线最多可实现 **525% 的吞吐量提升**；在实际工作负载下，Kimi 可多处理 **75% 的请求**。

### 研究背景与动机

随着大语言模型在各场景中的快速普及，LLM 推理服务的工作负载变得高度多样化。作为 MaaS（Model as a Service）提供商，月之暗面的核心目标是：**在满足多层级 SLO 约束的前提下，最大化有效吞吐量（goodput）**。GPU 集群的各类资源需要被充分且高效地利用。

关键洞察在于以下几点：

1. **Prefill 与 Decoding 计算特性迥异** — Prefill 是计算密集型（并行处理所有输入 token），随序列长度超线性增长；Decoding 是内存密集型（逐 token 自回归生成），随 batch size 亚线性增长。将两者耦合在同一节点上会导致严重的相互干扰。

2. **KVCache 是 LLM 推理调度的核心** — KVCache 伴随着请求从 Prefill 流向 Decoding，其调度效率直接影响整体系统吞吐与延迟。

3. **过载是常态而非异常** — 快速增长的用户请求远超 GPU 供应增速，MaaS 提供商面临持续的过载问题，这需要全新的调度策略。

## 二、技术方法详解

### 2.1 系统架构总览

Mooncake 的架构由以下核心组件构成：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Conductor (全局调度器)                       │
│           以 KVCache 为中心，负责请求分发与资源调度                   │
└─────────────────────────────────────────────────────────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
   ┌──────────┐      ┌──────────┐         ┌──────────────┐
   │ Prefill  │◄────►│ Decoding │         │ 分布式       │
   │ Instance │      │ Instance │         │ KVCache Pool │
   │   Pool   │      │   Pool   │         │ (CPU/DRAM/SSD)│
   └──────────┘      └──────────┘         └──────────────┘
         │                   │                    │
         └───────────────────┴────────────────────┘
                     RDMA 高速互联
```

- **Prefill Instance Pool（预填实例池）**：负责处理前向计算，生成 KVCache
- **Decoding Instance Pool（解码实例池）**：负责自回归生成，读取 KVCache
- **Distributed KVCache Pool（分布式 KVCache 缓存池）**：用 GPU 节点的 CPU/DRAM/SSD 构建大规模缓存层
- **Conductor（统筹器）**：全局调度中枢，调度请求、管理 KVCache 的复制与换入换出
- **Messenger**：部署在每个节点上的独立进程，负责基于 RDMA（GPUDirect）的高速跨机 KVCache 传输

### 2.2 KVCache 存储与复用机制

Mooncake 的 KVCache 以分页块（paged block）形式存储在 CPU 内存中，采用基于哈希的**前缀去重**机制：

- 每个 token block（512 token 为一块）与其所有前缀 block 共同计算一个哈希值
- 哈希链结构允许快速识别可复用的前缀缓存：
  ```
  A = Hash(a)
  B = Hash(A + b)    # 包含前缀 A 的哈希
  C = Hash(B + c)    # 包含前缀 B 的哈希
  ...
  ```
- 如果请求的哈希序列与前缀缓存完全匹配，则对应 block 可以直接复用，无需重新计算
- 缓存淘汰策略支持 LRU、LFU 或基于请求特征的定制策略

论文还提供了**首个开源的带有缓存复用关系的大模型服务请求 trace**（23,608 条记录），包含 timestamp、input_length、output_length 和 hash_ids 字段。分析显示：
- 平均输入长度 7,590 token，平均输出长度 182 token，输入输出比约 720:1
- LRUCache 在该工作负载下表现最佳
- 超过 50% 的缓存 block 访问频率为零，而部分热点 block 被访问数万次

### 2.3 Prefill 池的设计

#### 2.3.1 多节点预填（Multi-node Prefill）

面对从 8K 到 128K 甚至 1M token 的日益增长的长上下文需求，Mooncake 提出了 **Chunked Pipeline Parallelism（CPP，分块流水线并行）**：

- 将输入 token 按 `prefill_chunk` 大小切分成多个 chunk
- 多个节点以流水线方式并行处理不同 chunk
- 相比 Sequence Parallelism（SP）方案：
  - ✅ 仅需在流水线边界跨节点通信，可与计算重叠
  - ✅ 天然适配长短上下文，无需频繁动态调整节点分组
  - ✅ 减少网络资源竞争（与 KVCache 传输争用带宽）

CPP 是首次将训练中的流水线并行思想系统性地应用于推理场景。

#### 2.3.2 逐层预填（Layer-wise Prefill）

Mooncake 利用 Transformer 逐层计算的特性，将 KVCache 的加载和存储与计算重叠：

- **加载**：在每层 Attention 计算开始前，等待该层 KVCache 异步加载完成，同时触发下一层的异步加载
- **存储**：Attention 计算完成后，立即启动该层 KVCache 的异步存储

这使得：
- KVCache 在 GPU VRAM 上的占驻时间（occupation cost = S × T）大幅降低
- Prefill 调度可以忽略 VRAM 大小限制（只要能容纳单个请求即可）
- 空闲的 VRAM 可用于处理 batch API 等对延迟不敏感的任务

### 2.4 KVCache 中心的调度算法（Conductor）

#### 2.4.1 Cache-Aware Prefill Scheduling（缓存感知的预填调度）

**核心思想**：Prefill 实例的选择不只是看负载，还要考虑**前缀缓存命中长度**和**可复用 KVCache 的分布**。

**算法流程** (Algorithm 1)：

1. 对请求的输入 token 分块并计算哈希键
2. 针对每个 Prefill 实例，计算前缀匹配长度（prefix_len），估算预填执行时间
3. 加上排队等待时间，获得该实例上的预估 TTFT
4. 选择 TTFT 最短的实例
5. 如果所有实例都无法满足 SLO，返回 HTTP 429（Too Many Requests）

**关键工程实现**：
- **预填时间预测模型**：基于离线测试数据构建，利用 Transformer 计算模式正则性，误差很小
- **排队时间估算**：聚合所有排队请求的预填时间
- **传输时间预测**：考虑数据大小和当前网络拥塞状态

#### 2.4.2 缓存负载平衡（Cache Load Balancing）

面对 KVCache 访问严重不平衡的问题（系统 prompt 被频繁访问 vs. 用户文档仅被一人使用），Mooncake 采用**基于启发式的热点自动迁移**策略：

- Conductor 不一定把请求路由到缓存命中最长的实例；如迁移开销小于额外计算开销，则路由到负载较轻的实例
- 该实例主动从持有节点拉取 KVCache 并**本地缓存**
- 阈值（`kvcache_balancing_threshold`）决定是"计算"还是"迁移"
- 这自然地实现了热点 KVCache 的自动复制与扩散

**实验验证**（8 Prefill + 8 Decoding 实例）：
| 调度策略 | 平均 TTFT |
|---------|----------|
| Random | 92.07s |
| Load-balancing | 60.41s |
| Cache-aware | 14.36s |
| **KVCache-centric (Cache + Balance)** | **6.26s** |

### 2.5 过载场景调度（Overload-oriented Scheduling）

这是 Mooncake 与传统 LLM 推理系统设计最大的区别所在。

#### 2.5.1 提前拒绝策略（Early Rejection）

**问题**：在解构架构中，请求先经过 Prefill 再进入 Decoding。如果在 Decoding 阶段才拒绝（因负载过高），Prefill 阶段的计算资源就完全浪费了。

**解决方案**：在 Prefill 开始前，就综合评估 Prefill 和 Decoding 的负载状况。如果 Decoding 池也无法接受，则提前拒绝请求。

#### 2.5.2 负载波动问题

然而，Early Rejection 引入了一个新问题：**Prefill 和 Decoding 负载之间出现反相波动（anti-phase fluctuation）**。

**产生机制**（四阶段循环）：
1. 两阶段负载都低 → Conductor 大量接受请求
2. Prefill 满负荷 → 大量请求涌入 Decoding → Decoding 负载飙升
3. Conductor 拒绝新请求 → Prefill 负载降低 → Decoding 逐步清空
4. Decoding 变空闲 → Conductor 再次大量接受 → 回到阶段1

这种波动导致资源利用率严重下降。

#### 2.5.3 基于预测的提前拒绝（Prediction-based Early Rejection）

**核心思路**：预测请求经过 Prefill 阶段后 Decoding 池的负载状况，而不是基于当前 Decoding 负载做决策。

**系统级预测策略**（当前实现）：
- 假设每个请求的解码阶段使用固定时间 td
- 对时刻 t，计算 Prefill 已完成并将进入 Decoding 的请求
- 移除在 t 之前已完成（执行时间超过 td）的请求
- 计算所有 Decoding 实例的平均 TBT 与 SLO 比值

**实验结果**（8 Prefill + 8 Decoding 实例，2x 回放速度模拟过载）：
| 策略 | 拒绝请求数 |
|------|-----------|
| Baseline | 4,183 |
| Early Rejection | 3,771 |
| **Early Rejection based on Prediction** | **3,589** |

## 三、实验评估

### 3.1 实验设置

- **硬件**：8× NVIDIA A800-SXM4-80GB GPU 节点，NVLINK 互联，RDMA 800Gbps
- **模型**：基于 LLaMA2-70B 架构的 dummy 模型
- **数据集**：

| 数据集 | 平均输入长度 | 平均输出长度 | 缓存命中率 | 到达模式 |
|--------|------------|------------|-----------|---------|
| ArXiv Summarization | 8,088 | 229 | ~0% | Poisson |
| L-Eval | 19,019 | 72 | >80% | Poisson |
| Simulated Data | 16K/32K/64K/128K | 512 | 50% | Poisson |
| Real Workload | 7,955 | 194 | ~50% | Timestamp-based |

- **基线**：vLLM（SOTA 开源 LLM Serving 系统）
- **SLO 指标**：TTFT P90 ≤ 10× baseline, TBT P90 ≤ 5× baseline

### 3.2 端到端性能

#### 公共数据集

| 配置 | ArXiv Summarization 吞吐提升 | L-Eval 吞吐提升 |
|------|---------------------------|---------------|
| Mooncake [3P+1D] vs vLLM [4M] | **+20%** | **+40%** |
| Mooncake [2P+2D] vs vLLM [4M] | 相近 | 与比例相关 |

L-Eval 上的更大提升得益于 prefix caching（>80% 缓存命中率）。

#### 模拟长上下文数据

| 输入长度 | Mooncake v.s. vLLM 吞吐提升 |
|---------|---------------------------|
| 16K | **+50%** |
| 32K | **+150%** |
| 64K | **+300%** |
| 128K | **+525%** |

长上下文场景下，vLLM 的解码阶段被预填阶段严重干扰，不得不退化为单请求处理。Mooncake 的解构设计使其解码阶段不受预填影响，TBT SLO 始终满足。

#### 真实工作负载

| 指标 | Mooncake [10P+10D] | vLLM [20M] |
|------|-------------------|------------|
| TTFT SLO 达标率 | ~100% | ~100% |
| TBT SLO 达标率 | **~100%** | **57%** |
| 总处理请求数 | **+75%** | 基线 |

## 四、亮点与局限

### 主要亮点 ✅

1. **生产级验证** — Mooncake 是 Kimi 的实际生产推理平台，经历了指数级的用户增长挑战，证明了其可扩展性和鲁棒性。这远比纯学术仿真实验更有说服力。

2. **KVCache 中心的创新设计** — 将 KVCache 提升到系统设计的"第一公民"地位，围绕其进行资源调度与优化，统一了缓存管理、请求调度和负载均衡三个维度。

3. **Chunked Pipeline Parallelism (CPP)** — 针对长上下文预填的创新方案，避免了 Sequence Parallelism 的高通信开销和弹性调度的复杂性，天然适配生产环境。

4. **过载场景调度的开拓性工作** — 首次系统性地讨论了 LLM 推理服务中的过载问题，提出了 Early Rejection、负载波动分析、基于预测的拒绝策略等一系列创新方案。

5. **Layer-wise Prefill** — 通过 KVCache 计算与传输的重叠，巧妙地隐藏了 KVCache 传输开销，释放了宝贵的 VRAM 资源。

6. **开源贡献** — 提供了首个包含缓存复用关系的真实请求 trace，对学术社区极具价值。

### 局限与不足 ⚠️

1. **基于启发式的缓存平衡** — 当前的热点迁移阈值需要手动调整，虽然论文提到可自适应调整，但未给出具体方案。

2. **预测模型较为简化** — 系统级预测假设所有请求有统一的解码时间 td，在负载模式变化剧烈时精度可能不足。请求级预测（输出长度预测）目前仍为未来工作。

3. **Prefill/Decoding 比例固定** — 实验表明 [3P+1D] 比 [2P+2D] 效果更好，但最优比例如何动态调整仍是开放问题。

4. **Dummy 模型验证** — 虽然论文解释了为何使用 dummy 模型（保护知识产权），但官方 LLaMA2-70B 或真实模型的验证结果缺失。

5. **仅支持同构加速器** — 异构加速器的探索（计算型 vs. 带宽型）仅停留在未来工作层面。

6. **未与同领域最新工作对比** — 论文仅以 vLLM 为基线，未与 DistServe、TetriInfer、LoongServe 等工作进行同口径对比（尽管这些工作多为同期或后续）。

## 五、个人评价

### 为什么它能获得 FAST 2025 Best Paper？

Mooncake 的获奖绝非偶然。这篇论文抓住了 LLM 推理系统设计中**从"学术假设资源充足"到"工业现实持续过载"的范式转变**，在以下几个维度都做出了突出贡献：

**1. 问题定义的前瞻性**
传统 LLM 推理研究（如 Orca、vLLM、DistServe）的隐含假设是"所有请求都会被处理"，优化目标是最大化吞吐量或 GPU 利用率。Mooncake 直面真实世界的残酷现实——GPU 永远不够用。它重新定义了优化目标：在 SLO 约束下最大化有效处理量（goodput），并系统地处理了"拒绝哪些请求"和"何时拒绝"的问题。

**2. 设计的系统性与完整性**
Mooncake 不是单一技术的堆叠，而是一套从前端调度到后端存储、从正常负载到过载场景**全覆盖**的系统设计：
- Prefill 池的 CPP + Layer-wise 设计
- Decoding 池的连续批处理
- 分布式 KVCache 池的分页存储与哈希去重
- Conductor 的缓存感知 + 缓存平衡调度
- Messenger 的 RDMA 高速传输
- 基于预测的过载控制

每个组件都有明确的 design rationale，并且以 KVCache 为线索贯穿始终，构成了一个高度自洽的整体。

**3. 工程实践的深度**
论文中大量篇幅讨论的是"工程实现中的复杂性"——预测传输时间受网络拥塞影响、排队时间与缓存命中的耦合、负载波动与提前拒绝的反身性效应——这些恰恰是纯学术论文最容易忽略但生产系统必须面对的核心问题。

**4. 可复现性承诺**
开源请求 trace、开源代码、详细的实验设置，使得后续研究能够在此基础上继续推进。

### 对后续工作的启示

- **KVCache 压缩技术的融合**：论文在 Future Work 中大量讨论了 KVCache 压缩（PyramidKV、KIVI、ZipCache 等），这可能是下一步的性能倍增器。
- **异构解构**：将 Attention 操作与 MLP 等线性操作分离到不同类型的加速器上，这个方向具有极大的想象空间。
- **更智能的调度**：基于强化学习或在线学习的调度策略，替代当前的手工阈值。
- **连续 P/D 比例调整**：将 Prefill 实例动态转换为 Decoding 实例（或反之）以应对负载漂移。

## 六、关键图表与数据速查

### 缓存命中率（不同策略与容量）

| 缓存容量 | LRU | LFU | LengthAware |
|---------|-----|-----|-------------|
| 无限 | 0.51 | 0.51 | 0.51 |
| 100,000 | 0.51 | 0.51 | 0.50 |
| 50,000 | 0.50 | 0.49 | 0.48 |
| 30,000 | 0.48 | 0.43 | 0.42 |
| 10,000 | 0.40 | 0.35 | 0.35 |
| 1,000 | 0.30 | 0.30 | 0.30 |

### 过载实验数据

| 策略 | 拒绝请求数 | 相对减少 |
|------|-----------|---------|
| Baseline | 4,183 | — |
| Early Rejection | 3,771 | -9.8% |
| Prediction-based Early Rejection | 3,589 | **-14.2%** |

## 相关链接

- [[LLM推理系统深度综述]]
- [[vLLM PagedAttention 原理]]
- [[LLM PD Disaggregation 架构对比 - Splitwise, DistServe, TetriInfer]]
- [[长上下文 LLM 推理优化技术]]
- [[分布式 KVCache 管理]]

### 外部链接

- **论文原文**: https://arxiv.org/abs/2407.00079
- **GitHub 仓库**: https://github.com/kvcache-ai/Mooncake
- **Kimi Chat**: https://kimi.moonshot.cn
- **papers.cool**（论文对话服务）: https://papers.cool/
- **Moonshot AI 官方**: https://moonshot.ai
