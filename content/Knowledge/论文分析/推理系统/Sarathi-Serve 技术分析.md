---
tags:
- 论文分析
- llm-inference
- system-design
- scheduling
- chunked-prefill
- throughput-latency-tradeoff
arxiv: 2403.0231
authors:
- Amey Agrawal
- Nitin Kedia
- Ashish Panwar
- Jayashree Mohan
- Nipun Kwatra
- Bhargav S. Gulavani
- Alexey Tumanov
- Ramachandran Ramjee
institutions: Microsoft Research India, Georgia Institute of Technology
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
venue: OSDI 2024
permalink: sarathi-serve
---

# Sarathi-Serve: 驯服 LLM 推理中的吞吐量-延迟权衡

## 一、论文概览

| 项目 | 内容 |
|------|------|
| **论文标题** | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve |
| **作者** | Amey Agrawal (Georgia Tech, MSR实习), Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee |
| **机构** | Microsoft Research India + Georgia Institute of Technology |
| **发表** | OSDI 2024 |
| **arXiv** | [2403.02310](https://arxiv.org/abs/2403.02310) |
| **代码** | [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) |
| **评分** | ⭐⭐⭐⭐⭐ |

### 核心贡献

Sarathi-Serve 提出 **分块预填充（Chunked-Prefill）** 与 **无停滞调度（Stall-Free Scheduling）**，从根本上解决了 LLM 推理服务中吞吐量（throughput）与尾延迟（tail latency, TBT）之间的矛盾。核心洞察是：预填充阶段是计算密集型（compute-bound），解码阶段是内存密集型（memory-bound），两者混批时预填充会"阻塞"解码导致**生成停滞（generation stall）**——而 Sarathi-Serve 通过将预填充拆分为等计算量的分块、构造混合批次，使得每次迭代的计算量均匀可控，彻底消除了生成停滞。

### 关键指标

| 模型 | 硬件 | 提升（vs vLLM） |
|------|------|----------------|
| Mistral-7B | 1× A100 | **2.6×** 服务容量 |
| Yi-34B | 2× A100 (TP-2) | **3.7×** 服务容量 |
| Falcon-180B | 8× A100 (TP-4 + PP-2) | **5.6×** 服务容量 |

---

## 二、技术方法详解

### 2.1 问题分析：两种调度策略的困境

论文将现有 LLM 推理调度器分为两大类：

#### 1. 解码优先调度（Decode-Prioritizing）

代表：FasterTransformer [7], Triton Inference Server [17]

**策略**：采用请求级批处理（request-level batching），一批请求的预填充全部完成后，才执行解码；新请求必须等待当前批次全部完成才能进入。

**优点**：TBT（token间延迟）极低——因为已有请求的解码阶段不会被新请求干扰。

**缺点**：吞吐量严重受损——当批次中部分请求提前完成时，批次缩小但仍在运行，GPU 利用率大幅下降。

```
Algorithm 1: Request-level batching
while True:
    if batch为空:
        batch = 新请求集合
        prefill(batch)
    else:
        decode(batch)       ← 只有decode，无干扰
        移除已完成请求
```

#### 2. 预填充优先调度（Prefill-Prioritizing）

代表：Orca [75], vLLM [53]

**策略**：采用迭代级批处理（iteration-level batching）。在每次迭代中，如果有新请求到达且 GPU 内存充足，**立即**执行其预填充，然后将其加入解码批次。

**优点**：吞吐量高——预填充被优先执行，解码批次规模得以快速扩大。

**缺点**：TBT 延迟高——预填充可能耗时极长（取决于 prompt 长度），导致正在解码的请求被"卡住"数秒。这就是论文定义的 **生成停滞（generation stall）**。

```
Algorithm 2: Iteration-level batching (vLLM)
while True:
    收集新请求
    if 有新请求:
        prefill(新请求)     ← 可能耗时数秒，阻塞decode
        加入批次
    else:
        decode(batch)
```

**生成停滞有多严重？** 论文 Fig. 1a 展示：Yi-34B 在 2× A100 上运行 arxiv-summarisation 负载时，vLLM 的生成停滞持续数秒，而 Sarathi-Serve 完全消除。

**关键矛盾**：两种策略都无法同时优化吞吐量与延迟——预填充优先优化吞吐但牺牲 TBT，解码优先优化 TBT 但牺牲吞吐。

### 2.2 Sarathi-Serve 核心设计

#### 核心技术 1：分块预填充（Chunked-Prefill）

**思想**：将输入 prompt 的预填充计算拆分为多个**等计算量**的分块。

- 传统做法：一次处理所有 prompt tokens（可能数千个 token）
- Sarathi-Serve：设定一个 **token budget**（如 512 或 1024），将长 prompt 的 prefill 分散到多次迭代中执行

**为什么有效？** 每个分块的计算时间是**有界且可预测的**——不再依赖 prompt 总长度。

#### 核心技术 2：无停滞调度（Stall-Free Scheduling）

**思想**：构造**混合批次**，在一次迭代中**同时**处理解码 token 和预填充分块。

- 每次迭代，收集所有正在解码的请求（每个贡献 1 个 token）
- 如果还有剩余算力预算（token budget），吸纳一个新请求的一个 prefill 分块
- 每个迭代的总计算量 ≈ batch_size × 1_token_decode + 1_prefill_chunk ≈ 均匀

**关键洞察**：解码 token 和 prefill 分块可以**共享相同的 compute budget**。因为两者的计算模式虽然在 attention 部分不同（prefill 需要 full attention，decode 需要 causal attention），但 FFN 和线性层的计算是相同的。最终，混合批次的整体计算时间 ≈ max(prefill_time, decode_time) 而非 prefill_time + decode_time。

#### Pipeline 并行中的均匀批次

当使用 pipeline parallelism (PP) 时，不同 micro-batch 的计算时间差异会导致 **pipeline bubble**（流水线气泡）。Sarathi-Serve 的均匀批次天然解决了这个问题：

- 每次迭代的计算量近似相等
- 每个 micro-batch 执行时间相近
- pipeline bubble 降至最低

### 2.3 与 vLLM 的对比

| 特性 | vLLM | Sarathi-Serve |
|------|------|---------------|
| 批处理粒度 | 迭代级别 | 迭代级别 |
| 预填充策略 | 立即执行完整预填充 | 分块执行 |
| 预填充阻塞解码 | ✅ 是（生成停滞） | ❌ 否（无停滞） |
| 批次类型 | 纯解码 或 纯预填充 | 混合批次 |
| 迭代计算量 | 高度不均衡 | 近乎均匀 |
| Pipeline 友好 | ❌ 不友好 | ✅ 友好 |

---

## 三、实验评估

### 3.1 实验设置

| 模型 | 参数量 | 硬件 | 并行策略 |
|------|--------|------|----------|
| Mistral-7B | 7B | 1× A100 (80GB) | 单卡 |
| Yi-34B | 34B | 2× A100 | 2-way TP |
| LLaMA2-70B | 70B | 8× A40 | 8-way TP |
| Falcon-180B | 180B | 8× A100 (2节点) | 4-way TP + 2-way PP |

数据集：openchat_sharegpt4, arxiv_summarisation (包含真实长文本负载)

### 3.2 主要结果

#### 服务容量（Capacity）

**定义**：在满足给定 SLO（Service Level Objective，如 TBT < 某一阈值）的前提下，系统能承受的最大查询速率（QPS）。

| 模型 | vLLM | Sarathi-Serve | 提升 |
|------|------|---------------|------|
| Mistral-7B (TBT < 100ms) | ~0.6 QPS | ~1.55 QPS | **2.6×** |
| Yi-34B (TBT < 250ms) | ~0.3 QPS | ~1.1 QPS | **3.7×** |
| Falcon-180B (严格 SLO) | ~0.07 QPS | ~0.32 QPS | **4.3×** |
| Falcon-180B (宽松 SLO) | ~0.35 QPS | ~1.96 QPS | **5.6×** |

#### 延迟对比

- **TBT（Time-Between-Tokens）**：Sarathi-Serve 的 P99 TBT 远低于 vLLM，在 Yi-34B 上，vLLM 在 0.7 QPS 时 P99 TBT 已超过 1s，而 Sarathi-Serve 在 1.0 QPS 时仍保持在 0.3s 以下。
- **TTFT（Time-To-First-Token）**：由于分块引入少量开销，TTFT 略有增加，但在合理范围内（通常 < 1s vs vLLM 的 0.5s）。

#### 消融实验

**分块开销（Fig. 14）**：
- Chunk size = 512：约 25% 的 prefill 额外开销
- Chunk size = 2048：开销几乎可忽略
- 这证明了分块的高效性——即使是非常小的分块，额外开销也很有限

**组件分离分析（Table 4）**：
- 仅使用 hybrid-batching（混合批次，但不对预填充分块）：TBT 仍然高（长预填充仍导致生成停滞）
- 仅使用 chunked-prefills（分块但纯 prefill/decode 分离）：TTFT 高（分块引入开销，且缺乏混批收益）
- **两者结合**：TTFT 和 TBT 都低——证明两项技术协同工作

### 3.3 Pipeline 并行效果

论文展示了一件重要的事：当跨节点部署（commodity Ethernet）时，TP 扩展性差（通信开销大），PP 是更好的选择。但 PP 在 vLLM 中因计算不均衡导致大量 pipeline bubble。

Sarathi-Serve 的均匀批次使 PP 在 Falcon-180B 上获得了 **3.6×（严格 SLO）、1.48×（宽松 SLO）** 的提升。

---

## 四、亮点与局限

### 亮点

1. **直击核心矛盾**：清晰地识别了 LLM 推理调度中吞吐量-延迟的根本矛盾，并用极为简洁优雅的方案（分块预制 + 无停滞调度）解决。

2. **实用性强**：不依赖复杂的硬件特性或模型改动，所有优化都在调度层面完成，可轻松集成到现有系统（如 vLLM）。

3. **面向真实 SLO**：在服务容量指标中引入 TBT 尾延迟约束，使评估更贴近生产环境。

4. **Pipeline 并行收益**：论文的一个意外收获——均匀批次对 PP 的额外好处——进一步扩大了其适用范围。

5. **开源可复现**：代码在 GitHub 上开源，实验脚本、trace 数据完整提供。

### 局限

1. **TTFT 略有牺牲**：分块预填充比一次完成稍慢（最多 25% 额外开销），对 TTFT 敏感的极端场景（如实时对话）可能不理想。

2. **Token budget 选择需要调优**：512 vs 2048 的选择依赖于负载特性和 SLO，系统部署时需要实验确定。

3. **与解耦架构的对比缺失**：论文提及 SplitWise/DistServe 等解耦方案（prefill 与 decode 在不同 GPU 上），但未做定量比较。解耦方案可能提供更好 TTFT，但引入 KV-cache 迁移等新挑战。

4. **聚焦在线推理**：不适用于离线批处理场景（如 FlexGen 的优化目标）。

5. **尚未考虑 MoE 等新架构**：对 Mixture-of-Experts 模型的支持和优化未讨论。

---

## 五、个人评价

### 学术贡献

Sarathi-Serve 是 LLM 推理系统领域一篇里程碑式的论文。它的核心价值不在于提出革命性的新硬件或新模型，而在于对已有调度机制的深刻洞察和一个**异常优雅的工程解决方案**。

**Chunked-prefill** 这个想法看起来简单，但其影响深远：
- 它让 prefill 从"不可控的长时间操作"变成"可控的有界操作"
- 它为调度器提供了**更细粒度的控制权**
- 它自然衍生出 stall-free scheduling 和 uniform batch 两个重要能力

### 产业影响

该论文发表于 OSDI 2024，代码开源后影响力迅速扩展。许多后续工作（如 APIServe、TetriInfer）直接采纳了 chunked-prefill 的设计。在现代 LLM 推理框架（如 vLLM 后续版本、SGLang 等）中，分块预填充已成为标准功能。

### 与其他工作的关系

| 工作 | 关系 |
|------|------|
| **vLLM** | 主要对比基线。vLLM 的 PagedAttention 解决了 KV-cache 碎片化问题，而 Sarathi-Serve 解决了调度层面的吞吐-延迟矛盾。两者互补。 |
| **Orca** | 前身工作。首次提出迭代级批处理，但未解决 prefill 阻塞 decode 的问题。 |
| **SplitWise / DistServe** | 解耦路线。将 prefill 和 decode 放在不同 GPU 上。避免干扰但增加 KV-cache 迁移开销。 |
| **Sarathi (原始版)** | Sarathi-Serve 的前身（Sarathi, arXiv:2311.xxxx），提出在 decode batch 中"捎带"预填充分块。Sarathi-Serve 在此基础上扩展到 pipeline 并行场景。 |
| **Vidur** | 同一团队开发的 LLM 推理模拟器，用于大规模仿真实验。 |

### 一句话总结

> **Sarathi-Serve 通过分块预填充将 LLM 推理调度从"粗粒度的选择"变为"细粒度的控制"，在几乎不牺牲延迟的前提下实现了服务容量数倍的提升。**

---

## 相关链接

- [[LLM推理系统深度综述]]
- [[vLLM核心论文分析]]
- [[Orca-LLM推理系统分析]]
- [[张量并行与流水线并行在LLM推理中的对比]]
- [[FlashAttention与PageAttention解析]]

### 延伸阅读

- **原始 Sarathi 论文**：Amey Agrawal et al., "Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills", arXiv:2311.XXXXX (2023)
- **SplitWise**：Patel et al., "Splitwise: Efficient Generative LLM Inference using Phase Splitting", arXiv:2311.18677 (2023)
- **DistServe**：Zhong et al., "Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", arXiv:2401.09670 (2024)
- **APIServe**：Abhyankar et al., "APIServe: Efficient API Support for Large-Language Model Inferencing", arXiv:2402.01869 (2024)
- **Vidur**：Agrawal et al., "Vidur: A Large-Scale Simulation Framework for LLM Inference", MLSys 2024
