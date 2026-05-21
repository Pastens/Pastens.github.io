---
tags:
- 论文分析
- llm-inference
- simulation
- configuration-search
- llm-serving
source: https://github.com/microsoft/vidur
arxiv: 2405.05465
authors:
- Amey Agrawal
- Nitin Kedia
- Jayashree Mohan
- Ashish Panwar
- Nipun Kwatra
- Bhargav Gulavani
- Ramachandran Ramjee
- Alexey Tumanov
institutions:
- Microsoft Research India
- Georgia Institute of Technology
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Vidur: A Large-Scale Simulation Framework For LLM Inference

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Vidur: A Large-Scale Simulation Framework For LLM Inference |
| **arXiv** | [2405.05465](https://arxiv.org/abs/2405.05465) |
| **分类** | cs.LG (Machine Learning), cs.AI, cs.CL |
| **机构** | Microsoft Research India, Georgia Institute of Technology |
| **代码仓库** | [github.com/microsoft/vidur](https://github.com/microsoft/vidur) |
| **Stars** | ~598 ⭐ |
| **发表状态** | MLSys 2025 (双盲审稿) |
| **提交日期** | 2024-05-08 (v1), 2024-05-21 (v2) |

### 核心贡献

1. **Vidur 模拟器**：首个面向 LLM 推理的高保真、大规模、易扩展的性能模拟框架，结合实验 profiling 和机器学习预测建模，以 <9% 的误差估计端到端推理延迟。

2. **Vidur-Bench 基准套件**：提供多套真实工作负载 trace（LMSys-Chat-1M、Arxiv-Summarization、Bilingual-Web-Book），集成 vLLM、Orca、Sarathi-Serve、FasterTransformer 等多种调度策略和 serving 框架。

3. **Vidur-Search 配置搜索工具**：在 CPU 机器上自动化搜索最优部署配置，以最大化 QPS/dollar。例如为 LLaMA2-70B 找到最优配置只需 ~1 小时（$9.93 的 CPU 成本），而实际硬件探索需 42K GPU 小时（~$218K）。

4. **算子三分类方法论**：创新性地将 LLM 算子分为 token 级、sequence 级和通信级三类，分别设计 profiling 和预测策略，大幅降低 profiling 成本。

5. **揭示配置-工作负载依赖关系**：通过大规模 what-if 分析证明最优部署配置不仅取决于模型，还高度依赖于工作负载特性，错误配置可导致高达 2× 的成本差异。

---

## 二、技术方法详解

### 背景与动机

LLM 推理部署涉及大量配置旋钮：并行化策略（TP、PP 及其组合）、调度算法（Orca、vLLM、Sarathi-Serve）、批大小（batch size）、批等待时间、块大小（chunk size）、GPU SKU 选择等。这些配置构成一个巨大的组合搜索空间。

论文的一个关键洞察是：**最优配置不是模型本身的函数，而是模型-工作负载对 (model-trace pair) 的函数**。同一模型在不同工作负载 trace 上的最优配置可能完全不同，错误配置的成本可达 2×。

当前 DNN 训练模拟器（Daydream、Habitat、Proteus）无法直接用于 LLM 推理，原因有三：
1. **时间尺度**：训练迭代通常是数百毫秒，而推理迭代可能短至几毫秒；
2. **迭代时间剧烈变化**：prefill 与 decode 阶段计算特性不同，序列长度变化大，批大小动态调整；
3. **误差级联**：推理的 stateful 特性使得单次预测误差会在后续调度决策中被放大。

### Vidur 模拟器架构

#### 2.1 总览

Vidur 由两个主要阶段组成：

**阶段一：模型 onboarding**
1. 模型规范 → 识别算子集合
2. Profiler（§4.3）采集运行时特性
3. Runtime Estimator（§4.4）训练小型 ML 模型，生成算子级运行时查找表

**阶段二：仿真执行**
1. 用户指定部署配置 + 工作负载
2. 事件驱动的仿真引擎运行 Hierarchical Scheduler（§4.5）
3. 输出请求级（TTFT、TBT、延迟、吞吐量）和集群级（MFU、内存利用率）指标

#### 2.2 关键洞察（Key Insights）

**(1) LLM 共享架构属性**：大多数 LLM 使用相似的 Transformer 架构，差异仅在激活函数、归一化层、残差连接等细节。这使得可以用统一的声明式模型规范描述不同模型，且只需建模少数共享算子。

**(2) 算子三分类（Operation Triaging）**：LLM 算子可按输入依赖关系分为三类：

| 算子类型 | 依赖维度 | 示例 | 预测策略 |
|---------|---------|------|---------|
| **Token-level（Token 级）** | 当前批处理的总 token 数（prefill + decode） | 线性层 (Linear)、激活函数 (Activation)、LayerNorm | 对 tensor parallel 分片配置 profiling，用随机森林回归插值 |
| **Sequence-level（序列级）** | 每个请求的上下文长度（KV-Cache 历史） | Attention（prefill & decode） | Prefill: 用等效单序列长度 √(Σpᵢ²) 近似；Decode: 基于总 KV-Cache 读取量 |
| **Communication（通信级）** | 传输数据量（独立于模型架构） | all-reduce、all-gather、send-recv | 独立于模型，提前对不同拓扑进行 profiling |

该分类的核心价值在于：attention 之外的算子在 decode 阶段与请求历史无关（MLP 层的计算量不取决于之前处理的 token 数量），而 attention 作为唯一的 memory-bound 算子，其 decode 运行时主要由总 KV-Cache 读取量决定。

**(3) 自动 profiling 并行策略**：Vidur 利用 LLM 并行策略的领域知识，从声明式模型规范自动识别 tensor sharding 配置，从而**在单 GPU 上进行 profiling 即可模拟多种并行方案**。

#### 2.3 Profiler（§4.3）

**Token-level 算子 profiling**：基于模型规范生成所有 tensor parallel 分片配置组合，使用标准 PyTorch kernel + CUPTI 测量性能。

**Sequence-level 算子 profiling**：
- **Prefill attention**：成本与序列长度的平方成正比 Σpᵢ²。通过构造"等效单序列"（长度为 √Σpᵢ²）来近似批处理的 attention 时间。
- **Decode attention**：主要是 memory-bound，运行时取决于总 KV-Cache 读取量。现代序列并行 attention kernel（PagedAttention v2、FlashDecoding）能有效处理各请求之间上下文长度的不均。

**Communication 算子 profiling**：独立于模型，提前对 all-reduce、all-gather、send-recv 在不同拓扑下进行 profiling。

#### 2.4 Runtime Estimator（§4.4）

对每种算子收集少量 profiling 数据点，使用**随机森林回归（Random Forest Regression）** 模型进行插值预测。

选择随机森林而非 MLP（Habitat 等使用）的原因：
- MLP 需要大量训练数据，且对闭源 CUBLAS/cuDNN kernel 的非线性行为（如 tile/wave quantization）拟合不佳
- 简单多项式回归无法捕捉 CUDA kernel 的非线性特征
- 随机森林在**数据效率**和**预测保真度**之间取得了最佳平衡

Runtime Estimator 为每种算子生成一个基于输入张量维度的运行时查找表，仿真时可直接查表得到算子执行时间。

#### 2.5 Hierarchical Scheduler（§4.5）

Vidur 采用**三层级调度架构**：

1. **Global Scheduler（全局调度器）**：请求路由。支持 round-robin、least outstanding requests 以及 stateful 调度（延迟绑定路由决策，应对突发流量）。

2. **Replica Scheduler（副本调度器）**：批处理 + 内存管理。包含 Memory Planner（根据模型规范和并行配置计算 KV-Cache 可用内存）和 Memory Manager（提供高层管理 API）。目前已支持 5 种批处理策略（FasterTransformer、Orca、Sarathi-Serve、vLLM、LightLLM），每种在 Vidur 中的实现仅需 <150 行 Python 代码。

3. **Replica Stage Scheduler（副本阶段调度器）**：处理 pipeline 内 micro-batch 的调度。当前仅支持同步 pipeline parallel 调度。

### Vidur-Bench（§5）

Vidur-Bench 提供三类工作负载和一个标准化的指标集：

**工作负载**：

| 数据集 | 内容 | 特点 |
|-------|------|------|
| LMSys-Chat-1M (Chat-1M) | 自然语言对话 | 短 prefill（均值 686 token），短 decode（均值 197 token），高 P:D ratio (2.3) |
| Arxiv-Summarization (Arxiv-4K) | 论文摘要生成 | 长 prefill（均值 2588 token），中等 decode（均值 291 token），极高 P:D ratio (15.7) |
| Bilingual-Web-Book (BWB-4K) | 中英双语文档翻译 | 中等 prefill（均值 1067 token），极长 decode（均值 1612 token），低 P:D ratio (0.65) |

**性能指标**：算子级（执行时间）、请求级（调度延迟、TTFT、TBT）、副本级（批大小、计算/内存利用率）、硬件级（GPU FLOPs 利用率）。

### Vidur-Search（§6）

Vidur-Search 是一个**配置优化搜索工具**，解决带约束的优化问题：

- **输入**：LLM 模型、工作负载、可用 GPU SKU、GPU 数量上限
- **约束**：TTFT P90 < 2s，TBT P99 < 200ms
- **搜索空间**：TP/PP 维度（1/2/4）、调度策略（vLLM/Orca+/Sarathi-Serve）、批大小（32/64/128/256/512）、chunk size、GPU SKU（A100/H100）
- **优化目标**：最大化 **QPS per dollar**

搜索算法：
1. 枚举所有部署配置组合（数百种）
2. 对每种配置，通过**二分搜索**找到最大可支持的 QPS（以 P99 调度延迟 < 5s 为约束）
3. 每次二分搜索迭代都运行一次 Vidur 仿真
4. 所有搜索可并行化（每个配置在一个 CPU 核上运行）

这一方法使得在 **96 核 CPU 上约 1 小时**即可完成 LLaMA2-70B 的最优配置搜索，等价于 ~42K GPU 小时的实际部署试验。

---

## 三、实验评估

### 实验设置

| 维度 | 配置 |
|------|------|
| **模型** | InternLM-20B (TP2), LLaMA2-70B (TP4), Qwen-72B (TP4), LLaMA2-7B |
| **硬件** | A100 (80GB), H100 (80GB) |
| **工作负载** | Chat-1M, Arxiv-4K, BWB-4K（静态 + 动态 Poisson 到达） |
| **调度器** | 默认使用 vLLM；what-if 分析使用 vLLM、Orca+、Sarathi-Serve |
| **评估指标** | 归一化端到端延迟（静态），归一化延迟百分位（动态） |

### 关键结果

#### 3.1 模拟保真度（Simulator Fidelity）

**静态工作负载**（Figure 3）：
- 四个模型 × 三个数据集上，Vidur 预测的中位数延迟误差 < 3.33%，P95 尾延迟误差 < 9%
- 7B 小模型误差略高，原因为 CPU 开销占比相对更高

**动态工作负载**（Figure 4）：
- 在系统容量 85% 的负载率下，几乎所有场景都达到 < 5% 的误差
- 接近容量点时，微小预测误差可能导致延迟的显著偏差（系统处于 tipping point），但实际生产系统会预留缓冲区，不会运行在满载状态

**级联误差分析**（Appendix A）：
- 在不同请求到达率下，Vidur 的保真度保持稳定，证明了事件驱动仿真中误差控制的鲁棒性

#### 3.2 What-if 分析（§7.3）

通过对 LLaMA2-70B、Qwen-72B、InternLM-20B、LLaMA2-7B 四个模型在三个工作负载上的最优配置搜索，得到以下关键发现：

**发现 1：工作负载变化显著改变最优配置**
- LLaMA2-70B 在 Chat-1M 上最优配置为 batch size 256，而在 BWB 上为 batch size 64
- BWB 的长 decode 序列导致高 KV-Cache 负载，限制了批大小
- 甚至 GPU SKU 选择也从 Chat-1M 的 H100 变为 BWB 的 A100

**发现 2：架构细节影响服务成本**
- LLaMA2-70B 使用 GQA（Group Query Attention），Qwen-72B 使用 MHA（Multi-Head Attention）
- MHA 的 KV-Cache 负载是 GQA 的 8× → Qwen-72B 的服务成本约为 LLaMA2-70B 的 2×

**发现 3：配置稳定性差**
- 将一个工作负载的最优配置用于另一个工作负载，可导致高达 2× 的成本差异
- 例如在 LLaMA2-70B 上，Chat-1M 的最优配置用于 Arxiv-4K 时，性能下降近 2×

**发现 4：SLO 微小变化导致成本大幅波动**
- TBT SLO 从 0.14s 放松到 0.12s（仅差 20ms），成本降低约 1.85×
- Pareto 前沿分析表明：在一个指标上的最优配置可能不满足另一个指标的 SLO 约束

**成本对比**：完整的 what-if 分析（12 个模型-工作负载对 × 数百种配置）在仿真中仅耗费 **$125**（CPU 计算），等价实际部署成本为 **$1,140,000**（GPU 计算），节省约 **9,120×**。

---

## 四、亮点与局限

### 亮点 👍

1. **高保真与低成本的最佳平衡**：<9% 的预测误差足以让配置搜索排名的可靠性远超随机探索，而成本仅为实际部署的数万分之一。

2. **算子三分类方法论简洁优雅**：通过领域知识将复杂问题分解为三类独立可预测的算子，大幅降低 profiling 工作量。这是 LLM 推理仿真领域截至目前最系统化的方法。

3. **高度可扩展的插件化架构**：新增调度策略只需 <150 行 Python，新增模型只需声明式规范 + 少量 profiling，降低了社区贡献门槛。

4. **揭示重要实践洞察**：证明了"没有通用的最优配置"，工作负载感知的配置优化应成为 LLM 推理部署的标准实践。这一发现对生产环境的成本优化具有直接指导意义。

5. **代码开源且质量高**：GitHub 上有完整实现（598 stars），包括 profiler、runtime estimator、hierarchical scheduler、benchmark 和 search 工具，便于复现和扩展。

### 局限 👎

1. **不支持 speculative decoding**：当前 Vidur 不支持推测性解码（speculative decoding）、异步通信、序列并行（sequence parallelism）等较新的推理优化技术，作者将其列为未来工作。

2. **Pipeline parallel 仅支持同步模式**：不支持异步 pipeline 调度、1F1B 等高级 PP 策略，限制了搜参空间中 PP 的使用。

3. **Profiling 仍然需要少量 GPU 时间**：虽然比实际部署的 GPU 小时少得多，但对于完全没有 GPU 资源的小团队，初始 profiling 可能仍是一个障碍。

4. **OOTB kernel 假设**：Vidur 假设 LLM 使用标准 PyTorch kernel 实现，对于高度优化的自定义 kernel（如 FlashAttention 的特定变体）可能需要额外适配。

5. **未覆盖全精度与量化混合场景**：论文主要关注 FP16/BF16 精度推理，对于量化推理（INT8/INT4）或混合精度场景的建模有待扩展。

---

## 五、个人评价

Vidur 是 LLM 推理系统领域一篇非常扎实的 MLSys 级别工作。其核心价值不仅在于提供了一个好用的模拟工具，更在于**系统性地揭示了 LLM 推理部署中配置优化的重要性及其与工作负载的强关联**。文中 "最优配置是模型-工作负载对的函数" 这一发现对于生产环境的成本优化有直接的实践指导意义。

从技术角度看，算子三分类 + 随机森林回归的组合在简洁性和有效性之间取得了很好的平衡，是一种"巧劲"而非"蛮力"的解决方案。Vidur-Search 中的二分搜索找最大 QPS 方法也展现了工程上的务实考量。

总体而言，Vidur 填补了 LLM 推理仿真的空白（此前只有 DNN 训练模拟器），有望成为 LLM 推理系统研究的**标准基础设施**。它使得研究人员可以低成本地进行大规模 what-if 分析、快速评估新调度算法、以及优化生产部署配置。强烈推荐阅读原文和尝试其开源代码。

与同类工作对比：
- **Daydream/Habitat/Proteus**：训练模拟器，不适用于推理场景
- **LLM-Emu**（后续工作）：更轻量的建模方式，但精度和细节不如 Vidur
- **Empirical deployment search**：成本极高（42K GPU hours vs Vidur 的 1 CPU hour），不具备可扩展性

---

## 相关链接

- [[Knowledge/论文分析/推理系统/LLM推理系统深度综述]]
- [[Knowledge/论文分析/推理系统/Sarathi-Serve 技术分析]]
