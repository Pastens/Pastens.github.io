---
tags:
- 论文分析
- llm-inference
- simulation
- moe
- disaggregation
arxiv: 2508.03148
authors: Yicheng Feng, Xin Tan, Kin Hang Sew, Yimin Jiang, Yibo Zhu, Hong Xu
institutions: 香港中文大学 (CUHK), StepFun
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# Frontier: Simulating the Next Generation of LLM Inference Systems
## Frontier：模拟下一代 LLM 推理系统

## 一、论文概览

| 项目 | 内容 |
|---|---|
| **论文标题** | Frontier: Simulating the Next Generation of LLM Inference Systems |
| **作者** | Yicheng Feng¹, Xin Tan¹, Kin Hang Sew¹, Yimin Jiang², Yibo Zhu², Hong Xu¹ |
| **机构** | ¹香港中文大学 (CUHK), ²StepFun |
| **发表** | arXiv:2508.03148, 2025年8月 |
| **页数** | 7页（含参考文献） |
| **代码** | 计划开源（论文中承诺） |
| **论文链接** | https://arxiv.org/abs/2508.03148 |

### 核心贡献

Frontier 是**首个**专为下一代 LLM 推理系统设计的高保真仿真器，其核心贡献包括：

1. **新的 stage-centric 仿真架构**：彻底告别传统的 replica-centric（副本中心）抽象，提出以"阶段"（stage）为核心的原语，原生支持分布式、多阶段的工作流建模。

2. **原生 MoE 支持**：首次在 LLM 推理仿真器中完整支持 Mixture-of-Experts 模型的推理模拟，包括 Expert Parallelism (EP)、跨集群专家路由、以及数据依赖的微工作流（如 token 负载不均导致的 straggler 效应）。

3. **全面的解耦架构支持**：同时支持 Prefill/Decode (PD) 解耦和 Attention/FFN (AF) 解耦两种主流解耦范式，能模拟跨集群的生产者-消费者动态和细粒度流水线。

4. **高精度算子级建模**：针对 Attention 和 GroupedGEMM 等异构敏感算子，使用细粒度特征工程 + ML 模型（如随机森林）实现高精度运行时预测，Attention 算子 94% 以上预测误差低于 10%，GroupedGEMM 95% 以上预测误差低于 6%。

5. **模块化、可插拔策略框架**：将动态批处理、请求调度、内存管理等系统级策略作为一等公民，支持研究者灵活组合与评估不同的策略。

### 与现有工作的对比

| 特性 | LLMServingSim [7] | Vidur [4] | **Frontier (本文)** |
|---|---|---|---|
| PD 解耦 | ✗ | ✗ | **✓** |
| AF 解耦 | ✗ | ✗ | **✓** |
| PP/TP（流水线/张量并行） | ✓ | ✓ | **✓** |
| DP（数据并行） | ✗ | ✗ | **✓** |
| EP（专家并行） | ✗ | – | **✓** |
| 高级调度策略 | ✗ | – | **✓** |

> ✓ = 完全支持 | ✗ = 不支持 | – = 部分/有条件支持

---

## 二、技术方法详解

### 2.1 设计动机：为什么现有仿真器不够用？

现有 LLM 推理仿真器（如 Vidur [4]、LLMServingSim [7]）基于 **replica-centric（副本中心）** 抽象，将系统视为一组同质化、自包含的推理副本（replica），核心问题简化为在副本间做负载均衡。这种抽象在下一代推理架构下存在三个根本性缺陷：

1. **无法表示分布式多阶段工作流**：在 PD/AF 解耦和 MoE 架构中，推理不再是单个副本内的单一任务，而是跨多个专用异构集群编排的多阶段流水线。Replica-centric 抽象缺少表示跨集群路由、KV-Cache 传输、复杂同步等操作的原语。

2. **算子级建模精度不足**：
   - Attention 算子：Vidur 使用单一代理长度（序列长度的平方根）简化估计，但在批次内序列长度差异大时效果差。如一个 72 请求的批次中，Vidur 对 FlashAttention 的预测误差超过 **55%**（0.151ms vs 0.340ms）。
   - 缺少新算子的建模：GroupedGEMM（MoE 中的异构 GEMM）等关键算子完全没有被覆盖。

3. **系统级策略抽象不足**：真实引擎（如 vLLM、SGLang、TensorRT-LLM）的动态批处理、请求调度和内存管理策略影响巨大，但现有仿真器往往将其过度抽象或忽略。

**这三个挑战的核心结论是：关键抽象已经从"管理一组副本"转变为"编排请求通过分布式系统的流程"。**

### 2.2 Frontier 整体架构

Frontier 采用**层次化事件驱动**架构，核心设计思想是将 LLM 推理系统视为一个"系统的系统"（system-of-systems）。

```
GlobalController（全局控制器）
  ├── Workload Generator（负载生成器）
  ├── Performance Collector（性能收集器）
  └── 管理多个 ClusterWorker

ClusterWorker（集群工作者）
  ├── ClusterScheduler（集群调度器）
  └── 管理多个 ReplicaWorker

ReplicaWorker（副本工作者）
  ├── Model Runner（模型运行器）
  │     ├── Replica Scheduler（副本调度器）
  │     └── Batching Engine（批处理引擎）
  └── ExecutionPredictor（执行预测器）
```

#### GlobalController（全局控制器）

有状态的工作流编排器，是解耦系统的核心：
- **PD 解耦**：管理预填充阶段的 KV-Cache 传输与解码阶段的内存信号之间的 backpressure
- **AF 解耦**：构建事件依赖图，编排跨 Attention 集群和 FFN 集群的微批次流水线

#### ClusterWorker（集群工作者）

专用硬件集群的抽象，包含：
- **ClusterScheduler**：管理本地资源，参与跨阶段协调（如 PD 中的内存可用性信号）
- **ReplicaWorker 池**：集群内的计算实例

#### ReplicaWorker & ExecutionPredictor

**ExecutionPredictor 是关键创新**——它将一个逻辑层分解为数据依赖的微工作流事件。对于 MoE 层，它模拟门控决策生成 token-to-expert 分配图，将专家计算建模为一组异构任务，通过取最大值来原生捕获 straggler 效应。

### 2.3 高精度算子运行时预测

#### Attention 算子的挑战与方案

**挑战**：批次内序列长度差异大时，Vidur 的单一代理长度方法失效。实际 GPU 内核执行涉及分区和分块（tiling），输入异质性导致波前量化（wave quantization）等复杂现象。

**Frontier 的方案**：
- 使用**丰富的特征集**：包括序列长度的聚合统计（均值、方差等）和分布统计
- 训练 **ML 模型（随机森林）** 进行预测，更精确地捕获工作负载动态

#### GroupedGEMM 算子的挑战与方案

**挑战**：MoE 模型中不同专家收到的 token 数量不同，导致内部工作负载不均。

**Frontier 的方案**：
- 提取反映输入属性和专家负载分布的特征：token 计数、专家数量、模型维度、专家选择比、负载均衡指标等
- 同样使用 ML 模型进行预测

**结果**：Attention 算子 94% 以上预测误差 < 10%；GroupedGEMM 95% 以上预测误差 < 6%（Fig. 2）。

### 2.4 解耦架构工作流仿真

#### PD 解耦：生产者-消费者模型

核心挑战：精确模拟两个速率不匹配的子系统（预填充 vs 解码）之间的协调和 backpressure。

Frontier 的仿真流程：
1. **预填充阶段（生产者）**：GlobalController 将请求路由到预填充集群，模拟排队和执行。完成后请求状态变为 PREFILL_COMPLETE，KV-Cache 保留在预填充内存缓冲区。
2. **解码阶段（消费者）**：ClusterScheduler 持续追踪 GPU 内存利用率。当解码完成释放 KV-Cache 时，向 GlobalController 发信号。
3. **全局协调（Backpressure）**：GlobalController 维护 PREFILL_COMPLETE 请求队列，仅在收到解码阶段的内存可用信号后才发起 KV_CACHE_TRANSFER 事件。

#### AF 解耦：事件依赖图

核心挑战：精确捕获多阶段微批次驱动工作流的**关键路径**，其中微小的阶段间不平衡就可能产生显著的流水线气泡（pipeline bubbles）。

Frontier 的方案：
1. GlobalController 发起解码步骤后，decode-attn 阶段的 ReplicaWorker 将全局批次分为 m 个微批次
2. GlobalController 和 ClusterScheduler 动态构建跨 L 个模型层的所有操作的事件依赖图
3. 事件驱动引擎在依赖满足时立即调度事件，**内在模拟了计算与通信的重叠**
   - 例如：A_TO_F_TRANSFER(i,k) 在传输时，ATTN_COMPUTE(i+1,k) 可在空闲的 Attention GPU 上调度
4. 一个 token 的生成时间 = 图中最后一个事件（通常是 FFN_COMPUTE）的时间戳

#### MoE 推理：微工作流分解

核心挑战：MoE 层的性能不是由平均行为决定，而是由 token 负载不均导致的**最差 straggler** 决定。

Frontier 的仿真流程：
1. 配置虚拟模型分片以满足拓扑约束（如 attn_dp * attn_tp == moe_tp * moe_ep）
2. 遇到 MoE 层时，ExecutionPredictor 模拟以下事件序列：
   - 门控网络的 GEMM
   - 调用可插拔路由模块，生成 token-to-expert 分配图
   - 对每个 expert i，使用实际分配的 token 数查询 GroupedGEMM 性能模型
3. 模拟隐式同步屏障：延迟 = max(T_expert1, T_expert2, ..., T_expertN)

---

## 三、实验评估

### 3.1 实验设置

| 配置 | 详情 |
|---|---|
| **硬件** | 8-GPU 节点（具体 GPU 型号论文未明确说明） |
| **软件** | vLLM 0.10.1 + SharedStorageConnector KV 接口 |
| **模型** | Qwen2-7B-Instruct |
| **评估内容** | 算子级精度 + 端到端系统吞吐量 |

### 3.2 算子级精度

| 算子 | Frontier 预测误差 | Vidur 对比 |
|---|---|---|
| **Attention** | >94% 的样本误差 < 10% | 在序列长度方差大时误差 > 55% |
| **GroupedGEMM** | >95% 的样本误差 < 6% | 不支持此算子 |

> **Fig. 2** 的 CDF 曲线清晰展示了 Frontier 在 Attention 和 GroupedGEMM 两个算子上的显著优势。Vidur 的 Attention 模型在动态工作负载下误差分布广泛，而 Frontier 的 CDF 曲线在低误差区间迅速攀升。

### 3.3 端到端系统精度

PD 解耦配置（预填充:解码实例 = 1:1），预测吞吐量 vs 实测吞吐量：

| 批次大小 | 平均输入长度 | 平均输出长度 | 实测吞吐量 (tokens/s/GPU) | 预测吞吐量 (tokens/s/GPU) | 相对误差 |
|---|---|---|---|---|---|
| 4 | 32 | 1024 | 111.355 | 90.498 | ~18.7% |
| 8 | 128 | 256 | 131.831 | 109.366 | ~17.0% |
| 16 | 256 | 128 | 151.425 | 127.157 | ~16.0% |
| 32 | 32 | 128 | 313.236 | 240.743 | ~23.2% |

**分析**：预测吞吐量持续低于实测值，相对误差在 16.0%–23.2% 之间。论文认为这属于"可接受"范围，但相比 Vidur 在传统部署上的精度（通常 <10%），**端到端精度仍有较大提升空间**。预测系统性偏低可能反映了仿真器在某些开销上过于保守。

### 3.4 局限性说明（论文自述）

- 仅进行了初步评估（preliminary evaluation），缺乏大规模的消融研究和更多场景验证
- 论文明确将以下内容列为未来工作：
  - 扩展核心算子的建模
  - 量化仿真保真度和成本
  - 通过多样化的案例研究展示 Frontier 在大规模系统设计和优化中的实用性

---

## 四、亮点与局限

### 🌟 亮点

1. **架构设计的根本创新**：Stage-centric 抽象是对传统 replica-centric 抽象的根本性颠覆。这一设计决策使 Frontier 能够原生支持下一代推理架构，而不是在旧架构上打补丁。

2. **MoE 支持的完整度**：从 GroupedGEMM 算子级建模到 EP（专家并行）、专家路由、straggler 模拟，Frontier 提供了 MoE 推理仿真的端到端解决方案——这是目前文献中首个完整覆盖这些能力的仿真器。

3. **PD 和 AF 双范式覆盖**：既支持目前主流讨论的 PD 解耦，也支持更前沿的 AF 解耦（如 MegaScale-Infer 和 Step-3 的方案），覆盖了行业正在探索的主要方向。

4. **ML 驱动的算子模型**：使用随机森林等 ML 模型替代传统的解析模型或单一代理方法，在不显著增加仿真开销的前提下大幅提升了 Attention 等复杂算子的预测精度。

5. **可插拔策略框架**：将系统级策略作为一等公民的设计让 Frontier 不只是一个仿真器，更是一个系统设计的实验平台。

### ⚠️ 局限

1. **端到端精度有待提升**：16%–23% 的端到端吞吐量误差在实际系统设计决策中可能不够可靠。对比 Vidur 在传统场景下 <10% 的误差，Frontier 的保真度还有明显差距。

2. **评估规模有限**：
   - 仅测试了 8-GPU 单节点 + Qwen2-7B 模型——远未达到其声称要模拟的"大规模分布式系统"的量级
   - 未展示 PD 和 AF 解耦在更大集群上的有效性
   - 没有 MoE 模型的端到端实验（只测了 GroupedGEMM 算子级精度）

3. **缺乏消融实验**：没有验证各组件（如 ML 算子模型 vs 解析模型、stage-centric vs 改造的 replica-centric）的边际贡献。

4. **未知的计算开销**：使用随机森林进行运行时预测相比于解析模型会有额外开销，论文未讨论仿真器本身的运行效率和可扩展性。

5. **对 Step-3 / MegaScale-Infer 的具体实现细节描述有限**：虽然论文声称能模拟 AF 解耦，但 AF 解耦的具体 pipelining 策略（如 ping-pong 流水线如何精确建模）仅给出了高层次描述。

6. **计划开源但尚未开源**：论文承诺开源代码，但截至阅读时尚未开放。这限制了可复现性和社区的进一步验证。

---

## 五、个人评价

### 定位与意义

Frontier 是一篇填补明确空白的系统论文。在 LLM 推理快速从"单节点部署 dense 模型"走向"大规模分布式部署 MoE + 解耦架构"的行业趋势下，缺乏合适的仿真工具已成为阻碍系统设计和优化的关键瓶颈。Frontier 敏锐地抓住了这一点，并提出了正确的设计方向。

### 与 Vidur 的关系

Frontier 在精神上继承了 Vidur 的许多设计原则（事件驱动、模块化、可扩展），但做出了关键的架构突破。如果把 Vidur 比作"单机批处理系统的仿真器"，Frontier 就是在尝试做"分布式数据流系统的仿真器"。这不是迭代改进，而是**架构层面的范式转换**。

### 技术路线的判断

使用 ML 模型（随机森林）替代解析模型来做算子级预测是一个有趣的选择：
- **优点**：精度高，特别是对 Attention 这种输入依赖性强、规律复杂的算子
- **风险**：训练数据的覆盖范围决定了预测的外推能力——如果训练数据未覆盖某类极端负载分布，ML 模型的预测可能比解析模型更不可靠
- **前景**：未来可以探索混合方法——解析模型做基础预测，ML 模型做残差修正

### 建议关注的方向

1. **开源的时机与质量**：代码开源后，社区的第一件事应该是复现论文结果并探索更大的配置空间
2. **与现有生态的集成**：Frontier 能否对接 vLLM / SGLang 的 trace 数据进行校准？能否输出可直接用于系统调优的建议？
3. **大规模场景的保真度验证**：在 32/64/128 GPU 集群上、使用 DeepSeek-V3 或类似规模的 MoE 模型进行验证，将是检验 Frontier 真正价值的关键实验

### 评分：⭐⭐⭐⭐（4/5）

- **创新性**：5/5 —— Stage-centric 抽象是真正的创新
- **技术深度**：4/5 —— 算子建模和 MoE 工作流建模有深度，但评估不够充分
- **实验验证**：2/5 —— 评估规模太小，缺乏关键消融和大规模验证
- **写作质量**：4/5 —— 动机清晰，方法描述有层次，图表质量高
- **影响力潜力**：5/5 —— 填补了行业的关键空白，有可能成为新一代 LLM 推理系统的事实标准仿真平台

---

## 相关链接

- [[Knowledge/论文分析/推理系统/LLM推理系统深度综述]]
- Vidur: A Large-Scale Simulation Framework for LLM Inference
- DistServe: Disaggregating Prefill and Decode for LLM Serving
- MegaScale-Infer: Serving MoE at Scale
- Step-3 is Large Yet Affordable

## 参考文献（本文引用）

1. Qwen2 Technical Report, 2024.
2. Nvidia Dynamo, 2025. https://github.com/ai-dynamo/dynamo
3. Nvidia TensorRT-LLM, 2025. https://github.com/NVIDIA/TensorRT-LLM
4. Agrawal et al., Vidur. MLSys 2024.
5. Agrawal et al., Sarathi-Serve. OSDI 2024.
6. Breiman, Random Forests. Machine Learning, 2001.
7. Cho et al., LLMServingSim. IISWC 2024.
8. Guo et al., DeepSeek-R1. arXiv 2025.
9. Kwon et al., vLLM / PagedAttention. SOSP 2023.
10. Li et al., Lina. USENIX ATC 2023.
11. Liu et al., DeepSeek-V3 Technical Report. arXiv 2024.
12. Mitra et al., Beyond the Buzz (Inference Disaggregation). arXiv 2025.
13. Singh et al., Hybrid Tensor-Expert-Data Parallelism for MoE. ICS 2023.
14. StepFun, Step-3. arXiv 2025.
15. Zheng et al., SGLang. NeurIPS 2024.
16. Zhong et al., DistServe. OSDI 2024.
17. Zhu et al., MegaScale-Infer. arXiv 2025.
