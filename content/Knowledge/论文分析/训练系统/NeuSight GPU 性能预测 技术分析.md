---
tags:
  - 论文分析
  - 训练系统
  - 性能建模仿真
source: ''
arxiv: '2407.13853'
authors: 'Seonho Lee, Amar Phanishayee, Divya Mahajan'
institutions: 'Georgia Institute of Technology, Meta'
created: 2026-05-12
rating: ⭐⭐⭐⭐⭐
---

# NeuSight — 深度学习训练与推理的GPU性能预测

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Forecasting GPU Performance for Deep Learning Training and Inference |
| **arXiv** | 2407.13853 (ASPLOS 2025) |
| **机构** | Georgia Institute of Technology, Meta |
| **代码** | https://github.com/sitar-lab/NeuSight (开源) |
| **DOI** | 10.1145/3669940.3707265 |

### 核心贡献

1. **NeuSight框架**：首个无需目标GPU实际执行即可预测DL模型（训练+推理）在**未见GPU**上端到端性能的框架，支持跨代和跨厂商（NVIDIA + AMD）预测。

2. **Tile粒度分解策略**：将深度学习kernel的执行分解为tile粒度，利用GPU的tiled执行特性将复杂预测问题拆解为更小的子问题，避免直接端到端预测的非线性建模困难。

3. **性能定律约束的ML建模**：创新性地仅用ML模型预测GPU设备利用率（utilization），再通过Roofline性能定律（峰值算力、峰值带宽）约束tile级延迟，确保预测结果有物理上限保障。

4. **与基线模型相比的重大改进**：在GPT3 on H100这一完全未见过的场景上，将预测错误率从121.4%和30.8%（基线模型）降低到**2.3%**；在跨所有模型和GPU的综合评估中达到8.9%的平均错误率，而MLP基线为140%，线性回归基线为60.8%。

5. **分布式场景支持**：扩展到单服务器多GPU（NVLink/DGX）的分布式训练预测，支持数据并行、张量模型并行和流水线并行，平均错误率5.4%。

---

## 二、技术方法详解

### 2.1 核心洞察

NeuSight的核心洞察源于GPU上DL kernel的实际执行机制：

1. **Tiled执行**：现代GPU库（cuDNN、CUTLASS）将kernel的输出矩阵分解为若干相同大小的tile，每个tile独立映射到一个Streaming Multiprocessor (SM)上执行
2. **多波次执行**：当tile数量超过SM数量时，GPU以多波（wave）方式执行，每波在所有SM上并行运行
3. **性能定律**：任何一个kernel的执行延迟受限于GPU的峰值算力和峰值内存带宽（Roofline模型）
4. **延迟隐藏**：GPU的利用率随可用线程数（即波数）增加而提升，呈现非线性关系

### 2.2 方法架构

NeuSight的工作流分为三个主要步骤：

**步骤1：Kernel级预测（核心创新）**

对于每个DNN kernel（GEMM、全连接层、element-wise、Softmax、LayerNorm等）：

1. **Tile分解**：将kernel的输出矩阵按tile粒度分解，tile大小通过PyTorch Profiler从kernel名称的元数据中提取或匹配已有数据库
2. **Tile数量与波数计算**：
   - `num_tiles = Π⌈xᵢ/tᵢ⌉` (各维度tile数的乘积)
   - `num_waves = ⌈num_tiles / num_sm⌉`
   - `PerOpLatency = PerTileLatency × num_waves`
3. **利用率预测**：用MLP预测GPU在当前kernel下的利用率，公式为：
   - `utilization = alpha - beta/num_waves`
   - 其中 `alpha, beta = σ(MLP(input_features))`
   - 通过sigmoid函数将预测值约束在[0,1]区间
4. **性能定律约束**：
   - `achievedBW = rooflineBW × utilization`
   - `PerTileLatency = flops_tile / achievedBW`
5. **输入特征**：每个tile的FLOPs、每SM的峰值FLOPs、每tile内存需求、每SM内存带宽、L2缓存/内存大小与工作集的比值、算术强度比等5个特征

**步骤2：单设备端到端延迟聚合**

使用Torch.fx提取模型的operator图（包括operator类型和张量维度），按数据流图顺序累加每个kernel的预测延迟得到单GPU端到端延迟。支持operator fusion（如element-wise加法与LayerNorm融合、GEMM与激活函数融合）。

**步骤3：分布式执行预测**

根据用户指定的并行策略（数据并行、张量模型并行、流水线并行），在计算图中插入对应的通信操作（all-reduce、send/recv），基于NVLink带宽估计通信延迟并累加。目前支持单服务器（NVLink/DGX）内分布式执行。

### 2.3 ML模型设计

- **架构**：5个独立的8层MLP（512隐藏单元），分别对应BMM、全连接层、element-wise、Softmax、LayerNorm
- **优化器**：AdamW with L2正则化
- **损失函数**：对称平均绝对百分比误差（SMAPE）
- **训练数据**：在5个较老的GPU（P4, P100, V100, T4, A100-40GB）上收集，共约15万+数据点
- **GPU特征**：仅使用公开可获取的硬件参数（内存大小、内存带宽、峰值FLOPS、SM数量、L2缓存大小）

---

## 三、实验评估

### 实验设置

| 维度 | 详情 |
|------|------|
| **训练GPU** | P4, P100, V100, T4, A100-40GB (2016-2020) |
| **测试GPU** | A100-80GB, L4, H100 (2022-2023) —— 完全未见过的GPU |
| **各厂商验证** | AMD MI100, MI210 (训练), MI250 (测试) |
| **模型** | BERT-Large, GPT2-Large, GPT3-XL, GPT3-2.7B, OPT-1.3B, Switch Transformer (5.3B) |
| **基线** | Roofline分析, Habitat [MLP-based], Li et al. [线性回归] |
| **精度** | FP32 (主实验), FP16/ Tensor Core (适应性实验) |

### 关键结果

#### 1. 单设备端到端推理和训练

| 框架 | 推断错误率 | 训练错误率 |
|------|-----------|-----------|
| Roofline | 31.2% | 31.9% |
| Habitat (MLP) | 220.9% | 725.8% |
| Li et al. (线性回归) | 61.2% | 58.3% |
| **NeuSight** | **9.7%** | **7.3%** |

#### 2. 未见GPU上的表现

| 框架 | 平均错误率 | 最大错误率 |
|------|-----------|-----------|
| Habitat | 724.3% | 4529.9% |
| Li et al. | 94.0% | 435.9% |
| **NeuSight** | **8.1%** | **28.2%** |

#### 3. GPT3 on H100 (两者皆未见)

| 场景 | Habitat | Li et al. | NeuSight |
|------|---------|-----------|----------|
| 训练 | 121.4% | — | **2.3%** |
| 推理 | 30.8% | — | **2.3%** |

#### 4. 跨厂商 (AMD GPU)

- 在AMD MI250上平均错误率：推理8.8%，训练15.7%
- 证明方法可泛化到不同GPU架构

#### 5. 分布式训练 (4-GPU)

- H100 DGX: 平均错误率 6.7%
- A100 NVLink: 平均错误率 10.5%
- 总体平均：**7.7%**（包含数据并行、张量模型并行、流水线并行）

---

## 四、亮点与局限

### 亮点

1. **极高的泛化能力**：在完全未见过的GPU（H100）和未见模型上，错误率仅为2.3%，这是性能预测领域的重大突破
2. **方法论创新**：将"直接预测延迟"的传统思路转变为"预测利用率+性能定律约束"，既保留了ML的拟合能力又保证了物理合理性
3. **数据效率高**：仅用5个老GPU上的~15万数据点就能准确预测新GPU，无需在新GPU上收集任何数据
4. **实用性强**：仅需公开的GPU规格参数（无需反向工程或专有信息），可立即用于预测最新发布（甚至未发布）的GPU
5. **跨厂商验证**：在AMD GPU上同样有效，证明方法不依赖特定厂商的硬件细节
6. **代码开源**：提供了完整的复现环境和数据集

### 局限

1. **分布式范围有限**：仅支持单服务器内多GPU场景，多服务器/跨网络的大规模分布式训练依赖于外部网络模拟器（如ASTRA-Sim）的集成
2. **精度格式局限**：主实验仅使用FP32，对FP16/BFloat16/Tensor Core的支持作为适应性实验展示，未系统评估
3. **Tile大小依赖**：预测时需要已知tile大小（从训练数据中匹配），对全新kernel或未记录的kernel可能匹配不准
4. **模型覆盖**：仅支持5类主要operator，其他operator（如embedding）简单视为访存受限，在处理新颖operator架构时可能需要扩展
5. **fused operator的简单处理**：对operator fusion的处理较为简化（累积FLOPs并丢弃中间结果内存），对复杂fusion模式的准确性待验证
6. **未覆盖推理优化**：未考虑KV-cache等LLM推理特有的优化技术对延迟的影响

---

## 五、个人评价

NeuSight是一篇非常出色的系统/架构论文。其核心贡献不在于ML模型本身，而在于**将GPU执行机制的先验知识（tiling、wave、roofline）与ML模型优雅地结合**，从而实现了远超纯ML方法和纯分析方法的预测精度。

从方法论的视角看，这篇论文的"分解-约束-聚合"范式具有广泛启示：
- **分解**（kernel → tile）让问题变得更简单、更规则
- **约束**（性能定律作为上限）确保预测不会违背物理规律
- **聚合**（tile → operator → model → distributed）让方法自然可扩展

这与近年来ML4Sys领域中"用ML替代物理建模"的主流思路形成鲜明对比——NeuSight证明了**领域知识和物理约束恰恰是ML泛化能力的关键保障**。

几个值得注意的技术洞见：
1. 论文揭示了纯ML预测kernel延迟的根本局限：kernel的执行涉及软件调优（如tiling策略、内存层级利用）与硬件微架构的复杂交互，ML模型在高维输入空间中的插值/外推能力不足以捕获这些交互
2. 利用率与波数的关系：`utilization = alpha - beta/num_waves` 这一公式既简单又富有洞察，抓住了GPU并行度与利用率之间的饱和增长关系
3. 特征工程的设计（如per-SM资源归一化、各种比率特征）体现了对GPU执行机制的深刻理解

作为ASPLOS 2025论文，这篇工作在方法创新、实验完整性、实际应用价值上均达到很高水准。它为GPU性能预测领域树立了新的基准，对各种需要预计算力规划、模型部署决策的场景都具有直接的应用价值。

> **一句话总结**：NeuSight通过"tile粒度分解 + ML预测利用率 + Roofline定律约束"的创新范式，首次实现了对未见GPU上DL模型性能的高精度预测（错误率<10%），是GPU性能建模领域的一个重要里程碑。
