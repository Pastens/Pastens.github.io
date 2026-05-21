---
tags:
- 论文分析
- llm-training
- survey
- literature-review
- distributed-training
- parallelism
- memory-optimization
created: 2026-05-11
---

# LLM 训练系统深度技术综述：并行策略、内存优化与通信架构

> 涵盖 21 篇论文 | 时间跨度：2018.06 – 2025.12

## 一、领域全景图

LLM 训练系统研究的四大维度：

- **并行策略框架**（Parallelism）：Megatron-LM, FSDP, Alpa, FlexFlow, TeraPipe
- **内存优化**（Memory Optimization）：ZeRO 系列, LOMO, GaLore, 激活检查点
- **通信与调度**（Communication & Scheduling）：TACCL, DeepSpeed Ulysses, Sequence Parallelism, Ring Attention
- **MoE 训练系统**（MoE Training）：Tutel, MiCS, Megatron-Core MoE
- **微调加速**（Fine-tuning Systems）：LOMO, GaLore, ZeRO-Offload

## 二、并行策略框架

### 数据并行（Data Parallelism, DP）
- 最早、最基础的并行方式，每 GPU 一份完整模型副本，梯度 AllReduce
- PyTorch DDP → FSDP (ZeRO-3 的思路) 的演进路径

### 张量并行（Tensor Parallelism, TP）
- 将单个 Transformer 层的权重矩阵沿 hidden 维切分
- **Megatron-LM** 奠基之作（1D TP），后续演变出 2D/3D TP（序列并行）
- 需节点内高带宽互联（NVLink），跨节点通信成本高

### 流水线并行（Pipeline Parallelism, PP）
- 按层切分：不同 GPU 负责不同层的计算
- **GPipe** → **PipeDream** → **1F1B** → **TeraPipe** 的演进路径
- 核心挑战：气泡率（bubble ratio）与负载均衡

### 序列并行（Sequence Parallelism, SP）
- 将 attention 维度沿序列长度切分
- **DeepSpeed Ulysses** / **Col-CoTran** / **Ring Attention**

### MoE 专家并行（Expert Parallelism, EP）
- 专家放置在不同 GPU 上，路由门控决定 token 分配
- **Tutel** / **MiCS** / **DeepSpeed-MoE**

## 三、技术谱系

| 类别 | 方法 | 代表性工作 | 核心难点 |
|------|------|-----------|---------|
| **数据并行** | 梯度 AllReduce | DDP → FSDP → ZeRO-DP | 通信开销随 GPU 数线性增长 |
| **张量并行** | 层内矩阵切分 | Megatron-LM (1D-TP) | 需节点内 NVLink，跨节点受限 |
| **流水线并行** | 按层切分 + 调度 | GPipe/1F1B/PipeDream | 气泡率 (>15-50%)、负载均衡 |
| **序列并行** | 序列维度切分 | Ulysses/Ring Attention/SP | 通信模式复杂，跨节点效率 |
| **专家并行** | 专家放置 + 路由 | Tutel/MiCS | 负载不均衡、动态性、通信 |

## 四、各方向深度分析

### 4.1 数据并行 → 混合精度 → ZeRO 演进

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **ZeRO** | 1910.02054 | 2019 | ZeRO-DP (P_g, P_os, P_p) 三阶段、ZeRO-R（内存卸载） |
| **ZeRO-Offload** | 2101.06840 | 2021 | 将优化器状态+梯度卸载至 CPU |
| **ZeRO-Infinity** | 2104.07857 | 2021 | 将模型参数/梯度/优化器状态全卸载至 NVMe |
| **ZeRO++** | 2306.10209 | 2023 | 分级通信：量化 + 分层 All2All + 直觉感知路由 |
| **FSDP** | 2304.11277 | 2023 | PyTorch 原生 ZeRO-3 实现，混合分片+反向预取 |

### 4.2 张量并行与混合并行

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **Megatron-LM** | 2104.04473 | 2021 | 1D-TP + PP + DP 混合并行，层内 Transformer 切分方案 |
| **Alpa** | 2201.12023 | 2022 | 自动并行编译器：ILP 优化 intra/inter-operator 并行 |
| **FlexFlow** | 1807.05358 | 2019 | 超越 DP/MP 二分法，SOAP 搜索最优并行执行计划 |

### 4.3 流水线并行调度

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **GPipe** | 1811.06965 | 2019 | 同步流水线并行，micro-batch 分割 + 梯度累积 |
| **PipeDream** | 1806.03377 | 2018 | 异步 1F1B 调度，减少气泡率的非对称流水线 |
| **TeraPipe** | 2102.07988 | 2021 | Token 级流水线并行，在序列维度做 PP |

### 4.4 长序列训练

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **Sequence Parallelism** | 2105.13120 | 2021 | 序列维度切分 attention + 通信优化 |
| **DeepSpeed Ulysses** | 2309.14509 | 2023 | Asymmetric All-to-All + ZeRO 集成，支撑百万级序列 |
| **Ring Attention** | 2310.01889 | 2023 | 类似 Ring AllReduce 的环形序列通信，无限上下文 |

### 4.5 MoE 训练系统

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **Tutel** | 2206.03382 | 2022 | 动态自适应 MoE 训练框架 + Drop Tolerance 策略 |
| **MiCS** | 2205.00119 | 2022 | 公有云上近线性扩展，等级化 EP 通信优化 |

### 4.6 通信优化

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **TACCL** | 2111.04867 | 2021 | 通信草图驱动的集合通信算法自动合成 |

### 4.7 微调/内存优化

| 工作 | arXiv | 年份 | 核心贡献 |
|------|-------|------|---------|
| **LOMO** | 2306.09782 | 2023 | 降级优化器到 SGD + 融合梯度计算，全参微调省 10.8× 内存 |
| **GaLore** | 2403.03507 | 2024 | 梯度低秩投影，训练内存降 65%，媲美 Adam 精度 |
| **GaLore 2** | 2504.20437 | 2025 | 扩展到大规模预训练场景，支持 8B 以上 |


## 五、横向对比

### 训练系统 vs 推理系统

| 维度 | 训练系统 | 推理系统（参考） |
|------|---------|----------------|
| **核心瓶颈** | 通信 vs 计算 | 显存带宽 vs 计算 |
| **并行策略** | DP+TP+PP+SP+EP 五维空间 | TP+PP+EP 为主 |
| **仿真工具** | 匮乏（TGS 等不成熟） | 丰富（Vidur, LLMServingSim 等）|
| **系统仿真精度** | 尚无公认高精度训练仿真器 | LLMServingSim 2.0 达 0.95% |
| **主流硬件** | A100/H100/B200 + InfiniBand | 同左，但更关注推理专用硬件 |
| **KV Cache** | 不适用 | 核心瓶颈 |
| **梯度通信** | AllReduce 核心瓶颈 | N/A |

### 关键观察

1. **训练系统仿真严重落后于推理**：推理有 GenZ (5.8%), LLMServingSim 2.0 (0.95%) 等高精度建模，而训练系统仿真仍以实验测量为主，缺乏公认的统一框架
2. **五维并行空间**：训练并行策略搜索空间巨大 (DP × TP × PP × SP × EP)，手动调优成本高昂
3. **Alpa 标杆**：Alpa 的自动并行方案启发大量后续工作，但实际工业界仍主要依赖 Megatron-LM 的手动配置
4. **MoE 训练仍是开放问题**：负载不均衡 + 动态路由 + 专家通信，现有方案各有取舍
5. **长序列训练的兴起**：Ring Attention 和 Ulysses 代表了两种不同的设计哲学（环形 vs 全对全）

## 六、推荐工具链

| 用户角色 | 推荐组合 |
|---------|---------|
| **训练工程师** | Megatron-Core (基础设施) → ZeRO (显存优化) → DeepSpeed Ulysses (长序列) |
| **系统研究者** | Alpa (自动并行) → FlexFlow (搜索空间) → TACCL (通信优化) |
| **MoE 研究者** | Tutel (动态性) → MiCS (缩放性) → Megatron-Core (生产级) |
| **微调优化者** | GaLore (预训练) → LOMO (全量微调) → ZeRO-Offload (资源受限) |

## 七、未来趋势

1. **训练仿真工具的出现**：训练系统尚缺类似 GenZ/Vidur 级的仿真框架，这是重要的空白
2. **PD 分离思想在训练中的延伸**：推理中 PD 分离思路启发「训练-推理生命周期统一调度」
3. **异构训练**：H100+B200+GB200 混合集群的自动并行分配
4. **FP8 训练精度仿真**：低精度训练对收敛性和最终精度的影响建模
5. **自动化并行编译器**：Alpa 思路的工业化落地

## 论文速查表

| # | 论文 | arXiv | 年份 | 类型 | 核心指标 | GitHub |
|---|------|-------|------|------|---------|--------|
| 1 | **Megatron-LM** | 2104.04473 | 2021 | TP+PP 框架 | 335B LLaMA 在 2048 A100 训练 | NVIDIA/Megatron-LM ⭐9.6K |
| 2 | **ZeRO** | 1910.02054 | 2019 | 内存优化 | 万亿参数训练，8× 吞吐提升 | microsoft/DeepSpeed ⭐37K |
| 3 | **ZeRO-Offload** | 2101.06840 | 2021 | 内存优化 | 单 GPU 训练 100B 模型 | microsoft/DeepSpeed |
| 4 | **ZeRO-Infinity** | 2104.07857 | 2021 | 内存优化 | NVMe 卸载，200T 模型 | microsoft/DeepSpeed |
| 5 | **ZeRO++** | 2306.10209 | 2023 | 通信优化 | AllReduce 量化，~50% 加速 | microsoft/DeepSpeed |
| 6 | **FSDP** | 2304.11277 | 2023 | DP 框架 | PyTorch 原生 ZeRO-3 | PyTorch 内置 |
| 7 | **Alpa** | 2201.12023 | 2022 | 自动并行 | 自动 ILP 搜索最优并行 | alpa-proj ⭐2.9K |
| 8 | **FlexFlow** | 1807.05358 | 2019 | 并行搜索 | 超越 DP/MP, SOAP 搜索 | flexflow ⭐2.7K |
| 9 | **GPipe** | 1811.06965 | 2019 | PP 调度 | 微批 + 梯度累积，SOSP 2019 | - |
| 10 | **PipeDream** | 1806.03377 | 2018 | PP 调度 | 1F1B 异步调度，SOSP 2019 | - |
| 11 | **TeraPipe** | 2102.07988 | 2021 | PP 调度 | Token 级流水线，近零气泡 | - |
| 12 | **Sequence Parallelism** | 2105.13120 | 2021 | 序列并行 | 序列维度切分 | ColossalAI |
| 13 | **DeepSpeed Ulysses** | 2309.14509 | 2023 | 序列并行 | 百万级 token 训练 | microsoft/DeepSpeed |
| 14 | **Ring Attention** | 2310.01889 | 2023 | 序列并行 | 环形通信，近无限上下文 | haoliu-1999/RingAttention |
| 15 | **Tutel** | 2206.03382 | 2022 | MoE 训练 | 动态自适应 MoE | microsoft/tutel |
| 16 | **MiCS** | 2205.00119 | 2022 | MoE 训练 | 公有云近线性 MoE 扩展 | - |
| 17 | **TACCL** | 2111.04867 | 2021 | 通信优化 | 通信草图自动合成通信算法 | - |
| 18 | **LOMO** | 2306.09782 | 2023 | 微调优化 | 全参微调省 10.8× 显存 | OpenAI/LOMO ⭐6K |
| 19 | **GaLore** | 2403.03507 | 2024 | 微调优化 | 梯度低秩投影，降 65% 内存 | jiaweizzhao/GaLore |
| 20 | **GaLore 2** | 2504.20437 | 2025 | 预训练优化 | 扩展到大模型预训练 | - |

## 相关笔记

- [[LLM推理系统深度综述]]
- [[Megatron-LM 技术分析]]
- [[ZeRO 技术分析]]
- [[ZeRO-Offload 技术分析]]
- [[ZeRO-Infinity 技术分析]]
- [[ZeRO++ 技术分析]]
- [[FSDP 技术分析]]
- [[Alpa 技术分析]]
- [[FlexFlow 技术分析]]
- [[GPipe 技术分析]]
- [[PipeDream 技术分析]]
- [[TeraPipe 技术分析]]
- [[DeepSpeed Ulysses 技术分析]]
- [[Sequence Parallelism 技术分析]]
- [[Ring Attention 技术分析]]
- [[Tutel 技术分析]]
- [[MiCS 技术分析]]
- [[TACCL 技术分析]]
- [[LOMO 技术分析]]
- [[GaLore 技术分析]]
