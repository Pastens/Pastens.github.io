---
tags:
- 论文分析
- llm-training
- parallelism
- automatic-parallelism
source: https://github.com/alpa-proj/alpa
arxiv: 2201.12023
authors: Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping
  Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Ion Stoica, Joseph E. Gonzalez, Eric
  P. Xing, Hao Liu
institutions: UC Berkeley, Stanford, Google, Petuum
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
permalink: alpa
---

# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning |
| **arXiv** | 2201.12023 |
| **机构** | UC Berkeley, Stanford, Google |
| **代码** | https://github.com/alpa-proj/alpa (⭐2.9K) |
| **顶会** | OSDI 2022 |

### 核心贡献

1. **自动并行编译**：输入 JAX 计算图 → 自动输出并行化执行计划
2. **两级搜索空间分割**：Inter-Operator 和 Intra-Operator 分离优化
3. **ILP + DP 组合搜索**：Intra-layer 层面用 ILP 做 TP/PP/DP 分配，Inter-layer 层面用 DP 做 PP 切分
4. **支持任意并行模式组合**：自动选择最优的 DP/TP/PP 组合方案

---

## 二、技术方法详解

### 2.1 整体流程

JAX 计算图 → MLIR HLO → 计算图分析 → 两级并行搜索 → NumPy 设备执行

### 2.2 层级并行（Intra-Operator Parallelism）

对计算图中的每个算子（如 matmul, attention），搜索：
- **Tensor Parallelism (TP)**：算子内切分
- **自动决定**：切分维度、通信方案、AllReduce vs AllGather 的选择

使用 **ILP (Integer Linear Programming)** 建模，目标：最小化单层的执行时间。

### 2.3 跨层级并行（Inter-Operator Parallelism）

将模型在不同 GPU 之间切分为多个阶段（Pipeline stages）：
- **动态规划**搜索最优切分点
- 考虑计算时间、通信时间、流水线气泡
- 决定每个 stage 的 GPU 数量

### 2.4 实验验证

| 模型 | Alpa 吞吐 | 手动调优基准 | 加速比 |
|------|-----------|-------------|--------|
| GPT-3.5B (64 GPU) | 54% MFU | Megatron-LM 52% | ~1.04× |
| GPT-175B (128 GPU) | 45% MFU | Megatron-LM 45% | ~1.0× |
| MoE-3.5B (64 GPU) | 43% MFU | N/A | 自动最优 |

---

## 三、亮点与局限

### 亮点
- **用户无需手动配置并行策略**：从 JAX 代码自动生成高效并行计划
- **统一优化 TP/PP/DP**：解决手动搜索指数级组合的问题
- **OSDI 2022 论文**：学术认可度高

### 局限
- **JAX 生态限制**：仅支持 JAX，不支持 PyTorch
- **搜索时间较高**：大模型搜索需数十分钟
- **不支持 MoE 的 EP**：未集成 Experts Parallelism

---

## 四、个人评价

Alpa 是自动并行编译领域的标杆工作。它的 ILP+DP 两级搜索框架清晰优雅，将 LLM 训练从手动配置推向自动化。局限在于 JAX-only 和搜索耗时，但在 GPU 集群成本和人力调优成本面前，搜索引擎的算力消耗是值得的。

## 相关链接
- [[Megatron-LM 技术分析]]
- [[LLM训练系统深度综述]]
