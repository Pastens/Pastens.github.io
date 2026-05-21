---
tags:
- 论文分析
- 训练系统
- 性能建模仿真
- 集群调度
arxiv: 2008.09213
title: 'Gavel: 异构感知集群调度策略'
authors: Deepak Narayanan, Keshav Santhanam, Fiodar Kazhamiaka, Amar Phanishayee,
  Matei Zaharia
institutions: Stanford University, Microsoft Research
created: 2026-05-12
rating: ⭐⭐⭐⭐⭐
permalink: gavel
---

# Gavel: 异构感知的深度学习集群调度策略

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads |
| **arXiv** | 2008.09213 (OSDI 2020) |
| **机构** | Stanford University, Microsoft Research |
| **源码** | https://github.com/stanford-futuredata/gavel (Rust) |

### 核心贡献

1. **异构感知调度框架**：首次系统性地将加速器异构性（GPU/TPU/FPGA之间的性能差异）纳入DL集群调度策略设计中
2. **有效吞吐量（Effective Throughput）**：将"分配矩阵A"（jobs × accelerators）与"每job在每类加速器上的吞吐量"结合，形式化为优化问题
3. **空间共享感知**：支持GPU多进程空间共享（MPS/MIG），能在job之间按比例分配GPU资源
4. **轮次调度机制**：支持分时调度和空间共享的混合调度策略
5. **吞吐预测器**：基于profile数据预测job在各类加速器上的性能

---

## 二、技术方法详解

### 2.1 核心框架

Gavel的核心抽象是**分配矩阵A**，其中A[i][k]表示分配给第i个job的第k类加速器的资源比例（0-1之间的连续值）。多种调度策略均可表示为：

```
maximize/minimize  f(A, S)
subject to  Σᵢ A[i][k] ≤ 1, ∀k  (不超过每种加速器总容量)
            0 ≤ A[i][k] ≤ 1       (非负分配)
```

其中S是所有job的预期完成时间集合。

### 2.2 支持的调度策略

| 策略 | 目标 | 数学表达 |
|------|------|---------|
| **最大吞吐 (Max-Throughput)** | 最大化集群吞吐 | max Σᵢ Σₖ A[i][k] × perf[i][k] |
| **最小完成时间 (Min-Completion)** | 所有job尽快完成 | min maxᵢ S[i] |
| **最短剩余时间优先 (SRTF)** | 优先短job | min Σᵢ S[i] |
| **公平策略 (Fairness)** | 一切片各job等量并行 | S[i] = S[j], ∀i,j |
| **加权公平 (W-Fairness)** | 按权重分配 | perf_weighted 分配 |
| **Gittins Index** | 最优MDP调度 | 不确定性的Bayesian最优 |

### 2.3 吞吐量估计器

- 对每个job在每类加速器上运行10-20步profile
- 使用profile数据构建**资源-性能曲线**
- 预测job在任意资源分配下的吞吐量

---

## 三、实验评估

### 物理集群实验

| 策略 | 加速比（vs Best Static） |
|------|------------------------|
| MaxProd (round-robin) | 1.0-1.2× |
| Gavel MaxThroughput | **1.3-1.5×** |
| Gavel MinCompletion | **1.2-1.4×** |

### 大规模仿真

- 在Trace驱动的1.5年仿真中验证策略有效性
- 异构感知策略在各种工作负载分布下均优于同构感知基线

---

## 四、亮点与局限

### 亮点

1. **问题定义清晰**：首次将加速器异构性纳入DL调度
2. **通用框架**：多种调度策略可在统一框架下实现
3. **实验充分**：物理集群+大规模仿真双验证
4. **直接影响后续工作**：为Sia、Pallavi等异构调度工作奠定基础

### 局限

1. 假设job性能profile可提前获取
2. 未考虑网络通信对异构集群拓扑的影响
3. 未考虑训练过程中的性能动态变化

---

## 五、个人评价

Gavel是异构DL集群调度的奠基之作（OSDI 2020），提出了将加速器异构性纳入调度优化的范式。其核心洞察——"不同模型在不同加速器上性能差异巨大"——在当今GPU代际加速迭代的背景下尤为关键。有效吞吐量的数学框架清晰优雅，为后续Sia、Pallavi等工作奠定了基础。
