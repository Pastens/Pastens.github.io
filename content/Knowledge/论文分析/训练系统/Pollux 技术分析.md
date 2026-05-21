---
tags:
- 论文分析
- 训练系统
- 性能建模仿真
- 集群调度
arxiv: 2008.1226
title: 'Pollux: 协同自适应集群调度'
authors: Aurick Qiao, Sang Keun Choe, Suhas Jayaram Subramanya, Willie Neiswanger,
  Qirong Ho, Hao Zhang, Gregory R. Ganger, Eric P. Xing
institutions: Carnegie Mellon University, Petuum Inc.
created: 2026-05-12
rating: ⭐⭐⭐⭐⭐
permalink: pollux
---

# Pollux: 协同自适应的Goodput优化深度学习集群调度

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning |
| **arXiv** | 2008.12260 (OSDI 2021) |
| **机构** | Carnegie Mellon University, Petuum Inc. |
| **源码** | https://github.com/petuum/pollux |

### 核心贡献

1. **Goodput作为调度目标**：提出goodput（有效吞吐量 = 统计效率 × 系统吞吐量）作为联合优化目标，同时考虑训练效率和系统性能
2. **协同适配（Co-adaptation）**：动态联合优化每个job的并行度配置（workers数+batch size）和集群资源分配，两者相互影响
3. **WS吞吐模型**：基于Walling-time（wall time per iteration）和Scaling efficiency的解析吞吐量预测模型

---

## 二、技术方法详解

### 2.1 核心洞察

现有调度器的关键缺陷：**job的并行度配置与集群调度决策相互孤立**。
- 固定并行度（用户指定）→ 资源无法充分利用
- 固定资源（调度器分配）→ 忽略了模型训练的batch size/workers数对收敛的影响

Pollux**同时优化**这两个维度，形成协同适配的正反馈循环。

### 2.2 Goodput定义

**Goodput = Statistical Efficiency × System Throughput**，其中：
- **Statistical Efficiency**：每次梯度更新对模型精度的提升量，与batch size直接相关
- **System Throughput**：每秒梯度更新次数，由并行度和资源分配决定

### 2.3 吞吐量预测模型（WS模型）

每个job维护一个WS模型，刻画吞吐量T与workers数w的关系：

`T(w) = w / (w · τ₀ + τ₁)`

其中：
- τ₀：**walling time per iteration**（受计算和通信瓶颈影响，不随w缩放的部分）
- τ₁：**linear overhead**（与w线性相关的开销）

**在线更新**：训练过程中持续测量T(w)的实际值，用最优化方法更新τ₀和τ₁，使模型始终反映实时集群负载状况。

### 2.4 Optimus调度器

基于WS模型的调度决策：
- 为每个job计算所有合法w下的goodput预测值
- 求解全局资源分配优化问题，最大化总goodput
- **资源弹性调整**：job运行期间可动态增减workers（需要elastic训练支持）
- 支持**packing**：将多job的workers打包到同一GPU上

---

## 三、实验评估

### 实验设置
- 物理集群：4节点 × 8 V100 = 32 GPU
- 仿真集群：1024 GPU
- 模型：VGG-19, ResNet-50, ResNet-152, Transformer (NMT), DLRM
- 数据并行训练（Horovod/NCCL）

### 关键结果

| 指标 | Pollux vs 最佳基线 |
|------|-------------------|
| **平均Job完成时间** | **1.68× 加速** |
| **资源利用率** | 显著提升（packing动态填补空闲GPU） |
| **吞吐量预测精度** | MAPE中位数 < 10% |
| **大规模仿真**（1024 GPU） | 保持2×以上优势 |

### 消融实验关键发现
- 固定batch size + 动态资源：50% improvement over baseline
- 固定资源 + 动态batch：30% improvement
- **两者联合优化 = 68% improvement** → 协同效应明显

---

## 四、亮点与局限

### 亮点

1. **Goodput指标创新**：将统计效率（ML视角）和系统吞吐量（系统视角）统一为单一优化目标
2. **WS模型的优雅设计**：仅2个参数的解析模型，在线更新使模型自适应集群负载变化
3. **协同适配的正反馈**：更多资源 → 更大batch → 更快收敛 → 释放资源给其他job
4. **packing支持**：不同job共享GPU，进一步提升利用率

### 局限

1. 仅支持数据并行（DP），不支持模型并行/流水线并行
2. 弹性训练要求框架支持动态增减workers（非所有框架都支持）
3. goodput需要预设统计效率曲线，对快速演进的模型架构可能不准确

---

## 五、个人评价

Pollux是OSDI 2021的系统论文，代表了深度学习集群调度的一次范式转变。核心创新不在于提出新的调度算法，而在于**重新定义了调度目标**——将"最小化训练时间"扩展为"最大化有效训练产出（goodput）"。WS模型的简洁性（仅2个参数）与其有效性形成强烈对比，体现了系统性思维的价值：复杂的集群问题不一定需要复杂的方法。

与Gavel（OSDI 2020）相比：Gavel关注异构加速器选择，Pollux关注并行度-资源联合优化，两者正交互补。结合两者（异构感知 + goodput优化）是现代深度学习集群调度的一个重要方向。
