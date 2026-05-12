---
tags:
  - 论文分析
  - recommendation-system
  - distributed-training
  - gpu
arxiv: 2210.08803
authors: Joey Wang, Yingcan Wei, Minseok Lee, Khaled Al Farisi, Wenchen Li, Swapnil Joshi, Narayanan Sundaraman, Sunghwan Kim et al.
institutions: NVIDIA
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference |
| **arXiv** | 2210.08803 |
| **机构** | NVIDIA |
| **代码** | (NVIDIA Merlin 框架的一部分) |

### 核心贡献

1. **GPU 端 Embedding 分片训练**：将 TB 级 Embedding 表分布到多 GPU
2. **Model Parallelism for Embedding**：支持按 table/row/column 三种分片策略
3. **All2All 通信优化**：Embedding 训练特有的 All-to-All 通信模式
4. **Triton 推理部署**：与 NVIDIA Triton Inference Server 集成

---

## 二、技术架构

### Embedding 分片策略

| 策略 | 划分方式 | 适用场景 |
|------|---------|---------|
| **Table-Wise** | 整个表在不同 GPU | 表数 > GPU 数 |
| **Row-Wise** | 表内行哈希分片 | 单表极大（TB 级） |
| **Column-Wise** | 表内列/维度分片 | 维度极大（1000+） |

### 训练流程

```
CPU Pipeline: 数据加载 → Embedding 分片 → All2All 通信 → GPU MLP 训练
GPU Pipeline: 接收分片 Embedding → Bottom MLP → Feature Interaction → Top MLP
```

### 性能

与 CPU baseline 相比，训练吞吐提升 **40-60×**（在 8×A100 上）。

---

## 相关链接
- [[推荐系统/DLRM 技术分析]]
- [[推荐系统/推荐系统性能建模综述]]
