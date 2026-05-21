---
tags:
- 论文分析
- recommendation-system
- deep-learning-recommendation-model
- dlrm
source: https://github.com/facebookresearch/dlrm
arxiv: 1906.00091
authors: Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan
  Sundaraman, Jongsoo Park, Xiaodong Wang et al.
institutions: Facebook (Meta)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# DLRM: Deep Learning Recommendation Model for Personalization and Recommendation Systems

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Deep Learning Recommendation Model for Personalization and Recommendation Systems |
| **arXiv** | 1906.00091 |
| **机构** | Facebook/Meta |
| **代码** | https://github.com/facebookresearch/dlrm |

### 核心贡献

1. **推荐系统的深度学习标准模型**：定义了稀疏特征（Embedding）+ 密集特征（MLP）+ 特征交互（Dot Product）的三层架构
2. **统一基准**：替代 Wide&Deep/DeepFM 等碎片化的 RecSys 架构，提供可复现的 PyTorch 实现
3. **性能分析框架**：系统性分析了 DLRM 在 CPU/GPU 上的性能特征

---

## 二、技术架构

DLRM 包含三大模块：

| 模块 | 功能 | 计算特征 |
|------|------|---------|
| **Embedding Tables (稀疏)** | 类别特征的 embedding lookup | 内存带宽密集，稀疏访问 |
| **Bottom MLP (密集)** | 数值特征的变换 | 计算密集 (GEMM) |
| **Feature Interaction (交叉)** | Embedding 间的 dot product | 计算密集（二次型）|
| **Top MLP (密集)** | 交互后的最终分类 | 计算密集 (GEMM) |

### 性能特征

**Embedding 查表阶段**：
- 瓶颈在 GPU HBM 带宽，而非算力
- Batch size 增大 → 带宽压力超线性增长
- 10-100+ 个 Embedding Table，每请求只查极少数

**MLP 阶段**：
- GEMM 计算密集，GPU 利用率高
- 随着 batch size 增大接近理论峰值算力

---

## 相关链接
- [[Knowledge/论文分析/推荐系统/DLRM 训练性能模型分析]]
- [[Knowledge/论文分析/推荐系统/推荐系统性能建模综述]]
