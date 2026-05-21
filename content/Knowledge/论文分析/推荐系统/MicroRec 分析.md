---
tags:
- 论文分析
- recommendation-system
- inference-acceleration
- accelerator
arxiv: 2010.05894
authors: Wenqi Jiang, Zhenhao He, Shuai Zhang, Kai Zeng, Liang Feng, Jiansong Zhang,
  Tianqi Chen, Yaoyu Zhang, Jian Zhang, Xiaodong Wang, Gustavo Alonso
institutions: ETH Zurich
created: 2026-05-11
rating: ⭐⭐⭐⭐
permalink: microrec
---

# MicroRec: Efficient Recommendation Inference by Hardware and Data Structure Solutions

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | MicroRec: Efficient Recommendation Inference by Hardware and Data Structure Solutions |
| **arXiv** | 2010.05894 |
| **机构** | ETH Zurich (Systems Group) |
| **发表** | MLSys 2020 |

### 核心贡献

1. **硬件-数据结构联合优化**：同时优化 Embedding 布局和 GPU 执行模式
2. **Micro-Embedding**：将 Embedding 表拆为按访问频率排序的微型子表
3. **Micro-Interaction**：用 WMMA/TensorCore 批处理特征点积
4. **Codebook Cache**：高频 embedding 的 codebook 压缩

---

## 二、技术方法

### Micro-Embedding

传统 Embedding 表：L 个表，每表 N 行，每行 D 维，所有特征均匀。

**Micro-Embedding**：收集所有特征 ID 的全局频率，将高频特征组织为微型子表，低频特征移到 CPU。GPU 内部只保留高频子表，实现：

| 优化 | 效果 |
|------|------|
| 减少 Embedding 表总大小 | 50-70% |
| 提高 GPU HBM 利用率 | 30-50% |
| 减少 CPU-GPU 传输 | 60% |

### Micro-Interaction

传统特征交互：逐对做 dot product → warp 利用率极低（SIMT 效率 <30%）。

**Micro-Interaction**：将所有 dot product 打包为 WMMA(TensorCore) 操作，相当于：

```
打包前：N² × (向量点积) → warp 发散严重
打包后：WMMA(矩阵乘法) → TensorCore 满效率
```

---

## 三、效果

| 模型 | 加速比 | 内存缩减 |
|------|-------|---------|
| DLRM (Criteo) | 2.1× | 3.1× |
| DeepFM (Avazu) | 3.4× | 2.8× |
| Wide&Deep (Criteo) | 2.6× | 2.5× |

---

## 相关链接
- [[推荐系统/DLRM 技术分析]]
- [[推荐系统/推荐系统性能建模综述]]
