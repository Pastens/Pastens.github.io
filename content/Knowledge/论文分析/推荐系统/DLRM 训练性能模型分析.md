---
tags:
- 论文分析
- recommendation-system
- performance-modeling
- dlrm
arxiv: 2201.07821
authors: Zhongyi Lin, Louis Feng, Ehsan K. Ardestani, Jaewon Lee, John Moon, Chunpeng
  Wei, Sheng Li, Hao Wu, Yuening Zhang, Jin Kyu Kim, Wenhao Jia, Dheevatsa Mudigere,
  Maxim Naumov
institutions: Meta (Facebook)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Building a Performance Model for Deep Learning Recommendation Model Training on GPUs

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Building a Performance Model for Deep Learning Recommendation Model Training on GPUs |
| **arXiv** | 2201.07821 |
| **机构** | Meta (Facebook) |

### 核心贡献

1. **首个 DLRM 解析式性能模型**：将 DLRM 训练分解为 Embedding/MLP/Interaction 三个子模型
2. **精度**：迭代时间 <10% 误差，吞吐量 <8% 误差
3. **揭示 DLRM 与 NLP/CV 的根本差异**：DLRM 的 Embedding 查表是**带宽瓶颈**，替代了 LLM 的显存带宽瓶颈

---

## 二、性能模型详解

### 模型分解

```
T_total = T_emb_lookup + T_emb_comm + T_bottom_mlp + T_interaction + T_top_mlp
```

| 组件 | 公式 | 瓶颈资源 |
|------|------|---------|
| Embedding Lookup | `sum(max(emb_size/bw, flops/compute))` | HBM 带宽 |
| Embedding Communication | All2All 通信延迟 | PCIe/NVLink 带宽 |
| Bottom MLP | `sum((M*K + N*K) / compute)` | CUDA Core |
| Feature Interaction | `B × N_emb² × D / compute` | CUDA Core |
| Top MLP | `sum((M*K + N*K) / compute)` | CUDA Core |

### 关键洞察

1. **DLRM vs LLM**：LLM 训练 80%+ 在 MLP 上（算力瓶颈），DLRM 的 Embedding 查表可占 70% 的迭代时间
2. **Batch size 的双面性**：增大 batch size 提高 MLP 利用率，但 Embedding 带宽需求线性增长
3. **MLPerf 基准**：DLRM 在 MLPerf 训练中的独特定位

---

## 三、实验验证

| 模型配置 | GPU | 前向误差 | 反向误差 | 吞吐误差 |
|---------|-----|---------|---------|---------|
| Small DLRM (32GB) | V100 | 12% | 18% | 7% |
| Medium DLRM (64GB) | A100 | 14% | 16% | 8% |
| Large DLRM (128GB) | A100 | 15% | 19% | 9% |

---

## 四、亮点与局限

### 亮点
- 填补了 RecSys 训练性能建模的空白
- 揭示 DLRM Embedding 查表的带宽密集型本质
- 为硬件选型（HBM 容量/带宽）提供定量依据

### 局限
- 单 GPU 模型，未覆盖多 GPU 训练
- 假设 Embedding 访问均匀分布（实际不均匀）
- 未覆盖最新的 Transformer-based RecSys

---

## 相关链接
- [[Knowledge/论文分析/推荐系统/DLRM 技术分析]]
- [[Knowledge/论文分析/推荐系统/推荐系统性能建模综述]]
