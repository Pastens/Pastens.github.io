---
tags:
- 论文分析
- llm-training
- moe
- expert-parallelism
arxiv: 2205.00119
authors: Zhen Zhang, Shuai Zheng, Yida Wang, et al.
institutions: Amazon Web Services (AWS)
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud |
| **arXiv** | 2205.00119 |
| **机构** | AWS |

### 核心贡献

1. **公有云上的近线性缩放**：受限于云网络带宽的环境下实现高效 MoE 训练
2. **等级化 EP**：将专家分片到 GPU 组内，组间 DP，组内 EP
3. **通信拓扑折衷**：用有限云带宽实现接近理论峰的扩展效率

---

## 二、技术方法

### 等级化通信

云计算环境的最大挑战是带宽不足且不均衡。MiCS 设计两级的 EP 通信：
- **GPU 组内（节点内）**：EP All-to-All 通信（NVLink 高带宽）
- **组间（跨节点）**：DP AllReduce（较少的通信量）

通过将大部分通信保持在节点内，大幅降低跨节点带宽需求。

### 实验

在 64-256 GPU 的 AWS 集群上，实现 85-90% 的弱扩展效率。

---

## 相关链接
- [[Knowledge/论文分析/训练系统/Tutel 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
