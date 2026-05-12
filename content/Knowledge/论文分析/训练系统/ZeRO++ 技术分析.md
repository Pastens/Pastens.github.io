---
tags:
  - 论文分析
  - llm-training
  - memory-optimization
  - communication
source: https://github.com/microsoft/DeepSpeed
arxiv: 2306.10209
authors: Guanhua Wang, Heyang Qin, Sam Jacobs, Connor Holmes, Samyam Rajbhandari, Olatunji Ruwase, Feng Yan, Yuxiong He et al.
institutions: Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | ZeRO++: Extremely Efficient Collective Communication for Giant Model Training |
| **arXiv** | 2306.10209 |
| **机构** | Microsoft |

### 核心贡献

1. **量化通信（Quantized AllReduce）**：FP16 → INT4/INT8 压缩梯度
2. **分层 All-to-All**：减少 AllGather 通信量，利用节点内 NVLink
3. **分区化参数通信**：每个 GPU 只拉取自己需要的分片

---

## 二、技术方法

### 三大优化

| 优化 | 技术 | 效果 |
|------|------|------|
| qAllReduce | 梯度量化 + 分层聚合 | 通信量减半 |
| hpzAllGather | 分层点对点 AllGather | 消除等传输效应 |
| pzAllToAll | 分区 All-to-All | 减少跨节点传输 |

在 384 A100 上达 1.5× 加速，通信开销降低 40-60%。

---

## 相关链接
- [[ZeRO 技术分析]]
- [[LLM训练系统深度综述]]
