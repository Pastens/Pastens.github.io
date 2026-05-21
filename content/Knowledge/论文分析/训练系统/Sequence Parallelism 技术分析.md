---
tags:
- 论文分析
- llm-training
- sequence-parallelism
- long-context
arxiv: 2105.1312
authors: Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, Yang You
institutions: NUS, ColossalAI
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# Sequence Parallelism: Long Sequence Training from System Perspective

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Sequence Parallelism: Long Sequence Training from System Perspective |
| **arXiv** | 2105.13120 |
| **机构** | NUS (ColossalAI 团队) |
| **代码** | ColossalAI (内置) |

### 核心贡献

1. **序列维度切分**：在序列维度上切分 attention + FFN 层
2. **通信优化**：与 TP 共享通信域，减少额外通信开销
3. **ColossalAI 集成**：作为 ColossalAI 框架的内置序列并行方案

---

## 二、技术方法

### 序列切分

将序列 `[B, S, D]` 沿 S 维度切分到 N 个 GPU：
- 每个 GPU 持有 S/N 个 token
- Attention 中通过 AllGather 收集完整序列做 softmax
- FFN 中每个 token 独立计算，无需通信

### 与 TP 的组合

SP 与 TP 使用相同的通信域，在 TP 的通信 (AllReduce) 中嵌入 SP 的 AllGather，实现零额外通信开销。

---

## 相关链接
- [[Knowledge/论文分析/训练系统/DeepSpeed Ulysses 技术分析]]
- [[Knowledge/论文分析/训练系统/Ring Attention 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
