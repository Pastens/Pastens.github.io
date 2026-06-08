---
tags:
- 论文分析
- llm-training
- sequence-parallelism
- long-context
source: https://github.com/microsoft/DeepSpeed
arxiv: 2309.14509
authors: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen
  Leon Song, Samyam Rajbhandari, Yuxiong He
institutions: Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models |
| **arXiv** | 2309.14509 |
| **机构** | Microsoft |
| **代码** | DeepSpeed (内置) |

### 核心贡献

1. **Asymmetric All-to-All 通信**：在序列维度使用 All-to-All 通信，替代 AllReduce
2. **与 ZeRO-3 无缝集成**：Ulysses 与 ZeRO 的显存优化共存
3. **百万级序列训练**：理论上支持数百万 token 的序列长度
4. **高效通信模式**：通信复杂度 O(P) 而非 O(P²)

---

## 二、技术方法

### 通信方案

传统 Ring Attention：环形 AllReduce 通信，延迟随节点数线性增长。

Ulysses 的核心新思路：**Asymmetric All-to-All**。每个 GPU 持有序列的一个子段和注意力的一个子集，通过 All-to-All 交换数据使得每个 GPU 获得全局序列的局部注意力。

- 前向：All-to-All（分散）+ All-to-All（收集）
- 反向：类似的对称通信

### 与 ZeRO 集成

Ulysses 的通信与 ZeRO-3 参数 AllGather 独立可 overlap，总通信开销近似等于二者的 max。

### 实验

在 64 A100 上：
- 序列长度从 32K 扩展到 1M
- 相比 baseline 的吞吐保持率 > 85%
- 在超长序列场景下优于 Ring Attention

---

## 相关链接
- [[Sequence Parallelism 技术分析]]
- [[Ring Attention 技术分析]]
- [[LLM训练系统深度综述]]
