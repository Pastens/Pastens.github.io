---
tags:
  - 论文分析
  - llm-training
  - sequence-parallelism
  - long-context
  - ring
arxiv: 2310.01889
authors: Hao Liu, Matei Zaharia, Pieter Abbeel
institutions: UC Berkeley
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Ring Attention with Blockwise Transformers for Near-Infinite Context

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Ring Attention with Blockwise Transformers for Near-Infinite Context |
| **arXiv** | 2310.01889 |
| **机构** | UC Berkeley (Hao Liu, Matei Zaharia, Pieter Abbeel) |
| **代码** | https://github.com/haoliu-1999/RingAttention |

### 核心贡献

1. **环形通信**：GPU 间以环形方式传输 KV blocks，类似 Ring AllReduce 思想
2. **近无限上下文**：N 个 GPU 可将 context 扩展到单 GPU 的 N 倍
3. **Blockwise 并行计算**：每个 GPU 同时计算本地 attention + 接收外部 KV block
4. **无 O(P²) 通信**：每个 GPU 仅与邻居通信，通信量 O(P)

---

## 二、技术方法

### 环形通信

N 个 GPU 组成一个环：
1. 每 GPU 持有序列的 S/N 长度
2. 迭代轮转：每轮将本地 KV block 传给下一个 GPU
3. 每个 GPU 累计处理完整的 N 个 KV block
4. 共 N-1 轮后完成全部 attention 计算

### 计算-通信 Overlap

Blockwise 计算允许每个 GPU 在处理当前 KV block 的同时，异步传输未使用的 block，隐藏传输延迟。

### 实验

在 8-64 A100 上，训练 **4M token** 序列：
- 64 GPU 上扩展到 4M 上下文，吞吐保持线性的 85%+
- 在 1M 序列上保持 <5% 的训练精度损失

---

## 三、个人评价

Ring Attention 的创新在于将 Ring AllReduce 的思路引入 attention 的序列并行。与 Ulysses 的 All-to-All 方案相比，Ring Attention 减少了全局同步，在跨节点场景中更具扩展性。两者互补，Ulysses 适合节点内高带宽，Ring Attention 适合跨节点扩展。

## 相关链接
- [[DeepSpeed Ulysses 技术分析]]
- [[Sequence Parallelism 技术分析]]
- [[LLM训练系统深度综述]]
