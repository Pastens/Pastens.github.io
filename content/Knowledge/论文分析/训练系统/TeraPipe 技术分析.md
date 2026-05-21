---
tags:
- 论文分析
- llm-training
- parallelism
- pipeline-parallelism
arxiv: 2102.07988
authors: Zhuohan Li, Siyuan Zhuang, Shiyuan Guo, Danyang Zhuo, Hao Zhang, Dawn Song,
  Ion Stoica
institutions: UC Berkeley
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models |
| **arXiv** | 2102.07988 |
| **机构** | UC Berkeley |

### 核心贡献

1. **Token 级流水线并行**：在序列维度上的不同 token 间做 PP，而非层间
2. **近零气泡率**：token 级细粒度分割几乎消除流水线气泡
3. **兼容 Megatron TP**：与张量并行正交组合

---

## 二、技术方法

### 2.1 Token 级调度

Causal LM（GPT 风格）的序列中，token t 的计算依赖 token 1..t-1。TeraPipe 利用这种自回归依赖结构，在序列中划分多个 token 子序列，每个 GPU 处理一个子序列。

### 2.2 气泡率对比

| 方法 | 气泡率公式 | 典型值 (L=96, N=16, S=2048) |
|------|-----------|---------------------------|
| GPipe | (K-1)/M | ~50% |
| PipeDream (1F1B) | (K-1)/(M+K-1) | ~30% |
| TeraPipe | ~0 | **<1%** |

### 2.3 实验

在 16 GPU 上训练 GPT-6.7B 模型：
- 与 Megatron-LM TP 组合后，吞吐提升 4.6×
- 气泡率接近理论下界

---

## 相关链接
- [[Knowledge/论文分析/训练系统/GPipe 技术分析]]
- [[Knowledge/论文分析/训练系统/PipeDream 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
