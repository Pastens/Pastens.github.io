---
tags:
- 论文分析
- llm-training
- parallelism
- pipeline-parallelism
arxiv: 1806.03377
authors: Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil
  Devanur, Greg R. Ganger, Phil Gibbons
institutions: CMU, Microsoft Research
created: 2026-05-11
rating: ⭐⭐⭐⭐
permalink: pipedream
---

# PipeDream: Fast and Efficient Pipeline Parallel DNN Training

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | PipeDream: Fast and Efficient Pipeline Parallel DNN Training |
| **arXiv** | 1806.03377 |
| **机构** | CMU, Microsoft Research |
| **顶会** | SOSP 2019 |

### 核心贡献

1. **1F1B 调度**：One-Forward-One-Backward，反向一准备好就执行，减少气泡
2. **异步流水线**：不同 stage 使用不同版本的权重（weight stashing）
3. **非对称切分**：不要求各 stage 计算量相同，自动匹配计算资源

---

## 二、技术方法

### 2.1 1F1B 调度

与 GPipe 的 "所有前向 → 所有反向" 不同，PipeDream 在每完成一个 micro-batch 的前向后立即开始该 micro-batch 的反向，前提是需要的参数版本已就绪。

气泡率 = `(K - 1) / (M + K - 1)`，在 K=4, M=8 时比 GPipe 低 2-3%。

### 2.2 Weight Stashing

每个 micro-batch 的反向需要使用与其前向相同版本的权重，避免梯度不一致。通过 stash 多个版本的权重实现。

---

## 三、与 GPipe 对比

| 特性 | GPipe | PipeDream |
|------|-------|-----------|
| 更新模式 | 同步 | 异步 |
| 气泡率 | (K-1)/M | (K-1)/(M+K-1) |
| 权重版本 | 统一 | Weight Stashing |
| 语义 | 严格等价 | 近似等价 |
| 吞吐 | 低气泡+同步 | 高吞吐+异步 |

---

## 相关链接
- [[GPipe 技术分析]]
- [[TeraPipe 技术分析]]
- [[LLM训练系统深度综述]]
