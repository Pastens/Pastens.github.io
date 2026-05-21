---
tags:
- 论文分析
- llm-training
- parallelism
- pipeline-parallelism
arxiv: 1811.06965
authors: Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen,
  HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen
institutions: Google
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism |
| **arXiv** | 1811.06965 |
| **机构** | Google |
| **顶会** | NeurIPS 2019 |

### 核心贡献

1. **Pipeline Parallelism 奠基**：将模型按层切分到多个 GPU 上，微批分割（micro-batching）
2. **同步梯度更新**：保持模型权重的严格语义等效性
3. **自动重计算**：反向时重放前向计算，节省激活值显存

---

## 二、技术方法

### 2.1 流水线并行

将 L 层模型分为 K 个 stage，分到 K 个 GPU：
- **Micro-batch 切分**：输入 batch 分为 M 个微批
- **前向流水线**：第 i 个 micro-batch 依次流过 K 个 stage
- **反向流水线**：全部 M 个 micro-batch 的前向完成后，再反向

### 2.2 气泡率分析

GPipe 的气泡率 = `(K - 1) / M`。当 M >> K 时，气泡率趋近于 0。但 M 增大也意味着更多的显存需求（存储微批的中间激活值）。

### 2.3 实验

在 256 个 Cloud TPU 上训练 AmoebaNet-8B：
- 相比单 GPU 的 79× 加速
- 1024 TPU 上 557× 加速（效率 54%）

---

## 三、亮点与局限

### 亮点
- **简单易用**：K 个 GPU 划分 clear
- **语义等效**：同步更新保证严格等价

### 局限
- **气泡率大**：M=K 时 50%，需 M >> K 缓解
- **显存压力**：M 大 → 中间激活值多

---

## 相关链接
- [[Knowledge/论文分析/训练系统/PipeDream 技术分析]]
- [[Knowledge/论文分析/训练系统/TeraPipe 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
