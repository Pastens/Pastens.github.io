---
tags:
- 论文分析
- llm-training
- memory-optimization
- gradient-low-rank
arxiv: 2403.03507
authors: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar,
  Yuandong Tian
institutions: Caltech, UT Austin, Meta AI (FAIR), MIT-IBM Watson
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
permalink: galore
---

# GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection |
| **arXiv** | 2403.03507 |
| **机构** | Caltech, UT Austin, Meta, MIT-IBM |

### 核心贡献

1. **梯度低秩投影**：将全参数梯度投影到低秩子空间，保持模型规模缩放不增优化器显存
2. **训练内存降 65%**：7B 模型 Adam 训练内存从 58GB → 20GB
3. **媲美 Adam 的精度**：与全参数 Adam 训练达到相同的收敛效果
4. **扩展到预训练**：支撑 7B 级别的大模型预训练

---

## 二、技术方法

### 为什么梯度是低秩的？

GaLore 的作者发现：在大型深度学习训练中，梯度的奇异值分布具有长尾特性，少量大奇异值贡献了大部分范数。因此，可将梯度投影到低秩子空间后再使用 Adam 优化，大幅降低优化器状态（momentum/variance）的存储量。

### 与现有方案对比

| 方法 | 训练内存 (7B) | 与 Adam 精度差距 | 可做预训练 |
|------|--------------|----------------|-----------|
| Adam Full | ~58GB | 基准 | ✅ |
| **GaLore** | **~20GB** | <1% | ✅ |
| LOMO | ~7.8GB | 5-10% (SGD) | ❌ |
| LoRA | ~14GB | 2-5% | ❌ |

### GaLore 2 (2504.20437)

GaLore 2 进一步扩展到更大规模的预训练：
- 支持 8B 参数的预训练
- 引入分块低秩策略，降低通信开销
- 在 C4/Wikipedia 等数据集上验证

---

## 相关链接
- [[LOMO 技术分析]]
- [[LLM训练系统深度综述]]
