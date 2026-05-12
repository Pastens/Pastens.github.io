---
tags:
  - 论文分析
  - chip-modeling
  - ml-based-modeling
  - performance-prediction
conference: MICRO 2018
institutions: MIT (Larry Rudolph, Una-May O'Reilly)
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks |
| **发表** | MICRO 2018 (Best Paper Nomination) |
| **机构** | MIT CSAIL |

### 核心贡献

1. **RNN-based 指令吞吐预测**：取代传统的静态分析（如 Intel IACA）
2. **精度优势**：平均 <10% 误差，优于 IACA（~15%）
3. **可移植性**：训练后可在不同微架构间迁移（但精度会下降）

---

## 二、方法

### 模型架构

```
x86指令序列 → Hierarchical LSTM → 基本块特征向量 → MLP → IPC预测
```

- 编码：每指令编码为 200 维向量（opcode, registers, immediates）
- 训练数据：100,000+ 基本块，通过 IACA/simulator 标注

### 局限

- 仅预测基本块级吞吐（完整程序的 IPC 预测需额外分析）
- 对长基本块（>100 指令）精度下降

---

## 三、意义

Ithemal 开启了 **"用 ML 替代微架构仿真"** 的方向，后续工作如 LiteSim (HPCA 2023) 将其思想扩展到全程序级性能预测。

---

## 相关链接
- [[芯片性能建模与仿真深度综述]]
