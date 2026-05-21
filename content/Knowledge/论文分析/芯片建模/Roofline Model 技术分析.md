---
tags:
- 论文分析
- chip-modeling
- roofline
- analytical-modeling
conference: CACM 2009
institutions: UC Berkeley (Samuel Williams, Andrew Waterman, David Patterson)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Roofline Model: An Insightful Visual Performance Model

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Roofline: An Insightful Visual Performance Model for Multicore Architectures |
| **发表** | Communications of the ACM (CACM), 2009 |
| **机构** | UC Berkeley (ParLab, David Patterson) |
| **引用** | 4000+ (计算机体系结构领域最高引论文之一) |

### 核心贡献

1. **运算强度（Operational Intensity）** 统一分类：计算密集 vs 内存密集
2. **Roofline 可视化**：二维图直观显示性能瓶颈
3. **跨架构适用性**：从 CPU → GPU → TPU → NPU 广泛应用

---

## 二、核心模型

### 基本公式

```
Attainable GFLOP/s = min(Peak GFLOP/s, BW × OI)
```

- OI (Operational Intensity) = FLOPs / Bytes（floating-point ops per byte DRAM traffic）
- 天花板 = min(compute ceiling, memory ceiling)

### 意义

Roofline 的贡献不仅在公式本身，更在于它是**联结硬件架构瓶颈与算法优化的桥梁**：它让程序员和架构师用同一张图理解性能。

---

## 三、扩展

| 变体 | 年份 | 改进 |
|------|------|------|
| **Cache-aware Roofline** | 2014 | 引入 cache 层次使 OI 依赖于工作集大小 |
| **Communication-avoiding Roofline** | 2019 | 引入通信成本维度 |
| **FP8 Roofline** | 2024 | 适配低精度训练推理 |

---

## 相关链接
- [[Knowledge/论文分析/芯片建模/gem5 技术分析]]
- [[Knowledge/论文分析/芯片建模/芯片性能建模与仿真深度综述]]
