---
tags:
- 论文分析
- llm-training
- moe
- expert-parallelism
source: https://github.com/microsoft/tutel
arxiv: 2206.03382
authors: Changhoon Kim, Sehoon Kim, Dan Zhang, Song Han, Kurt Keutzer
institutions: UC Berkeley, Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
permalink: tutel
---

# Tutel: Adaptive Mixture-of-Experts at Scale

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Tutel: Adaptive Mixture-of-Experts at Scale |
| **arXiv** | 2206.03382 |
| **机构** | UC Berkeley, Microsoft |
| **代码** | https://github.com/microsoft/tutel |

### 核心贡献

1. **动态自适应 MoE 框架**：支持 2D/3D/4D 并行分片策略
2. **Drop Tolerance**：智能动态跳过非关键专家（部分节省约 30% 计算）
3. **负载均衡训练**：通过 auxiliary loss + 动态路由实现专家负载均衡

---

## 二、技术方法

### 并行策略

Tutel 支持灵活的 EP（Expert Parallelism）组合：
- 2D: DP + EP （最常用）
- 3D: DP + EP + TP
- 4D: DP + EP + TP + PP

### Drop Tolerance

在训练中，若某专家在 batch 中未被分配到 tokens，该专家在那一轮的计算可被跳过。Tutel 利用这一特性设计动态 Drop Tolerance 策略：在达到某个 token 覆盖率阈值后，主动跳过剩余专家。

---

## 相关链接
- [[MiCS 技术分析]]
- [[LLM训练系统深度综述]]
