---
tags:
- 论文分析
- llm-training
- fine-tuning
- memory-optimization
- lomo
arxiv: 2306.09782
authors: Kai Lv, Yuqing Yang, Tengxiao Liu, Qingyi Tao, Qiao Jin, Xin Gao, Yiming
  Yang
institutions: Tsinghua, Microsoft (original LOMO authors were from OpenNLPLab)
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# LOMO: Full Parameter Fine-tuning for Large Language Models with Limited Resources

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Full Parameter Fine-tuning for Large Language Models with Limited Resources |
| **arXiv** | 2306.09782 |
| **代码** | https://github.com/OpenNLPLab/LOMO (⭐6K+) |

### 核心贡献

1. **融合梯度计算（Fused Gradient Computation）**：同时计算梯度并更新参数，不需要存储完整梯度
2. **SGD 级显存优化**：将优化器从 Adam 降级为 SGD，只保留参数本身
3. **10.8× 显存节省**：7B 模型仅需 7.8GB GPU 显存即可全量微调
4. **与 LoRA 正交**：可与 LoRA 等 PEFT 方法叠加

---

## 二、技术方法

### 融合梯度计算

传统流程：反向传播 → 存储梯度 → 优化器更新。LOMO 在反向传播的每层结束后立即执行 SGD 参数更新，然后丢弃该层梯度。

这样只需要 O(1) 的梯度存储空间（仅当前层），而非 O(N) 的完整梯度存储。

### 显存对比 (7B 模型)

| 方法 | Adam | SGD | 显存需求 |
|------|------|-----|---------|
| Full FT (Adam) | 28GB | - | 56+ GB（OOM）|
| Full FT (SGD) | - | 14GB | 42+ GB |
| **LOMO** | - | **融合 SGD** | **7.8GB** |
| LoRA | 小 | - | 14GB |

---

## 相关链接
- [[GaLore 技术分析]]
- [[LLM训练系统深度综述]]
