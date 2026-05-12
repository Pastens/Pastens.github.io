---
tags:
  - 论文分析
  - llm-training
  - parallelism
  - automatic-parallelism
arxiv: 1807.05358
authors: Zhihao Jia, Matei Zaharia, Alex Aiken
institutions: Stanford, UC Berkeley
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# FlexFlow: Beyond Data and Model Parallelism for Deep Neural Networks

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Beyond Data and Model Parallelism for Deep Neural Networks |
| **arXiv** | 1807.05358（初版）/ 2022 SysML 版 |
| **机构** | Stanford, UC Berkeley |
| **代码** | https://github.com/flexflow/FlexFlow |

### 核心贡献

1. **SOAP 搜索空间**：将并行策略分解为 S（样本）/O（算子）/A（属性）/P（参数）四个可切分维度
2. **自动搜索最优并行计划**：通过仿真+分析模型组合搜索
3. **超越 DP/MP 二分法**：展示了"混合并行"的完整搜索空间

---

## 二、技术方法

### SOAP 并行空间

| 维度 | 含义 | 示例 |
|------|------|------|
| **S (Sample)** | 样本维度 | Data Parallelism |
| **O (Operator)** | 算子间并行 | Pipeline Parallelism |
| **A (Attribute)** | 属性/维度切分 | Tensor Parallelism |
| **P (Parameter)** | 参数维度 | 参数分片 (ZeRO style) |

FlexFlow 在 SOAP 四维空间中使用 **MMD (Minotaur Discovery)** 算法搜索最优方案。

---

## 相关链接
- [[Alpa 技术分析]]
- [[LLM训练系统深度综述]]
