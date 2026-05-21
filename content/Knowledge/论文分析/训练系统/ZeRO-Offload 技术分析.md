---
tags:
- 论文分析
- llm-training
- memory-optimization
- cpu-offload
source: https://github.com/microsoft/DeepSpeed
arxiv: 2101.0684
authors: Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan
  Yang, Minjia Zhang, Dong Li, Yuxiong He
institutions: Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# ZeRO-Offload: Democratizing Billion-Scale Model Training

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | ZeRO-Offload: Democratizing Billion-Scale Model Training |
| **arXiv** | 2101.06840 |
| **机构** | Microsoft |
| **代码** | DeepSpeed (built-in ZeRO-Offload) |

### 核心贡献

1. **CPU 卸载架构**：将优化器状态和梯度卸载到 CPU 内存，单 GPU 训练 100B 参数模型
2. **计算-通信-卸载 overlap**：三流异步执行，隐藏卸载延迟
3. **选择性卸载（Selective Offload）**：仅卸载"高计算量、低带宽敏感"部分

---

## 二、技术方法

### 卸载策略

- **优化器状态** → CPU：Adam momentum/variance 整周期更新，适合 CPU
- **梯度** → CPU：反向传播中分批传输
- **参数** → GPU：保留在 GPU，前向/反向频繁使用

### 三流异步架构

1. GPU 计算流：前向 + 反向
2. CPU-GPU 传输流：梯度/参数双向传输
3. CPU 计算流：优化器更新

通过异步 overlap 隐藏 ~95% 的传输延迟。

### 性能分析

单 GPU 场景（V100-32GB）：ZeRO-Offload 可训练高达 **100B 参数**模型，而原生 DP 受限于 1.5B。

---

## 相关链接
- [[Knowledge/论文分析/训练系统/ZeRO 技术分析]]
- [[Knowledge/论文分析/训练系统/ZeRO-Infinity 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
