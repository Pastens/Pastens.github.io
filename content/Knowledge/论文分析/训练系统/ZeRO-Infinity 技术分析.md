---
tags:
- 论文分析
- llm-training
- memory-optimization
- nvme-offload
source: https://github.com/microsoft/DeepSpeed
arxiv: 2104.07857
authors: Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He
institutions: Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning |
| **arXiv** | 2104.07857 |
| **机构** | Microsoft |

### 核心贡献

1. **NVMe 卸载**：将 GPU → CPU → NVMe SSD 三级存储层级联合管理
2. **无限显存抽象**：模型大小不再受 GPU 显存限制
3. **200T 参数训练验证**：在 512 GPU 上验证超大规模可行性

---

## 二、技术方法

### 三级存储层次

| 层级 | 设备 | 带宽 | 容量 |
|------|------|------|------|
| L1 | GPU HBM | ~2 TB/s | 16-80GB |
| L2 | CPU DRAM | ~50 GB/s | 256GB-2TB |
| L3 | NVMe SSD | ~3-7 GB/s | 1-10TB |

### 核心创新

- **参数预取**：基于数据流分析预测参数访问模式，提前从 NVMe 加载
- **基于分区的卸载调度**：将模型分片成独立分区，流水线卸载
- **计算-IO overlap**：与 ZeRO-Offload 类似，三流异步

在 512 V100 GPU 上验证了 200T 参数训练的可行性。

---

## 相关链接
- [[Knowledge/论文分析/训练系统/ZeRO 技术分析]]
- [[Knowledge/论文分析/训练系统/ZeRO-Offload 技术分析]]
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
