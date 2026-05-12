---
tags:
  - 论文分析
  - chip-modeling
  - npu-simulator
  - dataflow
arxiv: 1805.02566
conference: HPCA 2018
institutions: Georgia Tech, NVIDIA
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# MAESTRO: Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows Using MAESTRO |
| **arXiv** | 1805.02566 |
| **发表** | HPCA 2018 + Emeralds 2018 |
| **机构** | Georgia Tech, NVIDIA |
| **代码** | https://github.com/maestro-project/maestro |

### 核心贡献

1. **数据重用距离（Data Reuse Distance）理论**：量化 DNN 数据流中的空间/时间重用
2. **硬件成本模型**：从映射自动计算所需 MAC 数、缓冲区容量、能耗
3. **与 Timeloop 互补**：MAESTRO 做数据流分析，Timeloop 做映射搜索

---

## 二、技术方法

### 数据重用分析

MAESTRO 将 DNN 映射分解为 Nesting & Temporal/Spatial loops，计算每层数据在不同级别的重用距离。基于重用距离估算：

| 指标 | 计算方式 |
|------|---------|
| **吞吐 (Throughput)** | 每个时钟周期的 MAC 操作数 |
| **内存访问 (DRAM BW)** | 基于重用距离计算 buffer miss |
| **能耗 (Energy)** | 数据在不同存储层级间搬运的能量成本 × 搬运次数 |

### 与 Timeloop/SCALE-Sim 分工

| 工具 | 职责 |
|------|------|
| **MAESTRO** | 给定映射 → 数据流分析 + 性能估算 |
| **Timeloop** | 搜索最优映射空间 |
| **SCALE-Sim** | 对选定映射做周期级验证 |

---

## 相关链接
- [[SCALE-Sim 技术分析]]
- [[芯片性能建模与仿真深度综述]]
