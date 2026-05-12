---
tags:
  - 论文分析
  - chip-modeling
  - npu-simulator
  - systolic-array
source: https://github.com/ARM-software/SCALE-Sim
arxiv: 1811.02883
institutions: Georgia Tech, ARM, NVIDIA
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# SCALE-Sim: Systolic CNN Accelerator Simulator

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | SCALE-Sim: Systolic CNN Accelerator Simulator |
| **arXiv** | 1811.02883 (v1, 2018); ISPASS 2020 发表 |
| **机构** | Georgia Tech (Ananda Samajdar, Yuhao Zhu) |
| **代码** | https://github.com/ARM-software/SCALE-Sim |

### 核心贡献

1. **脉动阵列（Systolic Array）开源的周期精确仿真器**
2. **支持三种主流数据流**：Weight Stationary (WS)、Output Stationary (OS)、Input Stationary (IS)
3. **TPU 精度验证**：与 Google TPUv1 真实数据对比，误差 <15%
4. **SCALE-Sim v3/TPU** (2603.22535): 扩展到 TPU 架构建模

---

## 二、技术方法

### 建模层次

| 层级 | 组件 | 描述 |
|------|------|------|
| 顶层 | Loop nest | Nested loop 映射到脉动阵列的仿真 |
| 中间 | Buffer hierarchy | Global Buffer → Local Registers → MAC |
| 底层 | MAC array | N×M systolic array, 周期级数据流动 |

### 数据流（Dataflow）支持

| 数据流 | 特点 | 适合 |
|--------|------|------|
| **Weight Stationary** | 权重固定在 MAC 阵列，输入数据流动 | 卷积计算（TPU 风格）|
| **Output Stationary** | 部分和固定在 MAC，权重流动 | 大核卷积 |
| **Input Stationary** | 输入固定，权重移动 | 小核卷积 |

---

## 三、扩展

- **SCALE-Sim v2**: 异构脉动阵列 + DRAM 控制器
- **SCALE-Sim v3/TPU**: 面向 TPU 架构，验证 TVM/VTA 的周期级行为
- 与 **Timeloop/MAESTRO** 互补：SCALE-Sim 做周期验证，MAESTRO 做映射分析

---

## 相关链接
- [[MAESTRO 技术分析]]
- [[芯片性能建模与仿真深度综述]]
