---
tags:
  - 论文分析
  - chip-modeling
  - gpu-simulator
  - function-timing-decoupling
source: https://github.com/accel-sim/accel-sim-framework
conference: MICRO 2020
institutions: UBC, UTRGV
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Accel-Sim: Function-Timing Decoupled GPU Simulation

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Accel-Sim: Simulating GPU Acceleration for AI and HPC Workloads |
| **发表地** | MICRO 2020 |
| **机构** | UBC, University of Texas Rio Grande Valley |
| **代码** | https://github.com/accel-sim |

### 核心贡献

1. **功能-时序解耦**：GPU 功能模型（通过 NVIDIA 驱动）与性能模型分离
2. **真实驱动集成**：使用 NVIDIA 原生驱动采集执行 trace，消除功能模拟偏差
3. **精度**：**<10% 平均误差** — GPU 仿真器中的最佳精度之一
4. **速度**：比 GPGPU-Sim 快 2-4×

---

## 二、技术方法

### 架构

| 组件 | 说明 |
|------|------|
| **Trace Driven** | 通过 RAPIDS/cuda-memcheck 记录 GPU kernel trace |
| **Timing Model** | 独立可插拔的 timing model，解耦自 functional correctness |
| **Memory Model** | 详细建模 L1/L2/DDR + HBM |
| **Issue Logic** | Warp issue, scoreboard, scheduler |

### 与 GPGPU-Sim 对比

| 特性 | GPGPU-Sim | Accel-Sim |
|------|-----------|-----------|
| 功能模型 | 内置 (PTX→仿真的functional) | 外部 (真实驱动) |
| 时序模型 | 耦合 | 解耦 |
| 精度 | ~20% | **<10%** |
| 速度 | 10-50 KIPS | 30-80 KIPS |
| 新架构支持 | 需数年 | 可由驱动更新 |

---

## 相关链接
- [[GPGPU-Sim 技术分析]]
- [[芯片性能建模与仿真深度综述]]
