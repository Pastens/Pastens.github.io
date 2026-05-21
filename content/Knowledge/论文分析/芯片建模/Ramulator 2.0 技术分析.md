---
tags:
- 论文分析
- chip-modeling
- memory-simulation
- dram
source: https://github.com/CMU-SAFARI/ramulator2
arxiv: 2308.1103
conference: IEEE CAL 2023
institutions: CMU SAFARI, ETH Zurich
created: 2026-05-11
rating: ⭐⭐⭐⭐
permalink: ramulator-20
---

# Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator |
| **arXiv** | 2308.11030 |
| **发表** | IEEE Computer Architecture Letters (CAL), 2023 |
| **机构** | CMU SAFARI (Haocong Luo et al.) |
| **代码** | https://github.com/CMU-SAFARI/ramulator2 |

### 核心贡献

1. **模块化设计**：协议层 / 控制器层 / 设备层完全解耦
2. **支持最新标准**：DDR4/DDR5/HBM2e/HBM3/LPDDR5/GDDR6
3. **可扩展性**：支持新型存储技术（PIM / CXL 内存 / OpenCAPI）
4. **速度**：~500 kHz（比 DRAMsim3 快 2-3×）

---

## 二、架构

### 三层模块化

| 层级 | 功能 | 可替换性 |
|------|------|---------|
| **Device Layer** | DRAM 芯片时序参数（tRCD/tCL/tRAS…） | ✅ 每芯片类型配置 |
| **Controller Layer** | 调度策略（FR-FCFS, PAR-BS…） | ✅ 可插拔调度器 |
| **Protocol Layer** | DDR/DDR5/HBM 协议时序 | ✅ 协议模块切换 |

### 精度验证

在 DDR4-3200 和 HBM2e 上验证，延迟误差 <5%

---

## 相关链接
- [[芯片性能建模与仿真深度综述]]
