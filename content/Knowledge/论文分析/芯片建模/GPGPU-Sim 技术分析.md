---
tags:
- 论文分析
- chip-modeling
- gpu-simulator
- gpgpu
source: https://github.com/gpgpu-sim/gpgpu-sim_distribution
conference: MICRO 2009
institutions: UBC (Simon Fraser → Fatahalian)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
permalink: gpgpu-sim
---

# GPGPU-Sim: A Performance-Cycle Accurate Simulator for GPU Computing

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | A Performance-Cycle Accurate Simulator for GPU Computing |
| **发表地** | MICRO 2009 |
| **机构** | UBC / Simon Fraser / NVIDIA / AMD |
| **代码** | https://github.com/gpgpu-sim |

### 核心贡献

1. **GPU 周期精确仿真奠基**：首个公开可用的 GPGPU 性能仿真器
2. **架构覆盖**：从 Fermi (CC 2.0) → Turing (CC 8.6) → RDNA
3. **PTX + SASS 双模式**：高级中间表示 + 原生 ISA 精度
4. **完整的 GPU 微架构建模**：Warp scheduler, SIMT stack, shared memory, memory hierarchy

---

## 二、技术架构

### 建模组件

| 组件 | 功能 |
|------|------|
| **Shader Core** | SIMT 单元：warp scheduling, register file, SIMD datapath |
| **Memory Partition** | GDDR/hbm memory控制器, L2 cache |
| **Interconnection** | Crossbar/Mesh 片上网络 |
| **DRAM Model** | GDDR3/5/HBM 时序仿真 |

### 精度与速度

- 平均误差：13-28%（取决于应用）
- 仿真速度：~10-50 KIPS（500-1000× 慢于真实 GPU）
- 支持 OpenCL / CUDA 应用

---

## 三、影响与局限

**影响**：GPGPU-Sim 催生了 1000+ 篇 GPU 微架构研究论文，成为 GPU 体系结构研究的默认仿真平台。

**局限**：
- 功能模型与时序模型耦合 → 维护困难，无法快速跟进 NVIDIA 新架构
- 不支持 Tensor Core / RT Core 等专用单元
- 仿真速度极慢
- 2020 年后社区逐渐迁移到 Accel-Sim（解耦架构）

---

## 相关链接
- [[Accel-Sim 技术分析]]
- [[芯片性能建模与仿真深度综述]]
