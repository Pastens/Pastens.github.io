---
tags:
- 论文分析
- chip-modeling
- cpu-simulator
- full-system-simulation
source: https://www.gem5.org/
conference: ISCA 2011
institutions: Michigan, Texas, Wisconsin, MIT, ARM, AMD, Google, etc.
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
permalink: gem5
---

# gem5: The gem5 Simulator

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | The gem5 Simulator |
| **发表地** | ISCA 2011 (Computer Architecture Letters 2011) |
| **机构** | Michigan, Texas, Wisconsin, MIT, ARM, AMD, Google, HP Labs |
| **代码** | https://gem5.googlesource.com (gem5.org) |

### 核心贡献

1. **M5 (Michigan) + GEMS (Wisconsin) 合并** → gem5：前者的 CPU 模型 + 后者的内存子系统
2. **多 ISA 支持**：ARM/x86/RISC-V/MIPS/SPARC/ALPHA，单一代码库
3. **Ruby 内存系统**：灵活的 coherence protocol 建模（MOESI/MESI/Directory）
4. **开源社区基础设施**：30+ 学术机构贡献，标准验证套件

---

## 二、技术架构

### 核心组件

| 组件 | 功能 | 可选性 |
|------|------|--------|
| **CPU 模型** | Atomic (功能级), Timing (时序简单的), O3 (out-of-order) | 可替换 |
| **Ruby** | 缓存一致性协议 + 互连网络 | 可选 |
| **Classic Memory** | Simple memory hierarchy | 默认 |
| **System Emulation (SE)** | 功能级，仅模拟用户态 | ✓ |
| **Full System (FS)** | 全系统，运行真实 OS/kernel | ✓ |

---

## 三、应用与影响

- 学术引用 10,000+，计算机体系结构论文的默认仿真平台
- 支持：单核→多核→众核→GPU (VEGA) → RISC-V Vector
- 性能：~1-5 MIPS (百万指令/秒)，O3 模式最慢
- 2025 年最新：gem5 24.0，GCN3 GPU, ARMv9, CHI 协议

---

## 相关链接
- [[GPGPU-Sim 技术分析]]
- [[芯片性能建模与仿真深度综述]]
