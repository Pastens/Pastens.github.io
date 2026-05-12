---
tags:
  - 论文分析
  - chip-modeling
  - power-modeling
  - mcpat
conference: MICRO 2008
institutions: HP Labs, UCSB
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# McPAT: An Integrated Power, Area, and Timing Modeling Framework

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | McPAT: An Integrated Power, Area, and Timing Modeling Framework for Multicore and Manycore Architectures |
| **发表** | MICRO 2008 |
| **机构** | HP Labs, UCSB |
| **引用** | 3500+ |

### 核心贡献

1. **CPU 全芯片功耗+面积建模**：从 core → cache → NoC → memory controller
2. **CMOS 工艺缩放**：支持 7nm-32nm（但未更新至 FinFET）
3. **动态+漏电功耗**：活动因子驱动的功耗计算框架

---

## 二、方法

### 输入/输出

- **输入**：处理器配置参数 + 活动因子（来自仿真器的寄存器翻转统计）
- **输出**：动态功耗 (mW) + 漏电功耗 (mW) + 总面积 (mm²)

### 精度

与 Intel 65nm 芯片实测对比，平均误差 <25%

### 局限

- 未更新至 7nm/5nm FinFET 工艺
- 活动因子准确度高度依赖外部仿真器

---

## 相关链接
- [[gem5 技术分析]]
- [[芯片性能建模与仿真深度综述]]
