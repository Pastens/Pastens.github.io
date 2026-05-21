---
tags:
- 论文分析
- llm-inference
- survey
- literature-review
- simulation
- hardware-architecture
- performance-modeling
created: 2026-05-09
---

# LLM 推理系统深度技术综述：仿真、建模与架构设计

> 涵盖 11 篇论文 | 时间跨度：2023.11 – 2026.03

## 一、领域全景图

LLM 推理系统研究的四象限分类：

- **仿真模拟**（Simulation）：Vidur, LLMServingSim 1.0/2.0, Frontier, APEX
- **在线仿真**（Online Emulation）：LLM-Emu
- **解析建模**（Analytical）: GenZ
- **系统设计与优化**：Splitwise, Mooncake, Sarathi-Serve, Debunk CUDA Myth

## 二、工具深度分析

### 仿真/模拟工具

| 工具 | 方法 | 精度 | 亮点 |
|------|------|------|------|
| **Vidur** (2405.05465) | 算子级 profiling 仿真 | <9% | ⭐597 Stars，微软出品，引用最高 |
| **LLMServingSim 1.0** (2408.05499) | HW/SW 协同，迭代级重用 | 14.7% | KAIST，可插拔 Roofline 模型 |
| **LLMServingSim 2.0** (2602.23036) | Profile+运行时 | **0.95%** ✅ | 最高精度，支持 PD 分离 |
| **Frontier** (2508.03148) | 精细化算子仿真 | TBD | 港中文/华为，支持 MoE |
| **APEX** (2411.17651) | 动态感知并行策略搜索 | TBD | Stanford/UC Merced |

### 在线仿真（已分析）
- **LLM-Emu** (2605.00616) — Profile-driven，运行真实 vLLM，<5% 误差

### 解析建模（已分析）
- **GenZ** (2406.01698) — Roofline 解析式，5.82% 误差，唯一支持 MoE+Mamba

### 系统设计

| 论文 | 核心贡献 | 影响力 |
|------|---------|--------|
| **Splitwise** (2311.18677) | Prefill/Decode 相分离奠基 | 极高 |
| **Mooncake** (2407.00079) | KV-centric 分离架构 | ⭐5285 Stars，FAST 2025 Best Paper |
| **Sarathi-Serve** (2403.02310) | Chunked-prefill 调度 | ⭐496 Stars，微軟 |
| **Debunk CUDA** (2501.00210) | Gaudi vs A100 系统评估 | ISCA-52 2025 |

## 三、横向对比

### 精度排名
1. **LLMServingSim 2.0** — 0.95%
2. **LLM-Emu** — <5%
3. **GenZ** — 5.82%
4. **Vidur** — <9%
5. **LLMServingSim 1.0** — 14.7%

### 架构支持对比
- **MoE 支持**：仅 GenZ + LLMServingSim 2.0 + Mooncake + Frontier
- **Mamba 支持**：仅 GenZ
- **Speculative Decoding**：仅 GenZ
- **EP（Expert Parallelism）**：仅 GenZ + LLMServingSim 2.0 + Mooncake

### 建模方法谱系
- **解析式**：GenZ (快速、可解释、适合 DSE)
- **仿真**：Vidur, LLMServingSim, Frontier, APEX (精度-速度折中)
- **Profile-driven**：LLM-Emu (高保真、依赖采集数据)
- **实验/部署**：Splitwise, Mooncake, Sarathi-Serve (真实系统验证)

## 四、关键洞察与趋势

### 趋势 1：仿真→真实系统运行的演进
从纯仿真 (Vidur/LLMServingSim) 到 "真实代码 + 定点替换" (LLM-Emu)

### 趋势 2：分离式推理 (Disaggregation)
Splitwise (2023) → Mooncake (2024) → Arrow, KVDirect 集群成主流范式

### 趋势 3：Profile-driven vs Analytical 融合
GenZ（解析式）和 LLM-Emu（Profile-driven）互补而非替代

### 趋势 4：行业影响力
Mooncake 已部署于字节跳动豆包，Vidur 被多个团队采用，LLMServingSim 2.0 达 0.95% 误差

## 五、推荐工具链

| 用户角色 | 推荐组合 |
|---------|---------|
| **架构师** | GenZ (快速 DSE) → LLMServingSim 2.0 (高精度验证) |
| **系统工程师** | Vidur (配置搜索) → APEX (并行优化) → Mooncake (部署参考) |
| **AI 工程师** | GenZ (模型-硬件匹配) → LLM-Emu (无 GPU SLO 验证) |
| **学术研究者** | GenZ + LLMServingSim + LLM-Emu + Vidur (方法对比基线) |

## 论文速查表

| # | 论文 | arXiv | 年份 | 类型 | 精度 | GitHub |
|---|------|-------|------|------|------|--------|
| 1 | **GenZ** | 2406.01698 | 2024.06 | 解析建模 | 5.82% | ⭐112 |
| 2 | **LLM-Emu** | 2605.00616 | 2026.05 | 在线仿真 | <5% | 新 |
| 3 | **Vidur** | 2405.05465 | 2024.05 | 仿真 | <9% | ⭐597 |
| 4 | **LLMServingSim 1.0** | 2408.05499 | 2024.08 | HW/SW协同 | 14.7% | - |
| 5 | **LLMServingSim 2.0** | 2602.23036 | 2026.02 | HW/SW协同 | **0.95%** | - |
| 6 | **Frontier** | 2508.03148 | 2025.08 | 仿真 | TBD | - |
| 7 | **APEX** | 2411.17651 | 2024.11 | 策略搜索 | TBD | - |
| 8 | **Splitwise** | 2311.18677 | 2023.11 | 系统设计 | N/A | - |
| 9 | **Mooncake** | 2407.00079 | 2024.06 | 系统设计 | N/A | ⭐5285 |
| 10 | **Sarathi-Serve** | 2403.02310 | 2024.03 | 系统设计 | N/A | ⭐496 |
| 11 | **Debunk CUDA** | 2501.00210 | 2024.12 | 硬件评估 | N/A | - |

## 相关笔记

- [[Knowledge/论文分析/推理系统/GenZ 深度技术分析]]
- [[Knowledge/论文分析/推理系统/LLM-Emu 技术分析]]
- [[Knowledge/论文分析/推理系统/Vidur 深度技术分析]]
- [[Knowledge/论文分析/推理系统/LLMServingSim 1.0 技术分析]]
- [[Knowledge/论文分析/推理系统/LLMServingSim 2.0 技术分析]]
- [[Knowledge/论文分析/推理系统/Frontier 技术分析]]
- [[Knowledge/论文分析/推理系统/APEX 技术分析]]
- [[Knowledge/论文分析/推理系统/Splitwise 技术分析]]
- [[Knowledge/论文分析/推理系统/Mooncake 技术分析]]
- [[Knowledge/论文分析/推理系统/Sarathi-Serve 技术分析]]
- [[Knowledge/论文分析/推理系统/Debunk CUDA Myth 技术分析]]
