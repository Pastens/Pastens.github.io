---
tags:
- 论文分析
- kv-cache
- inference-simulation
- hisim
- tair
- alibaba
- open-source
source: https://github.com/alibaba/tair-kvcache
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# Tair KVCache & HiSim：阿里巴巴的 KVCache 管理与推理仿真系统

## 一、项目概览

| 属性 | 内容 |
|------|------|
| **仓库** | https://github.com/alibaba/tair-kvcache |
| **Stars** | ⭐167 |
| **语言** | C++ (Manager) + Python (HiSim) |
| **许可证** | Apache-2.0 |
| **Topics** | kv-cache, kvcache, hisim, simulator, llm |

### 系统组成

Tair KVCache 开源了两大组件：

```
Tair KVCache
├── Tair KVCache Manager     ← 全局 KVCache 元数据管理服务
│   ├── 缓存逻辑（前缀/滑动窗口/KV匹配）
│   ├── 存储管理（兼容 Mooncake/HF3FS/NFS）
│   ├── 容量管理 + 逐出策略（LRU/RandomLRU/LeafAwareLRU/TTL）
│   └── Optimizer            ← KV Cache 行为仿真器
└── Tair KVCache HiSim       ← LLM 推理全流程仿真系统
    ├── TimePredictor
    │   ├── AIConfigurator（算子级 profiling 预测）
    │   └── ScheduleReplay（trace 回放）
    └── SGLang 劫持接口（替换前向传播，保留调度+HTTP栈）
```

---

## 二、回答你的问题：HiSim 是 KVCache 行为的性能仿真吗？

**既是也不是**。准确地说，Tair KVCache 包含**两个**仿真器，面向不同层次：

### 2.1 Optimizer — 这是真正的 KV Cache 行为仿真器

位于 `kv_cache_manager/optimizer/`，功能：

| 功能 | 说明 |
|------|------|
| **Trace 回放** | 重放 KVCache 访问 trace，评估不同逐出策略 |
| **逐出策略** | LRU / RandomLRU / LeafAwareLRU / TTL |
| **多级存储** | GPU HBM (L1) → CPU DRAM (L2) → SSD (L3) |
| **输出指标** | Cache Hit Rate, 容量消耗, 吞吐量 |
| **用途** | 部署前评估策略效果，指导容量配置 |

这是一个**KV Cache 管理的行为仿真器**，回答"在这个 trace、这个容量、这个逐出策略下，命中率是多少"。

### 2.2 HiSim — 这是端到端 LLM 推理仿真系统

位于 `hisim/`，功能远比 KV Cache 行为仿真广泛：

| 功能 | 说明 |
|------|------|
| **架构** | 劫持 SGLang 框架，替换 GPU 前向传播 |
| **预测** | TTFT、TPOT、吞吐量、E2E 延迟 |
| **方法** | AIConfigurator（算子级 profiling + 插值）|
| **精度** | **<5% MAPE**（Qwen3-8B/32B on H20）|
| **三种场景** | no_cache / L1 (GPU HBM hit) / L2 (GPU+DRAM hit) |
| **无需 GPU** | 纯 CPU 运行 |

HiSim **包含了** KV Cache 行为的影响（通过区分 no_cache/L1/L2 场景来建模缓存命中/未命中的性能差异），但它是一个**全流程推理仿真器**，而非单纯的 KV Cache 行为仿真。

---

## 三、与现有系统对比

| 维度 | HiSim | LLM-Emu | Vidur | GenZ |
|------|-------|---------|-------|------|
| **方法** | 框架劫持 + 算子预测 | Profile-driven online | 算子级仿真 | 解析式 RoofLine |
| **KV Cache 建模** | ✅ L1/L2/no_cache 场景 | ✅ implicit through profile | ✅ 显式 | ❌ 无显式 |
| **精度** | <5% | <5% | <9% | 5.82% |
| **开源** | ✅ Apache 2.0 | ✅ | ✅ MIT | ✅ |
| **需 GPU** | ❌ | ❌ | ❌ | ❌ |
| **支持的引擎** | SGLang | vLLM | 自研仿真器 | N/A（解析） |
| **支持模型** | Qwen3 系列 | 通用（需 profile） | 通用（需 profile） | 通用（配置） |

---

## 四、技术方法详解

### 4.1 Optimizer — KV Cache 策略仿真

Optimizer 是专门分析 KVCache 逐出策略的行为仿真器：

```
输入: Trace 文件 (JSONL) → 缓存配置 (容量/策略/层级)
                                     ↓
           逐出策略引擎 (LRU/RandomLRU/LeafAwareLRU/TTL)
                                     ↓
           输出: Hit Rate, 容量消耗曲线, 逐出统计
```

- **逐出策略**: lru / random_lru / leaf_aware_lru / ttl
- **逐出模式**: GROUP_ROUGH / INSTANCE_ROUGH / INSTANCE_PRECISE
- **TTL 语义**: 支持滑动窗口（读续命）和固定窗口（读不续命）

### 4.2 HiSim — 推理全流程仿真

HiSim 的技术路线与 LLM-Emu 高度相似——**劫持真实推理框架的执行流**：

```
真实请求 Trace → SGLang 调度器 (真实代码) → 前向传播 (被劫持) → 延迟预测 → 指标输出
                                                    ↓
                                        AIConfigurator 算子数据库
                                        (离线 profiling 各算子延迟)
```

**TimePredictor** 使用 **AIConfigurator**（https://github.com/ai-dynamo/aiconfigurator）进行算子级时间预测，支持任意 batch size 和序列长度的延迟插值。

**配置示例**：
```json
{
  "platform": {
    "accelerator": {"name": "H20"},
    "disk_read_bandwidth_gb": 4,
    "disk_write_bandwidth_gb": 4,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64
  },
  "predictor": {
    "name": "aiconfigurator",
    "device_name": "h20_sxm",
    "prefill_scale_factor": 1.02040816,
    "decode_scale_factor": 1.01010101
  }
}
```

注意配置中有 **disk_bandwidth** 和 **memory_bandwidth** —— 这些参数直接影响 KV Cache 在 L2 (DRAM) 和 L3 (SSD) 层级间的传输时间建模。

---

## 五、个人评价

HiSim + Optimizer 的组合恰好填补了我们之前讨论的**结构性的空白**：

| 空白 | 填补者 |
|------|--------|
| KV Cache 策略行为仿真 | ✅ Optimizer（逐出策略/命中率分析） |
| 推理全流程系统仿真（含 KV Cache 影响） | ✅ HiSim（<5% 精度） |
| 统一缓存仿真框架 | ⚠️ 仍在发展（当前仅支持 SGLang+Qwen3+H20） |

**亮点**：
- Optimizer 支持多种逐出策略的 trace 回放，是纯粹 KV Cache 行为仿真的最好例子
- HiSim <5% 的精度极具竞争力，且完全开源
- 双组件设计（策略仿真 + 系统仿真）有清晰的层次分离

**局限**：
- 模型/硬件支持范围窄（目前仅 Qwen3 + H20 + SGLang）
- HiSim 依赖 AIConfigurator 的算子数据库，扩展需要专业 profiling
- Optimizer 目前是单机模拟，未模拟分布式 Cache 共享的场景
- 业界最具对比性的工具是 **LLM-Emu**（支持 vLLM，泛化性更好）和 **Mooncake**（分布式 KV Cache 共享）

## 相关链接
- [[Knowledge/论文分析/缓存系统/缓存系统性能建模洞察分析]]
- [[Knowledge/论文分析/推理系统/LLM-Emu 技术分析]] — 同为框架劫持路线的推理仿真器
- [[Knowledge/论文分析/推理系统/Mooncake 技术分析]] — 分布式 KV Cache 共享系统
