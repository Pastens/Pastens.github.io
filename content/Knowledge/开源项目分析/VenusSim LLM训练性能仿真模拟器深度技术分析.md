---
tags:
  - 开源项目分析
  - 训练系统
  - 性能建模仿真
source: https://github.com/Infrawaves/VenusSim
authors: Infrawaves
created: 2026-07-20
updated: 2026-07-20
---

# VenusSim LLM 训练性能仿真模拟器深度技术分析

## 一、概览

| 属性 | 内容 |
|------|------|
| **项目名称** | VenusSim — LLM Training Simulator |
| **开发者** | Infrawaves 团队 |
| **代码仓库** | https://github.com/Infrawaves/VenusSim |
| **核心定位** | 专为 LLM 训练设计的轻量、精确的离散事件仿真模拟器 |
| **底层框架** | 基于 [Astra-Sim](https://astra-sim.github.io/)（MSR 的分布式训练仿真框架）构建 |
| **输入格式** | JSON 配置文件 + Megatron-LM 风格并行策略参数 |
| **输出格式** | Perfetto Trace（.json）性能分析文件 + CLI 性能报告 |

### 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                     VenusSim (Python CLI)                     │
│                     llm_sim.py (入口)                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │  Trace Generator    │──▶│  LLM Simulator (Astra-Sim)    │  │
│  │  (generator/)       │   │  (llm-simulator/)             │  │
│  │  · 模拟 Megatron    │   │  · 离散事件仿真引擎            │  │
│  │    训练流程         │   │  · 网络拓扑仿真                │  │
│  │  · 生成 Chakra/     │   │  · 系统调度仿真                │  │
│  │    简化 Trace       │   │  · 通信集体量仿真              │  │
│  └─────────┬───────────┘   └───────────────┬──────────────┘  │
│            │                                │                 │
│            ▼                                ▼                 │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │  Chakra Converter   │   │  Timeline Visualizer         │  │
│  │  (chakra_convert)   │   │  (chakra_linker_visualizer)  │  │
│  │  · JSON → .et       │   │  · Perfetto trace 输出      │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 核心工作流

```
配置文件 (JSON) 
    │
    ▼
1. Trace 生成 (et_generator.py)
    ├── 初始化并行状态 (PP/TP/DP/CP)
    ├── 模拟 train_step 完整流程
    │   ├── zero_grad → forward_backward → optimizer.step
    │   ├── P2P 通信 (send/recv) 插桩
    │   ├── 分布式 Optimizer (all-gather + reduce-scatter)
    │   └── All-reduce / All-gather 通信集体
    ├── 输出简化 Trace (.et) 或 Chakra Trace (.json → .et)
    └── 模板化 Trace 生成 (通过 p2p_communication 模板化)
    
    ▼
2. Astra-Sim 仿真 (C++ 二进制)
    ├── 读取 Trace 作为 Workload 输入
    ├── 网络拓扑仿真 (FullyConnected / Ring / Switch)
    ├── 系统调度 (LIFO, Rooffline 模型)
    └── 输出实际执行时间

    ▼
3. 可视化 (Perfetto)
    ├── chakra_timeline_visualizer 转换
    └── Perfetto UI (ui.perfetto.dev) 查看

    ▼
4. 性能统计 (get_perf)
    ├── forward-compute / backward-compute 时长
    ├── send / recv / all-grads-sync / params-all-gather 时长
    └── TFLOP/s/GPU + per-iteration 时间
```

---

## 二、Trace Generator 深度分析

### 2.1 整体设计思路

VenusSim 的 Trace Generator 是**整个项目的灵魂**。它不是简单地读取真实 Profiler 数据，而是**模拟 Megatron-LM 的训练流程来逐节点生成计算/通信 Trace**。这种"从先验知识出发的生成式仿真"思路，使其可以在**没有真实硬件**的情况下预测训练性能。

其核心设计遵循：
1. **Megatron-LM Pipeline Simulation** — 完整模拟 Megatron 的训练数据流，包括 forward step、backward step、P2P 通信、distributed optimizer
2. **FLOPs 驱动计算时间建模** — 根据模型参数（hidden_size, num_layers, seq_length 等）计算每个 layer 的 FLOPs，再通过 Roofline 模型映射为执行时间
3. **通信量精确计算** — 根据并行策略（PP/TP/DP/CP）和张量大小，精确计算每个通信原语的数据量

### 2.2 代码结构

```
trace-generator/
├── llm_sim.py                          # 主入口编排
├── utils.py                            # 工具函数 (性能统计, 配置解析)
├── generator/
│   ├── generator.py                    # TraceNode/PytorchPlusTrace 数据结构
│   ├── et_generator.py                 # 核心生成逻辑 + FLOPs 计算
│   ├── utils.py                        # form_comp_node, merge_branch_nodes
│   ├── overlap_grad_reduce_helper.py   # Grad Reduce 重叠模拟
│   ├── train_step_hook_helper.py       # Hook 机制
│   └── megatron_core/
│       ├── training.py                 # train() / train_step()
│       ├── schedules.py                # 1D/Interleaved 1F1B Pipeline 调度
│       ├── p2p_communication.py        # P2P send/recv 通信模拟
│       ├── parallel_state.py           # PP/TP/DP/CP 状态管理
│       ├── distrib_optimizer.py        # Distributed Optimizer 模拟
│       ├── optimizer.py                # MixedPrecisionOptimizer 基类
│       └── arguments.py                # 参数解析与验证
├── chakra_convert_standalone/          # Chakra 原生格式转换
└── demo/llm-sim-example/               # Demo 配置
```

### 2.3 关键数据结构：TraceNode

`TraceNode` 是 Trace 的最小单位，支持 6 种节点类型：

| 节点类型 | 枚举值 | 用途 | 关键属性 |
|---------|--------|------|---------|
| COMP_NODE | 3 | 计算节点 | `num_ops` (FLOPs), `tensor_size` |
| COMM_SEND_NODE | 4 | P2P Send | `comm_src`, `comm_dst`, `comm_tag`, tensor 大小 |
| COMM_RECV_NODE | 5 | P2P Recv | 同上 |
| COMM_COLL_NODE | 6 | 通信集体 | `num_elem`, `elem_bytes`, `comm_group` |
| REPLAY_NODE | 7 | 回放节点 | `dur` (ns) — 直接指定时长 |
| CPU_NODE / GPU_NODE | 1/2 | 元数据 | 标记执行设备 |

`PytorchPlusTrace` 封装了整个 Trace，提供：
- 节点 DAG 管理（ctrl_deps 控制依赖）
- 环检测（DFS 检测 cyclic dependencies）
- 悬挂节点清理
- 多格式输出（JSON / 简化文本格式）

### 2.4 简化 Trace 格式

VenusSim 支持三种 Trace 格式：

| 格式 | 标志 | 特点 | 内存占用 |
|------|------|------|---------|
| Chakra (protobuf) | `use_chakra=1` | 完整 Chakra 标准格式 | 高 |
| 简化 Trace | `use_simplified_trace=1` | 自研文本格式，去除 JSON 冗余 | 中 |
| 简化模板 Trace | `use_simplified_template_trace=1` | 基于模板的简化 Trace (PP rank 间共享模板) | 低 |

**简化 Trace 文本格式示例：**
```
<节点数>
<id>
<name>
<ctrl_deps_count> <ctrl_dep_1> <ctrl_dep_2> ...
<simplified_attr_1> <simplified_attr_2> ...
```

简化属性根据节点类型不同：
- **COMP_NODE**: `[num_ops, tensor_size]`
- **COMM_COLL_NODE**: `[num_elem, elem_bytes, comm_group_name]`
- **COMM_SEND/RECV_NODE**: `[num_elem, elem_bytes, src, dst, tag]`
- **REPLAY_NODE**: `[dur_ns]`

### 2.5 模板化 Trace (Template Trace)

VenusSim 的核心创新点之一：**同一个 PP group 内，不同 rank 的 Trace 只在通信 src/dst 上有差异**，计算 Trace 结构完全一致。利用这一特性：

1. **首先生成一个模板** — 为 primary_pp_group 中的所有 rank 生成一次 Trace（包含完整计算+通信节点）
2. **其他 rank 只记录映射关系** — 记录每个 rank 对应的模板节点中 src/dst 的替换规则
3. **Astra-Sim 在运行时扩展模板** — 根据 `trace-template-mapping.txt` 将模板展开

这在大规模仿真时可以**显著减少 Trace 文件大小**（PP group × DP 规模越大，节省越多）。

### 2.6 计算时间建模

#### A. FLOPs 计算 (`calculate_transformer_flops`)

VenusSim 精确计算每个 training step 的 FLOPs：

```
Total FLOPs = Expansion_Factor × Global_Batch_Size × Seq_Length × Num_Layers × Hidden_Size² × (
    Attention + MLP + Shared_Expert + Softmax + Embedding
)
```

其中 Attention/MLP/Softmax/Embedding 各自的分量根据模型架构精确计算，支持：
- Multi-Query Attention (MQA) / Grouped-Query Attention (GQA) 的比例修正
- SwiGLU 激活函数的 3/2 参数因子
- MoE 路由后的 expert FLOPs
- Shared Expert + Gated Linear Unit

#### B. 执行时间映射

分为两种模式：

**模式一：FLOPs 驱动（默认）**
```
单节点 FLOPs = (总 FLOPs / num_layers / total_microbatches)
执行时间 = FLOPs / (Peak_Perf × MFU)
```
其中 `Peak_Perf` 和 `MFU` 由 `system.json` 中的 `peak-perf` 和 `mfu` 配置。

**模式二：Snapshot 驱动**
当 `use_step_snapshot=1` 时，直接使用用户提供的实测 PP stage 计算时间：
```
pp-0: X ms
pp-mid: Y ms  
pp-end: Z ms
(前向:后向 = 1:2)
```

#### C. 后向计算量

后向计算量 = 前向计算量 × 2（梯度计算的双倍 FLOPs 经验假设）

### 2.7 通信建模

#### P2P 通信 (Pipeline Parallelism)

VenusSim 精确模拟 Megatron 的 P2P 通信——包括 send/recv 的匹配（通过 `comm_tag` 验证）。对于 interleaved 1F1B schedule，P2P 通信发生在以下场景：

```
P2P Send:
  [pp_rank] → send_tensor_to_next(pipeline_model_parallel_next_rank)
  [pp_rank] → send_tensor_to_prev(pipeline_model_parallel_prev_rank)

P2P Recv:
  [pp_rank] → recv_tensor_from_prev(pipeline_model_parallel_prev_rank)
  [pp_rank] → recv_tensor_from_next(pipeline_model_parallel_next_rank)
```

每个 send/recv 节点记录：
- `comm_src`, `comm_dst` — 通信源/目标 rank
- `comm_tag` — 通信标签（用于 send/recv 配对验证）
- `num_elem` × `elem_bytes` — 通信数据量

#### 通信集体 (TP/DP)

| 通信类型 | 对应场景 | 通信组 |
|---------|---------|--------|
| all-reduce | TP 内的 forward 计算后、DP 的 grad sync | tp_group / dp_group |
| all-gather | DP 的 param all-gather | dp_group |
| reduce-scatter | DP 的 grad reduce-scatter | dp_group |
| all-to-all | 特定并行策略 | comm_group |

### 2.8 Pipeline Schedule 模拟

`schedules.py` 完整模拟 Megatron 的两种 Pipeline Schedule：

| Schedule | 条件 | 说明 |
|----------|------|------|
| **Without Interleaving** | VPP size = None | 标准 1F1B，每个 rank 一个 model chunk |
| **With Interleaving** | VPP size > 1 | 多个 model chunk 交错执行，提高 pipeline bubble 利用率 |

模拟的调度逻辑：
1. **Forward Pass** — 从数据加载开始，逐 microbatch 执行 forward，P2P 通信在 stage 间传递激活值
2. **Backward Pass** — 1F1B 调度，backward 与 forward 交错执行
3. **Optimizer Step** — 参数更新 + zero_grad

### 2.9 Distributed Optimizer 模拟

`distrib_optimizer.py`（1409 行）完整模拟 Megatron 的 Distributed Optimizer：

**核心流程：**
```
0. zero_grad() — 清空梯度缓冲区
1. Forward-Backward — 生成梯度
2. Reduce-Scatter — 将梯度按 DP rank 分散
3. All-Gather — 将更新后的参数收集回所有 rank
```

**关键优化特性模拟：**
- `overlap-grad-reduce` — 在 backward 过程中重叠 gradient reduce-scatter
- `overlap-param-gather` — 在 optimizer step 后重叠参数 all-gather
- `delay-grad-reduce` — 延迟梯度 reduce，配合 overlap
- `delay-param-gather` — 延迟参数 gather，配合 overlap

---

## 三、Astra-Sim 集成的仿真引擎

VenusSim 的底层仿真引擎是 **Astra-Sim**（Microsoft Research 开源的分布式训练仿真框架）。llm-simulator/ 目录包含了完整 Astra-Sim 代码以及 VenusSim 的增强。

### 3.1 Astra-Sim 组件

| 组件 | 路径 | 说明 |
|------|------|------|
| **Workload** | `astra-sim/workload/` | Trace 驱动的 workload 模拟器 |
| **System** | `astra-sim/system/` | 系统调度 + 拓扑 |
| **Network** | `astra-sim/network_frontend/` | 网络仿真前端（Analytical / NS3） |
| **Collective** | `astra-sim/system/collective/` | Ring, DoubleBinaryTree, HalvingDoubling, AllToAll |

### 3.2 系统配置 (system.json)

关键系统参数：

| 参数 | 含义 | 示例 |
|------|------|------|
| `scheduling-policy` | 调度策略 | LIFO |
| `endpoint-delay` | 端点延迟 (ns) | 10 |
| `all-reduce-implementation` | 各维度 AllReduce 算法 | ["ring", "ring"] |
| `roofline-enabled` | 是否启用 Roofline 模型 | 1 |
| `peak-perf` | GPU 峰值性能 (TFLOP/s) | 1000 |
| `mfu` | 模型利用率 (Model FLOPs Utilization) | 1 (100%) |
| `local-mem-bw` | 本地内存带宽 (GB/s) | 50 |
| `simplified-trace` | 简化 Trace 模式 | 1 |

### 3.3 网络配置 (network.yml)

VenusSim 支持多维网络拓扑配置：

```yaml
topology: [FullyConnected, Ring]     # Dim0: FC, Dim1: Ring
npus_count: [8, 2]                    # Dim0: 8 NPU, Dim1: 2 NPU → 共16 NPU
bandwidth: [450.0, 50.0]             # GB/s per direction
latency: [1000.0, 6000.0]            # ns
```

这一设计使 VenusSim 可以灵活模拟各种实际网络拓扑（如 DGX 内部的 NVLink + 跨节点 InfiniBand 的二维配置）。

---

## 四、Demo 示例分析

### 4.1 配置解读

Demo 配置仿真一个 **16 GPU (2 节点 × 8 GPU)** 的训练场景：

```
模型: Llama 2 级别 (hidden=4096, FFN=16384, 32 layers, 16 heads, MQA=4)
并行: PP=8, TP=1, DP=2 (world=16)
VPP: 开启，每 stage 2 层 → virtual pipeline stages = 2
微批次: micro-batch=2, global-batch=64
优化: overlap-grad-reduce, overlap-param-gather, distributed-optimizer
网络: Dim0 FullyConnected 450GB/s, Dim1 Ring 50GB/s
```

### 4.2 运行流程

```
1. Trace 生成 → demo.trace (简化模板 Trace)
2. Astra-Sim 仿真 → trace_output.log
3. 可视化 → trace_output.json (Perfetto)
4. 性能统计:
   ── throughput per GPU (TFLOP/s/GPU): XX.XX
   ── elapsed time per iteration (ms): XX.XX
   ── forward-compute: (min, max)
   ── backward-compute: (min, max)
   ── send: (min, max)
   ── recv: (min, max)
   ── all-grads-sync: (min, max)
   ── params-all-gather: (min, max)
```

---

## 五、亮点与局限

### 亮点

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **生成式仿真管线** | 无需真实硬件/Profiler，从模型参数和并行配置直接生成 Trace |
| 2 | **Megatron-LM 忠实模拟** | 代码结构参照 Megatron 源码，P2P 通信、Pipeline Schedule、Distributed Optimizer 均有精确建模 |
| 3 | **模板化 Trace 压缩** | PP group 内共享 Trace 模板，显著减少大规模仿真的 Trace 文件大小 |
| 4 | **多层次 Trace 格式** | Chakra / 简化文本 / 模板 Trace 三种格式，灵活权衡精度与内存 |
| 5 | **灵活的配置系统** | 支持几乎所有 Megatron 训练优化特性（VPP, overlap, delay, distributed opt 等） |
| 6 | **多维网络拓扑** | 支持 FullyConnected / Ring / Switch 多维混合拓扑 |
| 7 | **FLOPs 精确计算** | 支持 GQA/MQA、SwiGLU、MoE、Shared Expert 等现代 Transformer 架构 |
| 8 | **Snapshot 模式** | 允许注入实测时间作为计算时长，混合仿真以提高精度 |

### 局限

| # | 局限 | 说明 |
|---|------|------|
| 1 | **Rooffline 精度依赖** | 计算时间通过 FLOPs / (Peak_Perf × MFU) 计算，MFU 设置直接影响精度 |
| 2 | **无真实内存建模** | 不对实际显存分配进行仿真，memory.json 保持默认配置无需修改 |
| 3 | **无计算-通信重叠的细粒度建模** | 重叠效果通过配置开关（overlap-grad-reduce 等）控制，而非真实合宿场景 |
| 4 | **仅支持 Pipeline Parallel (>1)** | 不支持无 Pipeline 的 no_pipelining 模式 |
| 5 | **依赖 Astra-Sim C++ 编译** | 仿真引擎需要编译，容器构建过程较复杂 |
| 6 | **无 MoE 完整建模** | MoE FLOPs 计算框架已准备，但通信建模（all-to-all expert routing）不完整 |
| 7 | **通信 Group 自动写入** | comm-group.json 由 Trace Generator 自动生成，用户无法手工指定复杂的通信组拓扑 |
| 8 | **无推理场景支持** | 仅覆盖训练场景，不支持推理延迟仿真 |

---

## 六、个人评价

VenusSim 作为 Infrawaves 团队的开源 LLM 训练仿真器，在技术路线上采取了务实的**生成式**仿真策略——通过模拟 Megatron 训练流程生成 Trace，而不是从真实 Profiler 采集数据。这使其在灵活性上远胜于基于真实 Trace 回放的仿真器（如 SimuMax），用户只需修改 JSON 配置即可探索任意模型大小、任意并行策略的训练性能。

与同类工具对比：

| 项目 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **VenusSim** | 生成式 Trace + Astra-Sim | 灵活、支持所有 Megatron 特性、模板化 Trace | 计算时间依赖 MFU 估计 |
| **SimuMax** | 静态分析模型 | 精度高（算子级建模） | 可配置性低 |
| **Vidur** | 参数化仿真 | 轻量、快速 | 模型精度有限 |
| **msOpProf** | 实测回放 | 最真实 | 需要真实硬件 |

VenusSim 最适合的场景是：**在没有真实硬件的设计探索阶段，快速评估不同并行策略和网络拓扑对 LLM 训练性能的影响**。其详细的 Megatron 优化特性模拟（overlap, VPP, distributed optimizer 等）使其在工程价值上显著高于简单的线性扩展模型。

---

## 参考文献

- [Astra-Sim](https://astra-sim.github.io/) — 底层离散事件仿真引擎
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — Pipeline 调度与分布式优化器参考实现
- [Chakra](https://github.com/mlcommons/chakra) — Trace 格式标准
- [Perfetto](https://ui.perfetto.dev/) — 可视化工具
