---
tags:
  - 开源项目分析
  - 训练系统
  - 性能建模
  - 仿真
source: https://github.com/MooreThreads/SimuMax
license: Apache 2.0
created: 2026-06-03
rating: ⭐⭐⭐⭐⭐
---

# SimuMax: 大模型分布式训练静态分析模型 深度技术分析

## 一、概览

| 属性 | 内容 |
|------|------|
| **项目名称** | SimuMax |
| **开发者** | Moore Threads Technology Co., Ltd（摩尔线程） |
| **代码** | https://github.com/MooreThreads/SimuMax |
| **许可** | Apache 2.0 |
| **语言** | Python (25,728 lines) |
| **最新版本** | v1.2 (2026-05-11) |
| **核心定位** | LLM 分布式训练吞吐量、峰值内存、Pipeline 行为的**静态分析模型**（无需启动真实训练） |

### 核心贡献

1. **三输入驱动建模** — 以 `system`（硬件能力）、`strategy`（并行策略）、`model`（模型架构）三个 JSON 配置文件驱动，将训练性能预测转化为配置组合问题
2. **支持 Dense + MoE 全系并行策略** — 涵盖 TP / PP / EP / CP / SP / ZeRO-1 / recompute / MLA / VPP
3. **可搜索的 batch 与策略空间** — 内置 `strategy_searcher` 可在给定硬件和模型下自动搜索可行的 micro_batch 和并行策略组合
4. **Simulator 仿真器追踪** — 除 perf 估计外，还能生成 pipeline schedule trace、memory snapshot，支持真实运行对比验证
5. **B200 / A100 公开基准验证** — 提供经过真实环境对标校准的 public benchmark，包括 CP A2A 长序列场景
6. **高效的 perf-to-real 对比** — 通过 `accurate_efficient_factor`（shape 级别的算子效率系数）实现高精度 timing 拟合

---

## 二、技术架构详解

### 2.1 整体设计：三输入驱动

SimuMax 的核心设计哲学是**将训练性能预测抽象为三个正交输入配置组合的问题**：

```
                   ┌─────────────┐
                   │  model.json │ 架构：hidden_size, layer_num, head_num, ...
                   └──────┬──────┘
                          │
  ┌─────────────┐   ┌────▼─────┐   ┌──────────────┐
  │system.json  │──▶│  PerfLLM │◀──│ strategy.json│
  │硬件能力/效率 │   │ (核心引擎)│   │ 并行策略/超参│
  └──────┬──────┘   └────┬─────┘   └──────┬───────┘
         │               │                │
         ▼               ▼                ▼
    ┌───────────────────────────────────────────┐
    │           run_estimate()                  │
    │   ├─ cost_result: 吞吐/MFU/step time      │
    │   ├─ mem_result: 峰值内存/激活/参数        │
    │   └─ simulate(): trace + memory snapshot   │
    └───────────────────────────────────────────┘
```

**配置解耦**的设计使得：
- 更换硬件（A100 → B200）只需换 `system.json`
- 更换模型（LLaMA 3 → DeepSeek V2）只需换 `model.json`
- 探索并行策略只需换 `strategy.json`

### 2.2 目录结构

```
SimuMax/
├── simumax/                   # 核心库 (~14K 行)
│   ├── core/
│   │   ├── config.py          # 配置系统：ModelConfig / StrategyConfig / SystemConfig
│   │   ├── perf_llm.py        # 性能分析引擎：PerfLLM (3696行，核心)
│   │   ├── base_struct.py     # 数据结构和仿真基类 (2764行)
│   │   ├── simu_runner.py     # 仿真编排引擎
│   │   ├── simu_memory.py     # 内存跟踪器
│   │   ├── graph.py           # ONNX 计算图构建
│   │   ├── generate_tracing.py # trace 日志解析
│   │   └── trace_export.py    # trace 导出
│   │   └── transformer/       # 模型模块定义
│   │       ├── dense_module.py    # Dense 模块 (2988行)
│   │       ├── moe_module.py      # MoE 模块 (1566行)
│   │       ├── language_model.py  # LLM 模型组装
│   │       ├── pipeline_schedule.py # Pipeline 调度模拟 (959行)
│   │       ├── simu_ops.py        # 张量操作
│   │       └── function.py        # 函数封装
│   ├── tuning/
│   │   └── strategy_searcher.py # 策略搜索
│   └── testing/
│       └── base_test_tool.py
├── configs/                   # 配置文件仓库
│   ├── models/                # 25+ 模型配置（llama3, deepseekv2, mixtral, qwen3...）
│   ├── system/                # 硬件配置（a100_pcie, b200_bf16_ceperm）
│   └── strategy/              # 策略配置（tp2_pp1_dp4, ep4_pp2_dp4...）
├── simu_tools/                # 辅助工具
│   ├── efficency_test/        # 算子效率测量工具链
│   └── megatron_scripts/      # Megatron 基准测试脚本
├── app/                       # Streamlit 交互式应用
├── examples/                  # 使用示例
└── docs/                      # 文档
```

### 2.3 核心模块详解

#### 2.3.1 Config 系统（config.py, 1227 行）

配置系统以 `Config` 基类为根，通过 `@dataclass` 注解的三个子类实现：

- **`ModelConfig`** — 描述模型架构：`model_type`（dense/moe）、`hidden_size`、`head_num`、`kv_head_num`、`intermediate_size`、`layer_num`、`vocab_size`、`use_swiglu`，以及 MoE 专用字段（`expert_num`、`topk`、`moe_ffn_hidden_size`）、MLA 专用字段（`v_head_dim`、`qk_head_dim`、`q_lora_rank`、`kv_lora_rank`）
- **`StrategyConfig`** — 并行策略与运行时超参：TP/PP/EP/CP/SP 大小、`seq_len`、`micro_batch_size`/`micro_batch_num`、`zero_state`、`recompute_granularity`、`interleaving_size`（VPP）、`enable_sequence_parallel` 等
- **`SystemConfig`** — 硬件能力：每节点 GPU 数、显存大小、各算子 `tflops` + `efficient_factor` + shape 级 `accurate_efficient_factor`、网络拓扑与带宽/延迟

关键设计细节：

- **`accurate_efficient_factor`** — 这是一个 shape 维度的效率校准表，以 `"b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16"` 这样的 string key 存储实测效率值。SIMT 架构上 GEMM 效率与输入形状高度相关，这个设计让 SimuMax 可以在关键热路径上使用实测数据，而非统一的理论峰值 × 固定系数
- **`AttentionRecomputeConfig` / `MLPRecomputeConfig`** — 细粒度的 recompute 控制，支持 attention 和 MLP 内部各个子步骤（layernorm、QKV、rope、core_attn、output 等）的独立 recompute 开关

#### 2.3.2 性能分析引擎（perf_llm.py, 3696 行）

`PerfLLM` 是 SimuMax 的核心分析引擎，提供三层能力：

**1. `run_estimate()`** — 构建模型训练计算图
- 根据 `model.json` 读取 `model_type` 决定 Dense/MoE 路径
- 按 `strategy.json` 中的 TP/PP/EP 参数切分计算图
- 通过 `strategy_str` 缓存机制（`_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS`）加速重复配置
- 输出 Chunk 级别的 `CachedChunkProfile`

**2. `analysis_cost()`** — 计算各模块计算/通信时间
- 逐模块计算 fwd/bwd/optimizer 的 FLOPs
- 通过 `SystemConfig.compute_net_op_time()` 计算 all_reduce / all_gather / reduce_scatter / all2all / p2p 通信开销
- 支持 PP bubble 计算（`_compute_bubble_time()`）
- 支持 straggler 模型（`STRAGGLER_BASE_FACTOR = 0.09`）
- 输出：step_time、MFU、各 stage 时间、通信占比等

**3. `analysis_mem()`** — 分析内存峰值
- 逐模块统计参数内存、优化器状态、激活值
- 支持选择性 recompute 对激活内存的影响建模
- 输出各 stage 峰值内存

**搜索缓存机制**：`_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS` 定义了哪些 strategy 字段变化需要重建缓存。核心思想是将"每个 unit 的本地计算成本"和"PP 排布"解耦——只要 unit 内部的 batch 大小、TP/EP 等不变，即使 PP 排布变化也可以复用缓存。这是一个精巧的优化。

#### 2.3.3 模型模块定义（transformer/）

模型定义采用**层次化的 MetaModule 架构**：

```
MetaModule (基类)
  ├── LLMModel (language_model.py)   — 完整 LLM（含 embedding + layers + loss）
  │   └── LLMBlock                   — 单层 Transformer Block
  │       ├── Attention / MLAAttention  (dense_module.py)
  │       │   └── QKV Projection, RoPE, Core Attention, Output Projection
  │       └── MLP / Swiglu / Gelu      (dense_module.py)
  ├── ExpertMLP (moe_module.py)      — MoE 专家层（Grouped MLP）
  │   └── Router / Permutation / Gate
  └── Embedding / LayerNorm / ParallelCE  (dense_module.py)
```

每个 `MetaModule` 通过 `prefill()` 方法构建其内部的执行序列（`FwdQue`），将计算和通信操作排列为可执行的操作队列。这与实际训练框架的 `forward()` / `backward()` 调用方式相似，但 SimuMax 将其简化为时间成本模型。

**Dense Module**（dense_module.py, 2988 行）：
- `LinearCol` / `LinearRow` — 列/行并行线性层，支持 TP 切分
- `Attention` / `MLAAttention` — 标准 Attention 和 MLA（DeepSeek V2 的 Multi-head Latent Attention）
- `MLP` / `Swiglu` / `Gelu` — 前馈网络及激活函数
- `ParallelCE` — 并行交叉熵损失函数
- `Embedding` — 并行 Embedding（支持 SP）

**MoE Module**（moe_module.py, 1566 行）：
- `ExpertMLP` — 专家网络，实现 grouped GEMM
- `Router` — 路由层，含 gate 线性层、topk 选择
- `all2all` — MoE 的 token 分发/收集通信

#### 2.3.4 仿真引擎（simu_runner.py + base_struct.py）

SimuMax 的仿真器是**轻量级的事件驱动离散仿真**：

- **`SimuSystem`** — 仿真系统，管理多个 `SimuThread` 和全局时钟
- **`SimuThread`** — 每个 PP stage 对应一个仿真线程
- **`SimuContext`** — 仿真上下文，含 `BarrierBackend`（同步 barrier）+ `memory_tracker`
- **`FwdQue`** — 操作队列，每个操作（`AtomModel` / `all_reduce` / `send_next` 等）包装为 `step` 方法，以 delta time 推进仿真时钟
- **`PpSchedule`**（pipeline_schedule.py, 959 行）— Pipeline 调度器，实现 1F1B / interleaving / sync-VPP 等调度策略
- **`OptimizerSimulator`** — 优化器阶段的通信（ZeRO-1 all_gather / reduce_scatter）和计算建模

仿真的核心机制是**barrier + 时间推进**：
1. 每个 rank 的操作以 `FwdQue` 形式排队
2. 仿真线程按时间片推进，遇到依赖（如 PP 通信的 recv）时在 barrier 上等待
3. 全局时钟由所有线程中最早可用的事件决定
4. 最终输出 `tracing_logs.json`（Chrome Trace Format）

#### 2.3.5 计算图系统（graph.py）

SimuMax 内部构建了一个**简化 ONNX 格式的计算图**（`SimuONNXGraphBuilder`）：

- `Node` — 操作节点，包含操作类型（MatMul、Add、Reshape 等）、recompute 标记、输入输出张量名
- `Graph` — 完整计算图，支持前向遍历、recompute 节点标记
- 支持 Graphviz 可视化（`visualize_with_graphviz`）

这个图主要用于 **recompute variance node 标记**（决定 recompute 边界在哪）和调试可视化，不是 SimuMax 的核心执行路径。

### 2.4 配置文件详解

#### 2.4.1 model.json（以 llama3-8b 为例）

```json
{
    "model_type": "dense",
    "model_name": "llama3_8b",
    "hidden_size": 4096,
    "head_num": 32,
    "kv_head_num": 8,
    "head_size": 128,
    "intermediate_size": 14336,
    "layer_num": 32,
    "vocab_size": 128257,
    "use_swiglu": true
}
```

极简的模型定义，仅需 `~10` 个字段即可描述一个模型。支持的特殊字段：
- `attention_type: "mla"` — 启用 DeepSeek 的 MLA
- `padded_vocab_size` — 对齐 vocab_size
- MoE 特定字段：`expert_num`、`topk`、`moe_ffn_hidden_size`、`dense_layers`、`moe_shared_expert_intermediate_size`
- 预训练模型配置（configs/models/: 25+ 个）：llama2/3、mixtral、deepseekv2/v3、aquila2、qwen3、kimi、ling 等

#### 2.4.2 strategy.json

```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "micro_batch_num": 8,
    "dtype": "bf16",
    "world_size": 8,
    "tp_size": 2,
    "pp_size": 1,
    "ep_size": 1,
    "enable_sequence_parallel": true,
    "zero_state": 1,
    "enable_recompute": true,
    "recompute_granularity": "selective_recompute",
    "mem_factor": 0.94,
    "mlp_recompute": true,
    "attn_recompute": true
}
```

关键字段：
- `megatron_recompute` / `megatron_recompute_modules` — Megatron-LM 0.14 风格的 selective recompute 语义
- `interleaving_size` — VPP（virtual pipeline）size
- `moe_dispatcher_policy` — MoE token 分发策略（`all2all` / `all_gather`）

#### 2.4.3 system.json（以 A100 PCIe 为例）

完整配置约 400+ 行，包含：
- `accelerator.op` — 每种算子的 `tflops` + `efficient_factor` + shape 级 `accurate_efficient_factor`
- `networks` — 多级拓扑（`nvlink`、`ib`、`ethernet`），每级有带宽、延迟、ring 拓扑参数
- `accelerator.mem_bw` — 显存带宽（用于 memory-bound 算子估计）

B200 配置更为精细（900 行），记录了对应 80+ 种 GEMM shape 的实测效率。

### 2.5 策略搜索（tuning/strategy_searcher.py）

`StrategySearcher` 提供两种搜索模式：

1. **batch 搜索** — 在固定并行策略下搜索 `micro_batch_size` 和 `micro_batch_num` 的可行组合（不 OOM 且合法）
2. **并行策略搜索** — 搜索 TP/PP 维度组合的最小搜索空间

使用 `sympy.divisors` 计算合法分解，利用 `_SEARCH_CACHE` 避免重复计算。通过 `gmi_error`（NCCL buffer 预留量）作为 OOM 判断的余量调节。

### 2.6 算子效率测量工具链（simu_tools/efficency_test/）

SimuMax 附带了完整的**真实测量工具链**，用于生成 `system.json` 中的 `accurate_efficient_factor`：

| 工具 | 用途 |
|------|------|
| `test_gemm_efficiency.py` | 测量各类 GEMM shape 的实际算力 |
| `test_grouped_gemm_efficiency.py` | 测量 MoE 场景的 Grouped GEMM 效率 |
| `test_fa_efficiency.py` | 测量 Flash Attention 效率 |
| `test_ce_permute_efficiency.py` | 测量 Cross Entropy + Permute 效率 |
| `nccl_fit.py` | NCCL 通信耗时拟合 |
| `reduce_scatter.py` | 独立测量 reduce_scatter 通信 |
| `measure_comm_burst_window_worker.py` | 通信突发窗口测量 |
| `run_one_click_benchmark.py` | 一键基准测试入口 |

### 2.7 Streamlit 交互式应用（app/streamlit_app.py）

提供一个 862 行的 Streamlit 应用，支持：
- 可视化配置 model / strategy / system 参数
- 预设 small / medium / large 配置模板
- 运行 perf 估计，实时显示结果
- 可视化对比多种策略

---

## 三、工作流程与数据流

### 3.1 Perf 分析流程

```
用户调用 PerfLLM.configure()
    │
    ├─▶ 解析 model.json → ModelConfig（model_type, hidden_size, ...）
    ├─▶ 解析 strategy.json → StrategyConfig（tp, pp, ep, ...）
    └─▶ 解析 system.json → SystemConfig（tflops, efficiency, ...）
    │
    ▼
run_estimate()
    │
    ├─▶ 决定 PP 各 stage 的层分配（first/middle/last chunk）
    ├─▶ 构建每个 chunk 的 LLMModel
    │   ├─ dense: Embedding → N×LLMBlock(Attention→MLP) → ParallelCE
    │   └─ moe:   + Router → ExpertMLP + all2all 通信
    ├─▶ 计算各模块 FLOPs（fwd / bwd_grad_w / bwd_grad_act）
    ├─▶ 计算各模块通信量（all_reduce, all_gather, p2p...）
    └─▶ 缓存 CachedChunkProfile
    │
    ▼
analysis_cost() + analysis_mem()
    │
    ├─▶ 聚合各 stage 的 compute time
    ├─▶ 计算通信重叠 / bubble
    ├─▶ 计算 memory 峰值（参数 + 优化器状态 + 激活）
    └─▶ 输出 compute_result.json + mem_result.json
```

### 3.2 Simulator 仿真流程

```
PerfLLM.simulate("save_path")
    │
    ├─▶ run_simulation() (simu_runner.py)
    │   ├─▶ 为每个 PP stage 创建 SimuThread
    │   ├─▶ 构建 PpSchedule（1F1B / interleaving / VPP）
    │   ├─▶ 创建 OptimizerSimulator（ZeRO-1 通信 + optimizer step）
    │   └─▶ SimuSystem.simu() — 事件驱动推进时钟
    │       ├─▶ 每个 thread 执行 FwdQue.step()
    │       ├─▶ 通信操作在 barrier 上同步
    │       └─▶ 写入 log.log
    │
    ├─▶ process_log_file() → tracing_logs.json (Chrome Trace)
    └─▶ [可选] memory_tracker → simu_memory_snapshot.json
```

### 3.3 通信建模

SimuMax 支持多级网络拓扑：

```
┌─────────────────────────────────┐
│          dp_net / edp_net        │ 数据并行 / 专家并行
│         (ethernet / ib)          │
├─────────────────────────────────┤
│          tp_net                  │ 张量并行
│         (nvlink / nvswitch)      │
├─────────────────────────────────┤
│          pp_net                  │ Pipeline 并行
│         (ib / ethernet)          │
└─────────────────────────────────┘
```

每个 `net` 类型通过 `SystemConfig.NetConfig` 定义带宽（`bandwidth_gbs`）和延迟（`latency_us`），并支持自定义 `alpha`/`beta` 通信模型参数。`compute_net_op_time()` 根据通信量、通信操作类型和拓扑参数计算传输时间。

---

## 四、亮点与局限

### 亮点

1. **极低的使用门槛** — 只需三个 JSON 文件即可运行，无需安装 GPU 驱动或分布式框架，5 分钟内可获得一个 8B 模型的训练吞吐估计
2. **高质量的 shape 级效率校准** — `accurate_efficient_factor` 的设计让 SimuMax 可利用实测数据精确建模算子性能，远超简单的理论峰值 × 固定系数法
3. **覆盖场景广泛** — Dense + MoE + MLA + VPP + CP + SP + ZeRO-1 + 多种 recompute 粒度，基本覆盖了当前主流 LLM 训练所需的所有并行和优化技术
4. **Simulator 可调试性** — 能输出 Chrome Trace Format 的 timeline，与实际训练 trace 直接对比，极大方便了 perf 偏差的定位
5. **MoE 全链路建模** — 从 Router → all2all → ExpertMLP（Grouped GEMM）→ all2all 返回到 loss 计算，完整建模了 MoE 训练的通信和计算开销
6. **策略搜索内置** — 自动搜索可行的 batch 和并行配置，省去手动试错
7. **公开基准验证** — 提供 A100 和 B200 的 perf-vs-real 对比数据，增强了模型可信度

### 局限

1. **无计算-通信重叠建模** — 当前版本的 roadmap 明确提到 compute/communication overlap 是规划中的功能，这意味着 SimuMax 可能高估 step time（因为真实框架中计算和通信可以部分重叠）
2. **Pipeline scheduler 支持有限** — 仅 sync-VPP 以 Preview 形式提供，async-VPP 不在 public support surface 内，1F1B 是唯一正式支持的调度策略
3. **无 offloading 支持** — CPU offloading、NVMe offloading 等内存优化策略尚未建模
4. **仅支持 Megatron-LM 风格** — 配置直接映射 Megatron 运行时选项，对 DeepSpeed、FSDP 等框架的支持需要用户自行映射
5. **静态分析本质限制** — 无法建模动态 shape（如变长序列）、运行时优化（如 dynamic batching、adaptive recompute）等实际系统中的复杂行为
6. **无 MPI 并行化** — 仿真器是单进程的，虽然代码中有 `mpi4py` 的导入尝试，但实际并未启用 MPI 分布式仿真
7. **效率校准成本高** — 对未覆盖的 GEMM shape 或新硬件，需要运行完整的 measurement pipeline，时间成本较高
8. **网络拓扑描述较简化** — 采用 3 级网络（tp_net / pp_net / dp_net），对于更复杂的多级拓扑（如 Fat Tree、Dragonfly+）可能需要自定义配置

---

## 五、与其他项目的对比

| 维度 | SimuMax | Calculon | 华为 MindStudio Profiler | nsight compute |
|------|:-------:|:--------:|:-----------------------:|:--------------:|
| 类型 | 静态分析模型 | 静态分析模型 | 性能分析工具 | 性能分析工具 |
| 需要 GPU | ❌ | ❌ | ✅ | ✅ |
| 需要运行训练 | ❌ | ❌ | ✅ | ✅ |
| 配置驱动 | ✅ 三输入 JSON | ✅ | ❌ | ❌ |
| MoE 支持 | ✅ 完整 | 有限 | ✅ | N/A |
| MLA 支持 | ✅ | ❌ | ❌ | N/A |
| Pipeline trace | ✅ Chrome Trace | ❌ | ✅ | ✅ |
| 策略搜索 | ✅ 内置 | ❌ | ❌ | ❌ |
| 通信建模 | ✅ 多级网络 | ✅ | ✅ | ✅ |
| 内存分析 | ✅ | ✅ | ✅ | ✅ |
| 效率校准 | ✅ shape 级 | ✅ | N/A | N/A |
| 开源 | ✅ Apache 2.0 | ✅ | ❌ | ❌ |

SimuMax 的独特定位在**第一个不需要 GPU 的、配置驱动的完整训练性能预测器**，覆盖了从模型定义到策略搜索的端到端流程。它与 Calculon 最为相似，但 SimuMax 在 MoE/MLA 支持和策略搜索方面更为领先。

---

## 六、个人评价

SimuMax 是目前所见最完善的 LLM 分布式训练静态分析模型之一。摩尔线程作为国内 GPU 厂商，开源这样一个工具对社区贡献很大——它不依赖特定硬件厂商的工具链，设计通用性很强。

三输入 JSON 驱动的设计是**工程上的优秀选择**：它让模型的扩展性极好，新增一个模型只需增加一个 10 行 JSON，新增一个硬件也只需一个配置文件。`accurate_efficient_factor` 的设计尤为精巧，在简易性和精确性之间取得了实用的平衡。

主要短板在 compute-communication overlap 建模和 pipeline scheduler 多样性上——这两个恰恰是实际训练中影响性能最重要的因素。期待后续版本补上。

总体来看，这是一个高质量的开源项目，代码结构清晰、文档完善、有公开基准验证。对于从事 LLM 训练基础设施的团队，无论是用于容量规划、策略探索还是 performance debugging，都值得一试。

---

## 参考文献

1. [SimuMax GitHub Repository](https://github.com/MooreThreads/SimuMax)
2. [Calculon](https://github.com/calculon-ai/calculon) — 设计参考来源
3. Megatron-LM: 训练并行策略和选择性 recompute 语义来源
