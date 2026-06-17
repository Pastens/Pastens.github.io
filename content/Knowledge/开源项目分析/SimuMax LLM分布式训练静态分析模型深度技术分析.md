---
tags:
  - 开源项目分析
  - 训练系统
  - 性能建模
  - 仿真
source: https://github.com/MooreThreads/SimuMax
license: Apache 2.0
created: 2026-06-03
updated: 2026-06-17
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

## 四、负载模型（Workload Model）实现机制

SimuMax 的工作负载模型本质是一个**配置驱动、静态分解的性能计算管道**。其核心是将"一次训练迭代"拆解为可独立计算的原子单元，再逐层聚合。

### 4.1 负载分解的层次结构

```
ModelConfig (model.json)
    │   hidden_size, layer_num, head_num, intermediate_size, ...
    ▼
LLMModel (language_model.py)
    │   Embedding + N×LLMBlock + ParallelCE
    ▼
LLMBlock (每层 Transformer)
    │   Attention/MLAAttention + MLP/Swiglu
    ▼
子模块 (dense_module.py)
    │   LinearCol / LinearRow / LayerNorm
    ▼
计算操作 (base_struct.py)
    AtomModel(fwd_cost, bwd_cost) — 最终执行单元
```

这个分解是**在 `prefill()` 阶段完成的**。每个 `MetaModule.prefill()` 将自身展开为 `AtomModel`（计算单元）和 `Com`（通信单元）组成的有序列表，这些单元最终被 `SimuSystem` 调度执行。

### 4.2 TensorSize：虚拟张量系统

SimuMax 不需要真正的张量数据，它使用 `TensorSize` 维护虚拟形状：

```python
# tensor.py
class TensorSize:
    shape: Tuple[int, ...]  # [B, S, H]
    dtype: str              # "bf16", "fp8"...
    
    def numel(self) -> int: return prod(self.shape)
    def mem_size(self): return self.numel() * BPE[self.dtype]
```

张量通过 `InputOutputInfo` 在模块间传递，每个 `MetaModule` 的 `create_output_info()` 基于输入形状和 TP 切分规则计算输出形状。这形成了一条**纯形状驱动的数据流图**——不涉及真实数值计算。

### 4.3 FLOPs 计算：`_comp_leaf_flops_info()`

每个叶子模块硬编码了自己的 FLOPs 公式。以 `LinearCol` 为例：

```python
def _comp_leaf_flops_info(self):
    base_flops = 2 * self.micro_hidden_state_size * self.output_size
    self._compute_info.fwd_flops = base_flops
    self._compute_info.bwd_grad_act_flops = base_flops
    self._compute_info.bwd_grad_w_flops = base_flops
    self._compute_info.recompute_flops = base_flops if self.enable_recompute else 0
```

关键设计：**forward / backward_grad_act / backward_grad_w 三个方向的 FLOPs 独立计算**。这比简单的 `forward * 2` 精确得多，因为不同模块的 FLOPs 分布差异很大——Embedding 为 0，MoE Router 仅有 gating 线性层，LayerNorm 仅涉及 element-wise 操作。

### 4.4 内存访问量计算：`_comp_leaf_mem_accessed_info()`

```python
def _comp_leaf_mem_accessed_info(self):
    weight_size = input_size * output_size * w_element_size
    input_size = B * S * H * a_element_size
    output_size = B * S * output_dim * element_size
    
    self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
    self._compute_info.bwd_grad_act_accessed_mem = weight_size + output_size + input_size
    main_grad_size = input_size * output_size * 4  # fp32
    self._compute_info.bwd_grad_w_accessed_mem = (
        output_size + input_size + weight_size + 
        (main_grad_size if use_fused_grad_accumulation else 0)
    )
```

这里包含**融合梯度累加**（`use_fused_grad_accumulation`）的内存优化假设——如果启用了 fused grad accumulation，backward 时不需要额外的临时 buffer。

### 4.5 FLOPs → 时间：`_comp_cost_info_impl()`

```python
def _comp_cost_info_impl(self, fwd_op, bwd_grad_act_op, bwd_grad_w_op, enable_recompute):
    def compute_details(op_name, stage, flops, accessed_mem):
        compute_details = system.compute_op_accuracy_time(op_name, flops, shape_desc, reture_detail=True)
        io_details = system.compute_mem_access_time(op_name, accessed_mem, reture_detail=True)
        end2end_time = compute_end2end_time(compute_time, mem_time)
        return end2end_time
    
    self._cost_info.fwd_compute_time = compute_details(fwd_op, 'fwd', ...)
    self._cost_info.bwd_grad_act_time = compute_details(bwd_grad_act_op, 'bwd_grad_act', ...)
    self._cost_info.bwd_grad_w_time = compute_details(bwd_grad_w_op, 'bwd_grad_w', ...)
```

其中 `compute_op_accuracy_time()` 是 FLOPs → 时间的转换核心：

```python
def compute_op_accuracy_time(self, op_name, flops, shape_desc):
    op = self.accelerator.op.get(op_name, None)
    if op.accurate_efficient_factor.get(shape_desc):
        eff = op.accurate_efficient_factor[shape_desc]  # 实测值
    else:
        eff = op.efficient_factor  # 默认值
    time = flops / (op.tflops * 1e12 * eff) * 1e3  # 单位 ms
    return time
```

**负载模型的核心抽象**：workload = `{flops, mem_bytes}` → time = max(flops/peak, mem/bw) × efficiency。这个二元组是所有性能预测的原子输入。

### 4.6 Workload 模型的局限

| 维度 | 当前实现 | 未建模 |
|------|---------|--------|
| 计算 | 纯 FLOPs 计数 + 校准效率 | 无 warp 调度、银行冲突、tile 划分细节 |
| 内存 | 宏观访问字节 | 无 cache 层次、L1/L2 miss 率、bank conflict |
| 通信 | 带宽 × 延迟模型 | 无拓扑感知、无 congestion、无多流 overlap |
| 动态性 | 固定 batch/seq | 无动态 shape、无 variadic sequence length |

---

## 五、训练框架优化建模（Training Framework Optimization Modeling）

### 5.1 Recompute 建模（最精细的部分）

Recompute 在 SimuMax 中有**三层建模**：

**Layer 1：策略配置解析** (`StrategyConfig.parse_attention_recompute()`)

```
strategy.json 中的配置
    recompute_granularity = "selective_recompute"
    attn_recompute = True
    mlp_recompute = False
    ↓
parse_attention_recompute()
    → q_up_recompute = True, kv_up_recompute = True
    → core_attn_recompute = True
    → input_layernorm_recompute = False
    ↓
parse_mlp_recompute()
    → linear_recompute = False
    → router_recompute = False
```

**Layer 2：激活缓存策略** (`RecomputeStatus`)

```python
class RecomputeStatus:
    NO_RECOMPUTE = "no_recompute"
    FIRST = "first"    # 第一个 checkpoint 模块——缓存输入
    MIDDLE = "middle"  # 中间模块——fwd 从上一个 checkpoint 重算
    LAST = "last"      # 最后一个——相当于无 checkpoint
```

在 `build_simu_mem_profile()` 中：
- `FIRST` 模块：缓存 `cache_size_bytes`（输入），在 bwd 开始时释放
- `MIDDLE` 模块：不缓存，backward 前从上一个 checkpoint 重算
- `LAST` 模块：等同无 recompute

**Layer 3：仿真调度** (`RecomputeBlockJob`)

```python
class RecomputeBlockJob:
    def bwd(self, t, ctx):
        if self._has_recompute and not self._recompute_done:
            ok, blk = self.recompute_fwd.step(t, ctx)  # 先重放 forward
            self._recompute_done = True
        return self.bwd_stk.bwd(t, ctx)  # 再执行 backward
```

这模拟了 Megatron-LM 的 checkpoint 语义：backward 开始前先重算 forward 算子。

**Megatron-LM 兼容的 selective recompute** (`megatron_recompute_modules`)：
- 支持 `core_attn`, `layernorm`, `mla_up_proj`, `moe_act`, `mlp`, `moe` 六种模块
- 精确映射到 `AttentionRecomputeConfig` / `MLPRecomputeConfig` 的子字段
- `recompute_variance`：最后一层 recompute 的激活节省（variance tail model）

### 5.2 Pipeline 调度建模

`PpSchedule` 实现了三种调度策略，代码量 959 行：

| 策略 | 代码路径 | 特点 |
|------|---------|------|
| 1F1B | `_prefill_batch_interleaved(interleaving_size=1)` | 标准 warmup→steady→cooldown |
| Interleaved VPP | `_prefill_batch_interleaved(interleaving_size>1)` | Megatron 风格的多 chunk 调度 |
| Sync-VPP | `_prefill_batch_interleaved(pp_comm_async=False)` | 阻塞式 batch_isend_irecv |

```python
# warmup forward
for k in range(num_warmup_microbatches):
    job.append(model.prefill_fwd())
    job.append(comms.send_next())

# 1F1B steady state
for k in range(num_microbatches_remaining):
    job.append(model.prefill_fwd())   # forward
    job.append(model.prefill_bwd())   # backward

# cooldown backward
for k in range(...):
    job.append(model.prefill_bwd())
```

SimuMax 精确模拟了 Megatron 的 VPP warmup 公式：
```python
num_warmup = (pp_size - pp_rank - 1) * 2 + (vp_size - 1) * group_size_per_vp_stage
```

异步 PP 通信方面，提供了完整的 async send/recv 原语：`async_recv_prev`（发起接收）、`async_wait_recv_prev`（等待完成）、`batch_blocking_comm`（Megatron 的 `batch_isend_irecv` 阻塞批量提交）。

Pipeline bubble 的解析计算：

```python
def _compute_bubble_time(self, fwd_bwd_time):
    bubble_time = fwd_bwd_time * (self.strategy.pp_size - 1)
    return bubble_time
```

注意：bubble 计算目前仅支持 1F1B，interleaving 场景标注为 TODO。

### 5.3 ZeRO 优化器建模

`OptimizerSimulator` 建模了 ZeRO-1 的完整通信和计算流程：

```python
class OptimizerSimulator(MetaModule):
    def prefill(self, args):
        layers.append(reduce_scatter(...))  # 稠密参数梯度归约
        layers.append(reduce_scatter(...))  # MoE 参数梯度归约
        layers.append(all_reduce(default_group))  # sync barrier
        layers.append(AtomModel(optim_time))  # optimizer step
        layers.append(all_gather(...))  # 稠密参数广播
        layers.append(all_gather(...))  # MoE 参数广播
```

在 `_compute_optim_time()` 中，optimizer step 被分解为 7 个子步骤：

```
zero_grad_buffer → l2_norm_before_reduce → mul_before_reduce →
l2_norm_after_reduce → grads_clip_after_reduce → adam_step →
copy_main_params_to_model_params
```

每个子步骤通过 `compute_mem_access_time()` 基于内存访问量计算时间，而非 FLOPs——因为 optimizer 是 **memory-bound** 的。

DP 通信建模了 Megatron-LM 的 **bucket 式 all-gather / reduce-scatter**：

```python
bucket_size = max(40000000, 1000000 * group_size) * 4
num_bucket = ceil(comm_size / bucket_size)
comm_time = num_bucket * compute_net_op_time("reduce_scatter", bucket_size, ...)
```

### 5.4 通信建模（`compute_net_op_time()`）

支持 5 种操作（all_reduce / all_gather / reduce_scatter / p2p / all2all）、7 级网络（tp / cp / ep / dp / pp / edp / etp）：

```python
def compute_net_op_time(self, op_name, size, comm_num, net, comm_stage, strategy):
    # (1) 调整通信量
    actual_size = size * op.scale + (actual_size / comm_num) * op.offset
    # (2) 带宽决策——跨节点时的带宽退化
    if net == "inter_node":
        if op_name == "p2p":  bw /= num_per_node        # PP 独占网卡
        if op_name == "all2all":  bw /= num_per_node
        if op_name in ["all_reduce", "all_gather", "reduce_scatter"]:
            dense_group_multiplicity = strategy.tp_size
            bw /= min(num_per_node, dense_group_multiplicity)
    # (3) 时间计算
    time = actual_size / (bw * 1024**3 * eff_factor) * 1e3 + latency / 1e3
```

`op.scale` 和 `op.offset` 来自 `system.json` 的 `NetOpConfig`，可模拟不同通信算子的额外开销（如 all2all 的 permutation 延迟）。

### 5.5 Straggler 模型

```python
STRAGGLER_BASE_FACTOR = 0.09

def estimate_straggler_increase_ratio(worker_count):
    n = max(1, int(worker_count))
    n_straggler = log2(n)
    return 1.0 + n_straggler / (n_straggler + 1.0) * 0.09 * sqrt(n_straggler)
```

这个公式对 step time 做放缩，模拟大规模集群中节点间性能差异导致的额外开销。

### 5.6 搜索缓存机制

```python
_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS = {
    "world_size", "pp_size", "micro_batch_num",
    "interleaving_size", "pp_comm_async", "enable_straggler_model",
    "pp_net", "dp_net", "edp_net",  # ... 及派生字段
}
```

核心思想：**将"单个 unit 的本地计算成本"和"PP 布局"解耦**。如果只是 PP 设置变化（如 pp_size=2 → pp_size=4），但 micro_batch_size / TP / EP 不变，则 unit 内部的 FLOPs 和通信量完全相同——可以复用缓存。

---

## 六、组件可替换性分析（Replaceability）

### 6.1 可替换的（配置化/接口化）

| 组件 | 替换方式 | 示例 |
|------|---------|------|
| **硬件模型** | 换 `system.json` | A100 → B200 只需换配置文件 |
| **模型架构** | 换 `model.json` | LLaMA → DeepSeek 只需配置 |
| **并行策略** | 换 `strategy.json` | TP2 → TP8 只需配置 |
| **计算效率** | 换 `accurate_efficient_factor` | 自定义实测效率表 |
| **网络拓扑** | 改 `system.json` 的 `networks` 节 | 自定义多级拓扑参数 |
| **通信参数** | 改 `NetOpConfig` 的 `scale`/`offset` | 自定义通信模型系数 |
| **Roofine 模式** | 改 `accelerator.mode` | `only_compute` ↔ `roofline` |

### 6.2 有条件可替换的

| 组件 | 替换难度 | 说明 |
|------|---------|------|
| **Pipeline 调度器** | ⭐⭐⭐ 中等 | `PpSchedule` 是独立类，可实现新 `prefill()` 替换，但需理解 barrier + 时间推进协议 |
| **优化器模型** | ⭐⭐⭐ 中等 | `OptimizerSimulator` 继承 `MetaModule`，可子类化替换 `prefill()` |
| **Memory tracker** | ⭐⭐ 较易 | `simu_memory.py` 的 `OpMemoryProfile` 相对独立 |
| **Strategy searcher** | ⭐⭐⭐ 中等 | 可替换搜索算法，但依赖 `PerfLLM` 的 API |

### 6.3 当前不可替换的

| 组件 | 不可替换原因 |
|------|-------------|
| **算子 FLOPs 公式** | 每个叶子模块的 `_comp_leaf_flops_info()` 是硬编码公式。无注册机制注入自定义算子 |
| **通信模型公式** | `compute_net_op_time()` 的带宽退化逻辑硬编码了 Megatron-LM 的通信模式。不支持 tree/ring 替换 |
| **Module 体系** | 无抽象基类/接口契约。替换需理解 `__call__()`、`_comp_cost_info()` 等调用链 |
| **仿真调度器** | `SimuSystem` 和 `PpSchedule` 通过 `barrier + yield` 协议协同，难以整体替换 |
| **计算-通信 overlap** | 多处标注 `TODO: support overlap`。当前通信开销都是暴露的 |
| **框架适配层** | 仅映射 Megatron-LM 配置。DeepSpeed/FSDP 需用户间接模拟 |

### 6.4 Hook 机制：有限的扩展点

`MetaModule` 提供了 PyTorch 风格的 hook 机制：

```python
class MetaModule:
    ordered_module_hooks    # 在子模块遍历时触发
    forward_pre_hooks       # forward 前触发
    forward_post_hooks      # forward 后触发
```

这些 hook 允许在每次 `__call__()` 前后注入自定义逻辑（如打印调试、记录统计），但**无法替换计算内核或通信算子**。

### 6.5 可替换性总结

```
┌─── 替换难度分级 ─────────────────────────────────────────────┐
│                                                               │
│  易                                       难                  │
│  ┌──────┐ ┌────────┐ ┌────────────┐ ┌──────────────┐        │
│  │Config│→│ 效率   │→│Pipeline   │→│算子 FLOPs   │         │
│  │(JSON)│ │ 数据   │ │ Scheduler │ │ 公式         │          │
│  └──────┘ └────────┘ └────────────┘ └──────────────┘        │
│  配置化    实测校准    独立类可替换    硬编码在模块中           │
└───────────────────────────────────────────────────────────────┘
```

SimuMax 的**配置层（model/strategy/system）开放度最高**，这是其设计亮点。但**引擎核心（算子公式、通信模型、仿真调度）的替换需要改源码**。没有 plugin/registry 架构，替换特定组件需通过继承 + 猴子补丁实现。

以替换 pipeline scheduler 为例：

```python
class MyCustomSchedule(PpSchedule):
    def prefill(self, args, call_stk='', com_buff=None):
        job = []
        # ... 自定义调度逻辑 ...
        return job
# perf_llm._pp_schedule_class = MyCustomSchedule
```

### 6.6 总结表

| 维度 | 评估 |
|------|------|
| **负载模型粒度** | 算子级（GEMM、norm、softmax 等），原子操作时间精度到 ms |
| **FLOPs 计算** | fwd/bwd_grad_act/bwd_grad_w 三项独立，考虑了 fused kernel 优化 |
| **Recompute 建模** | 业界最精细之一——5 种粒度、两层语义、Megatron 兼容 |
| **Pipeline 建模** | 1F1B + VPP Interleaved + Sync-VPP，精确的 Megatron warmup 公式 |
| **ZeRO 建模** | bucket 粒度通信、7 步 optimizer 分解 |
| **配置可替换性** | 高（三层 JSON） |
| **代码级可替换性** | 中低（无 plugin 机制，需改源码） |
| **核心改进方向** | 计算-通信 overlap、更多 pipeline 调度、算子插件系统 |

---

## 七、亮点与局限

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

## 八、与其他项目的对比

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

## 九、个人评价

SimuMax 是目前所见最完善的 LLM 分布式训练静态分析模型之一。摩尔线程作为国内 GPU 厂商，开源这样一个工具对社区贡献很大——它不依赖特定硬件厂商的工具链，设计通用性很强。

三输入 JSON 驱动的设计是**工程上的优秀选择**：它让模型的扩展性极好，新增一个模型只需增加一个 10 行 JSON，新增一个硬件也只需一个配置文件。`accurate_efficient_factor` 的设计尤为精巧，在简易性和精确性之间取得了实用的平衡。

主要短板在 compute-communication overlap 建模和 pipeline scheduler 多样性上——这两个恰恰是实际训练中影响性能最重要的因素。期待后续版本补上。

总体来看，这是一个高质量的开源项目，代码结构清晰、文档完善、有公开基准验证。对于从事 LLM 训练基础设施的团队，无论是用于容量规划、策略探索还是 performance debugging，都值得一试。

---

## 参考文献

1. [SimuMax GitHub Repository](https://github.com/MooreThreads/SimuMax)
2. [Calculon](https://github.com/calculon-ai/calculon) — 设计参考来源
3. Megatron-LM: 训练并行策略和选择性 recompute 语义来源
