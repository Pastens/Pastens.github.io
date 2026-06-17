---
tags:
  - 开源项目分析
  - 训练系统
  - 性能建模
  - 仿真
  - 代码深度分析
source: https://github.com/MooreThreads/SimuMax
license: Apache 2.0
created: 2026-06-17
rating: ⭐⭐⭐⭐⭐
---

# SimuMax 负载模型与训练框架优化建模 代码级深度分析

> 本文是 [[SimuMax LLM分布式训练静态分析模型深度技术分析]] 的补充，聚焦负载模型（workload model）的实现机制、训练框架优化的建模方式，以及各组件的可替换性分析。

---

## 一、负载模型（Workload Model）实现机制

SimuMax 的工作负载模型本质是一个**配置驱动、静态分解的性能计算管道**。其核心是将"一次训练迭代"拆解为可独立计算的原子单元，再逐层聚合。

### 1.1 负载分解的层次结构

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

### 1.2 TensorSize：虚拟张量系统

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

### 1.3 FLOPs 计算：`_comp_leaf_flops_info()`

每个叶子模块硬编码了自己的 FLOPs 公式。以 `LinearCol` 为例：

```python
def _comp_leaf_flops_info(self):
    # micro_hidden_state_size = B * S * H (考虑 SP 后还原全长)
    base_flops = 2 * self.micro_hidden_state_size * self.output_size
    self._compute_info.fwd_flops = base_flops
    self._compute_info.bwd_grad_act_flops = base_flops
    self._compute_info.bwd_grad_w_flops = base_flops
    # recompute = fwd flops（如果启用）
    self._compute_info.recompute_flops = base_flops if self.enable_recompute else 0
```

关键设计：**forward / backward_grad_act / backward_grad_w 三个方向的 FLOPs 独立计算**。这比简单的 `forward * 2` 或 `forward * 3` 精确得多，因为：
- `Embedding._comp_leaf_flops_info()` 中所有 flops = 0（无运算）
- MoE Router 的 FLOPs = `2 * B * S * H * expert_num`（仅有 gating 线性层）
- LayerNorm 的 FLOPs 只涉及 element-wise 操作

### 1.4 内存访问量计算：`_comp_leaf_mem_accessed_info()`

```python
def _comp_leaf_mem_accessed_info(self):
    weight_size = input_size * output_size * w_element_size
    input_size = B * S * H * a_element_size
    output_size = B * S * output_dim * element_size
    
    self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
    self._compute_info.bwd_grad_act_accessed_mem = weight_size + output_size + input_size
    # bwd_grad_w 多一个 gradient accumulation 的读写
    main_grad_size = input_size * output_size * 4  # fp32
    self._compute_info.bwd_grad_w_accessed_mem = (
        output_size + input_size + weight_size + 
        (main_grad_size if use_fused_grad_accumulation else 0)
    )
```

这里包含了**融合梯度累加**（`use_fused_grad_accumulation`）的内存优化假设——如果启用了 fused grad accumulation，backward 时不需要额外的临时 buffer。

### 1.5 FLOPs → 时间：`_comp_cost_info_impl()`

```python
def _comp_cost_info_impl(self, fwd_op, bwd_grad_act_op, bwd_grad_w_op, enable_recompute):
    def compute_details(op_name, stage, flops, accessed_mem):
        # (1) 计算时间: FLOPs / (peak_tflops * efficiency)
        compute_details = system.compute_op_accuracy_time(op_name, flops, shape_desc, reture_detail=True)
        # (2) 访存时间: bytes / (bw * efficiency)
        io_details = system.compute_mem_access_time(op_name, accessed_mem, reture_detail=True)
        # (3) Roofline: final_time = max(compute, mem)
        end2end_time = compute_end2end_time(compute_time, mem_time)
        return end2end_time
    
    self._cost_info.fwd_compute_time = compute_details(fwd_op, 'fwd', ...)
    self._cost_info.bwd_grad_act_time = compute_details(bwd_grad_act_op, 'bwd_grad_act', ...)
    self._cost_info.bwd_grad_w_time = compute_details(bwd_grad_w_op, 'bwd_grad_w', ...)
```

其中 `compute_op_accuracy_time()` 是关键：

```python
def compute_op_accuracy_time(self, op_name, flops, shape_desc):
    op = self.accelerator.op.get(op_name, None)
    # 优先使用 shape 级精确效率因子
    if op.accurate_efficient_factor.get(shape_desc):
        eff = op.accurate_efficient_factor[shape_desc]  # 实测值
    else:
        eff = op.efficient_factor  # 默认值
    time = flops / (op.tflops * 1e12 * eff) * 1e3  # 单位 ms
    return time
```

**负载模型的核心抽象**：workload = `{flops, mem_bytes}` → time = max(flops/peak, mem/bw) × efficiency。这个二元组（compute + memory access）是所有性能预测的原子输入。

### 1.6 Workload 模型的局限

| 维度 | 当前实现 | 未建模 |
|------|---------|--------|
| 计算 | 纯 FLOPs 计数 + 校准效率 | 无 warp 调度、银行冲突、tile 划分细节 |
| 内存 | 宏观访问字节 | 无 cache 层次、L1/L2 miss 率、bank conflict |
| 通信 | 带宽 × 延迟模型 | 无拓扑感知、无 congestion、无多流 overlap |
| 动态性 | 固定 batch/seq | 无动态 shape、无 variadic sequence length |

---

## 二、训练框架优化建模（Training Framework Optimization Modeling）

### 2.1 Recompute 建模（最精细的部分）

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
    FIRST = "first"    # 第一个被 checkpoint 的模块——缓存输入
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
            # 先重放 forward
            ok, blk = self.recompute_fwd.step(t, ctx)
            self._recompute_done = True
        # 再执行 backward
        return self.bwd_stk.bwd(t, ctx)
```

这模拟了 Megatron-LM 的 checkpoint 语义：backward 开始前先重算 forward 算子。

**Megatron-LM 兼容的 selective recompute** (`megatron_recompute_modules`)：
- 支持 `core_attn`, `layernorm`, `mla_up_proj`, `moe_act`, `mlp`, `moe` 六种模块
- 精确映射到 `AttentionRecomputeConfig` / `MLPRecomputeConfig` 的子字段
- `recompute_variance`：最后一层 recompute 的激活节省（variance tail model）

### 2.2 Pipeline 调度建模

`PpSchedule` 实现了三种调度策略，代码量 959 行：

| 策略 | 代码路径 | 特点 |
|------|---------|------|
| 1F1B | `_prefill_batch_interleaved(interleaving_size=1)` | 标准 warmup→steady→cooldown |
| Interleaved VPP | `_prefill_batch_interleaved(interleaving_size>1)` | Megatron 风格的多 chunk 调度 |
| Sync-VPP | `_prefill_batch_interleaved(pp_comm_async=False)` | 阻塞式 batch_isend_irecv |

**调度构建的核心逻辑**：

```python
# warmup forward
for k in range(num_warmup_microbatches):
    job.append(model.prefill_fwd())
    job.append(comms.send_next())  # 发送到下一 stage

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

异步 PP 通信方面，提供了完整的 async send/recv 原语：
- `async_recv_prev` → 发起接收
- `async_wait_recv_prev` → 等待完成
- `batch_blocking_comm` → Megatron 的 `batch_isend_irecv`（阻塞但批量提交）

**Pipeline bubble 的解析计算**：

```python
def _compute_bubble_time(self, fwd_bwd_time):
    bubble_time = fwd_bwd_time * (self.strategy.pp_size - 1)
    return bubble_time
```

注意：SimuMax 的 bubble 计算目前仅支持 1F1B，interleaving 场景的 bubble 计算标注为 TODO。

### 2.3 ZeRO 优化器建模

`OptimizerSimulator` 建模了 ZeRO-1 的完整通信和计算流程：

```python
class OptimizerSimulator(MetaModule):
    def prefill(self, args):
        # 1. reduce_scatter (all ranks → one rank per shard)
        layers.append(reduce_scatter(...))  # 稠密参数
        layers.append(reduce_scatter(...))  # MoE 参数
        # 2. sync barrier
        layers.append(all_reduce(default_group))
        # 3. optimizer step
        layers.append(AtomModel(optim_time))
        # 4. all_gather (one rank → all ranks)
        layers.append(all_gather(...))
        layers.append(all_gather(...))
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

### 2.4 通信建模（`compute_net_op_time()`）

`compute_net_op_time()` 是通信建模的核心，支持 5 种操作、7 级网络：

```python
def compute_net_op_time(self, op_name, size, comm_num, net, comm_stage, strategy):
    # (1) 调整通信量: actual_size = size * scale + chunk_size * offset
    actual_size = size * op.scale + (actual_size / comm_num) * op.offset
    
    # (2) 带宽决策——跨节点时的带宽退化
    if net == "inter_node":
        if op_name == "p2p":      bw /= num_per_node  # PP 独占网卡
        if op_name == "all2all":  bw /= num_per_node  # 同卡分发/收集
        if op_name in ["all_reduce", "all_gather", "reduce_scatter"]:
            # DP 组依赖于 TP 大小
            dense_group_multiplicity = strategy.tp_size
            bw /= min(num_per_node, dense_group_multiplicity)
    
    # (3) 时间计算: actual_size / (bw * eff) + latency
    time = actual_size / (bw * 1024**3 * eff_factor) * 1e3 + latency / 1e3
```

这里的 `op.scale` 和 `op.offset` 来自 `system.json` 中的 `NetOpConfig`，可以模拟不同通信算子的额外开销（如 all2all 的 permutation 延迟）。

### 2.5 Straggler 模型

SimuMax 包含一个经验性的 straggler 模型：

```python
STRAGGLER_BASE_FACTOR = 0.09

def estimate_straggler_increase_ratio(worker_count):
    n = max(1, int(worker_count))
    n_straggler = log2(n)
    return 1.0 + n_straggler / (n_straggler + 1.0) * 0.09 * sqrt(n_straggler)
```

这个公式对 step time 做放缩，模拟大规模集群中节点间性能差异导致的额外开销。

### 2.6 搜索缓存机制

`_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS` 定义了一个精巧的缓存策略：

```python
_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS = {
    "world_size", "pp_size", "micro_batch_num",
    "interleaving_size", "pp_comm_async", "enable_straggler_model",
    "pp_net", "dp_net", "edp_net",
    # ... 以及派生字段
}
```

核心思想：**将"单个 unit 的本地计算成本"和"PP 布局"解耦**。如果只是 PP 设置变化（如从 pp_size=2 改为 pp_size=4），但 micro_batch_size / TP / EP 不变，则 unit 内部的 FLOPs 和通信量完全相同——可以复用缓存。

---

## 三、组件可替换性分析（Replaceability）

### 3.1 可替换的（配置化/接口化）

| 组件 | 替换方式 | 示例 |
|------|---------|------|
| **硬件模型** | 换 `system.json` | A100 → B200 只需换配置文件 |
| **模型架构** | 换 `model.json` | LLaMA → DeepSeek 只需配置 |
| **并行策略** | 换 `strategy.json` | TP2 → TP8 只需配置 |
| **计算效率** | 换 `accurate_efficient_factor` | 自定义实测效率表 |
| **网络拓扑** | 改 `system.json` 的 `networks` 节 | 自定义多级拓扑参数 |
| **通信参数** | 改 `NetOpConfig` 的 `scale`/`offset` | 自定义通信模型系数 |
| **Roofine 模式** | 改 `accelerator.mode` | `only_compute` ↔ `roofline` |

### 3.2 有条件可替换的

| 组件 | 替换难度 | 说明 |
|------|---------|------|
| **Pipeline 调度器** | ⭐⭐⭐ 中等 | `PpSchedule` 是独立的类，可以实现新的 `prefill()` 方法替换，但需要理解仿真器的 barrier + 时间推进协议 |
| **优化器模型** | ⭐⭐⭐ 中等 | `OptimizerSimulator` 继承自 `MetaModule`，可以子类化并替换 `prefill()` 的通信序列 |
| **Memory tracker** | ⭐⭐ 较易 | `simu_memory.py` 的 `OpMemoryProfile` 相对独立 |
| **Strategy searcher** | ⭐⭐⭐ 中等 | 搜索逻辑在 `tuning/` 下，可以替换搜索算法，但依赖 `PerfLLM` 的 API |

### 3.3 当前不可替换的

| 组件 | 不可替换原因 |
|------|-------------|
| **算子 FLOPs 公式** | 每个 `LinearCol`、`MLP`、`Attention` 等叶子模块的 `_comp_leaf_flops_info()` 是硬编码的公式。没有注册机制让外部注入自定义算子的公式。 |
| **通信模型公式** | `compute_net_op_time()` 中的带宽退化逻辑（`bw /= min(n, group_multiplicity)`）硬编码了 Megatron-LM 的通信模式。不支持自定义通信拓扑算法（如 tree/ring 替换） |
| **Module 体系** | 没有抽象基类/接口契约。`MetaModule` 是一个 `dict` 式的动态类，替换模块需要理解其内部的 `__call__()`、`_comp_cost_info()` 等调用链。 |
| **仿真调度器** | `SimuSystem` 和 `PpSchedule` 通过 `barrier + yield` 协议协同，很难替换为全新的调度体系 |
| **计算-通信 overlap** | 代码中多处标注 `TODO: support overlap`。当前所有通信开销都是暴露的（`dp_comm_exposed_time = dp_comm_time`） |
| **框架适配层** | SimuMax 仅映射 Megatron-LM 的配置和运行时行为。DeepSpeed/FSDP 等框架的特性需要通过配置间接模拟 |

### 3.4 Hook 机制：有限的扩展点

`MetaModule` 提供了 PyTorch 风格的 hook 机制：

```python
class MetaModule:
    # 三类 hook
    ordered_module_hooks    # 在子模块遍历时触发
    forward_pre_hooks       # forward 前触发
    forward_post_hooks      # forward 后触发
    
    def register_module(self, sub_module):
        # 注册一个子模块到有序列表中
        self.children_ordered_module.append(sub_module)
```

这些 hook 允许在每次 `__call__()` 前后注入自定义逻辑（如打印调试、记录统计），但**无法替换计算内核或通信算子**。

### 3.5 可替换性总结

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

SimuMax 的**配置层（model/strategy/system）开放度最高**，这是其设计亮点。但**引擎核心（算子公式、通信模型、仿真调度）的替换需要改源码**。没有 plugin/registry 架构，如果要替换特定组件，需要通过继承 + 猴子补丁的方式实现。

### 3.6 如果要替换某个组件

以替换 pipeline scheduler 为例：

```python
# 方案：继承 PpSchedule 并重写 prefill
from simumax.core.transformer.pipeline_schedule import PpSchedule

class MyCustomSchedule(PpSchedule):
    def prefill(self, args, call_stk='', com_buff=None):
        # 实现你的调度逻辑
        # 需要理解 FwdQue, send_next, recv_prev 等原语
        job = []
        # ... 自定义调度逻辑 ...
        return job

# 然后让 PerfLLM 使用你的调度器：
# perf_llm._pp_schedule_class = MyCustomSchedule
```

这种方式虽然可行，但 SimuMax 的 `PpSchedule` 与 `PerfLLM` 的 `simulate()` 方法通过 `prefill_fwd()`/`prefill_bwd()` 接口耦合，替换时需要确保兼容这些接口的返回值和行为。

---

## 四、代码级关键发现

### 4.1 负载模型的"静"与"动"

SimuMax 的负载模型分为**两个阶段**：

- **静态阶段**（`_comp_cost_info_impl()`）：纯公式计算，秒级出结果
- **动态仿真阶段**（`simulate()` + `PpSchedule`）：事件驱动，产出 timeline

静态阶段使用 `ModuleComputeInfo` 和 `ModuleCostInfo` 作为数据载体；动态阶段使用 `FwdQue` + `AtomModel` 作为执行载体。这种两阶段设计允许在策略搜索时只用静态阶段（极快），只在需要 timeline 时才跑仿真。

### 4.2 效率校准的权衡

`accurate_efficient_factor` 的设计是 SimuMax 最精巧的工程选择：

- **命中**：使用实测 shape 效率（精确但有限覆盖）
- **未命中**：使用 `efficient_factor` 默认值（快速但可能偏差 10-30%）
- `record_miss_efficiency()` 自动记录哪些 shape 没有实测数据，指导 measurement pipeline 的补充方向

这种**兜底 + 精确覆盖**的模式是工程实用主义的最佳实践。

### 4.3 可替换性设计的缺失

SimuMax 最大的架构局限不在功能覆盖，而在**扩展机制**：

1. **没有算子注册表**：新增算子需要修改 `dense_module.py` 或 `moe_module.py`
2. **没有通信后端抽象**：硬编码了 NCCL 的通信模型（ring algorithm）
3. **没有框架适配层**：配置直接对应 Megatron-LM 的语义，其他框架需要用户自己映射
4. **仿真器的 barrier 协议**：`yield_keep` / `yield_done` 返回码是新调度器实现的障碍

这些局限不影响 SimuMax 作为分析工具的价值，但限制了它作为"可组合仿真平台"的潜力。

---

## 五、总结

| 维度 | 评估 |
|------|------|
| **负载模型粒度** | 算子级（GEMM、norm、softmax 等），原子操作时间精度到 ms |
| **FLOPs 计算** | fwd/bwd_grad_act/bwd_grad_w 三项独立，考虑了 fused kernel 优化 |
| **Recompute 建模** | **业界最精细之一**——5 种粒度、两层语义、Megatron 兼容 |
| **Pipeline 建模** | 1F1B + VPP Interleaved + Sync-VPP，精确的 Megatron warmup 公式 |
| **ZeRO 建模** | bucket 粒度通信、7 步 optimizer 分解 |
| **配置可替换性** | 高（三层 JSON） |
| **代码级可替换性** | **中低**（无 plugin 机制，需改源码） |
| **核心改进方向** | 计算-通信 overlap、更多 pipeline 调度、算子插件系统 |
