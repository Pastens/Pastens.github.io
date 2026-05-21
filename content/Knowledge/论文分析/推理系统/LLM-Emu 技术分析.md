---
tags:
- 论文分析
- llm-inference
- emulator
- vllm
source: https://github.com/AKafakA/llm-emu
arxiv: 2605.00616
author: WEI DA
created: 2026-05-09
rating: ⭐⭐⭐⭐⭐
---

# LLM-Emu: Profile-Driven Online Emulation of LLM Serving Systems

## 一、项目概览

| 项目属性 | 内容 |
|---------|------|
| **仓库** | [AKafakA/llm-emu](https://github.com/AKafakA/llm-emu) |
| **描述** | Profile-driven online emulator for vLLM — 用真实 GPU 上采集的 profile pack 替换 vLLM 的 GPU 前向传播，实现无 GPU 环境下的推理系统仿真 |
| **语言** | Python (86%) + Shell (14%) |
| **代码规模** | ~306 KB Python, ~51 KB Shell |
| **许可证** | Apache-2.0 |
| **论文** | arXiv: 2605.00616 |

### 核心思想

传统 LLM 推理仿真通常需要构建完整的模拟器，这导致模型与实际系统的行为有偏差。LLM-Emu 采用了截然不同的方法：**运行真实的 vLLM 代码路径**（调度器、HTTP 栈、admission 控制、分词器、输出管道等全部不变），**仅用一个 profile pack 中采样的延迟数据替换 GPU 前向传播**。这使得仿真器在宏观指标（TTFT、TPOT、ITL、E2E、TPS）上达到 <5% 的中位误差，且完全无需 GPU。

---

## 二、仓库结构与代码布局

```
llm-emu/
├── vllm_emulator/                    # LLM-Emu 核心插件 (~1.7K LoC 在线 + ~0.6K LoC 离线工具)
│   ├── __init__.py                   # 入口：条件导入 cuda_mock + 暴露 EmulatorPlatform
│   ├── platform.py                   # vLLM platform-plugin 入口点
│   ├── cuda_mock.py                  # torch.cuda 存根, 用于无 GPU 主机 (749 行)
│   ├── hooks/
│   │   └── executor_hook.py          # 拦截 execute_model(), 返回延迟 Future (275 行)
│   ├── oracle/
│   │   ├── base.py                   # 抽象基类 BaseGpuCostOracle
│   │   └── gpu_cost_oracle.py        # 自适应 K Shepard 池化密度感知 Oracle (299 行)
│   ├── profile/
│   │   ├── loader.py                 # Profile pack JSON 加载 + 校验
│   │   ├── validator.py              # Schema 验证 (128 行)
│   │   └── build_serving_profile_filtered.py  # Trace -> Profile pack 构建器
│   └── profiler/
│       └── trace_profiler.py         # StepCycleTracer: 真实 GPU 运行时 trace 采集
│
├── tools/                            # 14 个论文复现脚本
├── vllm_patches/                     # vLLM v0.18.1 桥接补丁 (~170 行)
│   ├── llm-emu-vllm-0.18.1.patch
│   └── overrides/vllm/v1/
│       ├── engine/core.py            # StepCycleTracer 初始化 (+135 行)
│       ├── executor/uniproc_executor.py  # 转发 execute_model 到 hook (+35 行)
│       └── core/sched/scheduler.py   # 三行 shim (+3 行)
│
├── example_profiling_data/
│   └── A40-Q8-Qwen3-8B.json         # Qwen3-8B on A40 预采集数据 (~11MB)
├── docs/
│   └── reproduce.md                  # 论文 Table 1 复现指南
├── pyproject.toml
└── README.md
```

---

## 三、核心技术架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────┐
│             客户端请求 (HTTP/vLLM API)              │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│              vLLM Engine Core (未修改)             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Scheduler│  │ Tokenizer│  │ Output Pipeline│  │
│  └─────┬────┘  └──────────┘  └───────────────┘  │
└────────┼────────────────────────────────────────┘
         │ scheduler_output
         ▼
┌─────────────────────────────────────────────────┐
│        ExecutorEmulatorHook (核心拦截点)           │
│  ┌──────────────────────────────────────────┐   │
│  │ ProfileGpuCostOracle (延迟预测 oracle)     │   │
│  │ - 2D (total_tokens, concurrency) 查表      │   │
│  │ - K=1 / K=auto (自适应 Shepard 池化)       │   │
│  │ - 从实时采集的分布中随机采样                  │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │ Fake Output Generator                     │   │
│  │ - 生成 ModelRunnerOutput (filler token)   │   │
│  │ - threading.Timer / time.sleep 控制延迟   │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 3.2 工作流程

**在线仿真模式 (Oracle Enabled)**

1. `vllm serve` 启动，`pyproject.toml` 注册的 `vllm.platform_plugins` 入口点被调用
2. `emulator_platform_plugin()` 检查 `VLLM_EMULATOR_ENABLE_ORACLE=1`，返回 `EmulatorPlatform`
3. `vllm_emulator/__init__.py` 被导入，自动执行 `cuda_mock.install()` 和 `install_model_stubs()`
4. `EmulatorPlatform` 接管 `check_and_update_config()`，设置 `device="cuda"` 但通过 mock 模拟全部 CUDA 行为
5. `ExecutorEmulatorHook` 初始化，加载 profile pack，创建 `ProfileGpuCostOracle`
6. 每个调度步骤，补丁过的 `uniproc_executor.py` 调用 `get_executor_hook()` 获取 hook
7. Hook 将 `scheduler_output` 传给 oracle 预测延迟，生成 filler token 假输出，通过 `threading.Timer` 异步返回
8. vLLM 引擎继续处理输出 — 所有代码路径与真实 GPU 运行完全一致

**Profile 采集模式 (Trace Enabled)**

1. 设置 `VLLM_EMULATOR_TRACE_STEP_CYCLE=1`，在真实 GPU 上运行
2. 补丁过的 `engine/core.py` 创建 `StepCycleTracer` 实例
3. 每个 `_process_engine_step()` 记录 `(total_tokens, num_new_reqs, num_decode_seqs, step_cycle_us)` 到 JSONL
4. `build_serving_profile_filtered.py` 将原始 trace 转换为 2D 桶分布的 profile pack JSON

---

## 四、关键模块深度分析

### 4.1 `cuda_mock.py` — CUDA 存根系统 (749 行) ⭐

项目中最精妙的模块。解决核心挑战：让 vLLM 这样的 GPU 专用框架在无 GPU 主机上正常运行。

**技术亮点：**

1. **多层修补策略**：从 Python 层到 C 扩展层全覆盖
   - `torch.cuda.*` 函数级替换 (is_available, device_count, mem_get_info 等)
   - `torch.Tensor.cuda()` / `.to('cuda')` 重定向到 CPU
   - `torch._C._cuda_init`, `torch._C._cuda_getDeviceCount` 等 C 层函数替换
   - `torch.accelerator` 模块全面覆盖
   - `FakeStream` / `FakeEvent` 类替代 CUDA stream/event
   - 全局扫描 `sys.modules` 中缓存的对原 `Stream`/`Event` 类的引用

2. **元路径导入钩子**：当 `vllm.v1.worker.gpu_model_runner` 和 `gpu_worker` 被导入时自动修补：
   - `GPUModelRunner.load_model` → no-op
   - `GPUModelRunner.profile_run` → no-op
   - `GPUModelRunner._dummy_run` → 返回空 hidden states
   - `GPUWorker.init_device` → 跳过 CUDA 初始化
   - `GPUWorker.determine_available_memory` → 从 profile pack 读取
   - `GPUWorker.get_kv_cache_spec` → 从 HF config 合成

3. **分布式通信短路**：world_size=1 时，`torch.distributed.barrier()` / `all_reduce()` 变为 no-op
4. **NCCL 强制禁用**：在 `CUDA_VISIBLE_DEVICES=""` 环境下强制 `is_nccl_available()` 返回 False
5. **GPU 参数从 profile pack 读取**：GPU 名称、显存大小、SM 数量等

### 4.2 `platform.py` — vLLM 平台插件 (139 行)

作为 `vllm.platform_plugins` 入口点，实现 `EmulatorPlatform(Platform)` 类：
- `check_and_update_config()`：根据环境变量设置 device，禁用编译
- `get_device_total_memory()`：支持环境变量覆写
- `get_attn_backend_cls()`：强制使用 `TritonAttentionBackend`（最轻量）
- 设备属性全部从 profile pack 读取

### 4.3 `hooks/executor_hook.py` — 执行器拦截钩子 (275 行)

**核心功能**：接收 `scheduler_output`，预测延迟，生成假输出。

- **异步模式 (non_block=True)**：使用 `threading.Timer` 创建延迟未来，维护虚拟 GPU 时间线 `_gpu_free_time`，链式叠加延迟
- **同步模式 (non_block=False)**：直接 `time.sleep(latency_s)`
- **加速模式 (accelerated)**：不 sleep 但依旧生成输出用于调试
- **假输出生成**：区分 prefill chunk 和 decode 步骤，从 profile pack 读取 `eos_token_id`
- **前缀缓存感知**：通过 `num_computed_tokens` 识别 prefix-cache 命中的请求

### 4.4 `oracle/gpu_cost_oracle.py` — 延迟预测 Oracle (299 行)

**最具学术创新性的模块。**

**算法：**
1. **2D 桶分布**：Profile pack 包含 `prefill_2d_distribution` 和 `decode_2d_distribution`，键为 `(total_tokens, concurrency)`
2. **K=1 最近邻 (默认)**：找到最近的桶，均匀随机采样
3. **K=auto 自适应 Shepard 池化**：
   - 计算所有桶到查询点的归一化欧氏距离 `d = sqrt((Δtt/rtt)² + (Δconc/rconc)²)`
   - 累加桶的样本数直到达到 `VLLM_EMULATOR_ORACLE_MIN_SAMPLES` (默认 30)
   - 使用 Shepard p=2 逆距离加权 (`w = 1/d²`) 进行概率选择
   - 精确匹配 (d=0) 时直接返回该桶
4. **百分位裁剪**：`VLLM_EMULATOR_SAMPLE_TRIM=lo,hi` 裁剪极端值
5. **确定性随机**：`random.Random(42)` 固定种子

### 4.5 `profiler/trace_profiler.py` — 步周期跟踪器 (374 行)

- `StepCycleTracer`：嵌入 vLLM EngineCore，测量完整的 `_process_engine_step()` 周期
- 自描述 header：自动采集 GPU 属性、模型架构、调度器配置
- 批量刷新：每 200 条记录 flush 一次
- 标记支持：支持 `__marker__: profiling_start/stop` 标记

### 4.6 `profile/build_serving_profile_filtered.py` (183 行)

- 2D 桶化：`total_tokens` 和 `concurrency` 按配置的桶宽度分桶
- 自动分类：根据 `num_new_reqs > 0` 将步骤分为 prefill 和 decode
- 输出格式：包含 `version`, `gpu_model`, `model_name`, `profile_type`, 三种分布表, `model_config`

### 4.7 `vllm_patches/` — vLLM 桥接补丁 (~170 行)

| 文件 | 改动量 | 作用 |
|------|-------|------|
| `engine/core.py` | +135 行 | StepCycleTracer 初始化 + 每步 trace 记录点 |
| `uniproc_executor.py` | +35 行 | 转发 `execute_model` 到 `ExecutorEmulatorHook` |
| `scheduler.py` | +3 行 | 暴露 `num_computed_tokens` 用于前缀缓存感知 |

所有改动仅在相应的环境变量设置时生效，未设置时 vLLM 行为与上游完全一致。

---

## 五、依赖与技术栈

| 组件 | 技术 |
|------|------|
| **核心框架** | vLLM == 0.18.1 (严格 pinned) |
| **深度学习** | PyTorch (仅使用 torch.cuda mock 接口) |
| **并行后端** | GLOO (强制禁用 NCCL) |
| **模型配置** | HuggingFace Transformers (AutoConfig) |
| **数据格式** | JSON / JSONL |
| **补丁系统** | unified diff (patch) |

**无额外 runtime 依赖。** `pyproject.toml` 中唯一的 Python 依赖是 `vllm==0.18.1`。

---

## 六、亮点与创新点

### 6.1 架构创新：实时替换 vs 模拟

最大的创新在于 **"真实系统 + 定点替换"** 的范式。不重建调度逻辑、不模拟内存管理，让 vLLM 本身运行所有真实代码，仅在 GPU 前向传播处插入 profile 驱动的延迟采样。自然继承了 vLLM 的所有行为特性（包括 bug、调度策略变化、内存碎片、前缀缓存等）。

### 6.2 精度：<5% 误差无 GPU

6 个实验细胞（不同模型、GPU、工作负载）验证：
- TTFT 误差通常 <10%
- TPOT/ITL 误差 **<5%**
- TPS 误差 **<2%**

### 6.3 侵入性极小

仅需修改 vLLM 的 3 个文件共 ~170 行，且全部是追加而非修改。插件模式通过 vLLM 官方 `vllm.platform_plugins` 入口点注册。

### 6.4 CUDA Mock 的完整性

从 Python 模块层 → C 扩展层 → `torch.accelerator` → 元路径导入钩子，展示了极其扎实的系统工程功底。

### 6.5 自适应 Shepard 池化

范围归一化的 2D 距离、自适应 K 值、逆距离加权概率采样、初始样本数下限。学术论文级别的设计。

### 6.6 完整的复现流水线

14 个工具脚本 + 3 阶段编排器 → 论文格式输出 + 审计脚本。

---

## 七、潜在改进方向

### 7.1 多 GPU / 张量并行支持

当前明确只支持单 GPU (tp=dp=1)。扩展到多 GPU 需要处理 collectives 延迟同步、通信开销建模。

### 7.2 KV 缓存开销建模

当前仅使用 `(total_tokens, concurrency)` 作为特征，`sum_kv` 参数已传入 oracle 接口但尚未使用。

### 7.3 调度器反馈回路

假输出不包含真实 logits/logprobs，vLLM 的输出采样（temperature, top-p 等）实际不生效，output length 分布与真实运行可能有差异。

### 7.4 更丰富的 profile 特征

可扩展维度：平均序列长度、KV cache 命中率、模型架构特定的计算图特征。

### 7.5 跨版本兼容性

Patch 严格绑定 vLLM v0.18.1，升级需要重新生成 patch。

### 7.6 性能开销

CPU-only 运行完整 vLLM 流程在小速率时有约 +0.34%，大速率时 +2-5% 的 E2E 开销。

---

## 八、总结与洞察

### 关键数据点

- **代码量**：核心插件仅 ~1.7K LoC，加上 ~170 行 vLLM 补丁
- **精度**：TTFT <10%, TPOT/ITL <5%, TPS <2%
- **部署开销**：无 GPU 即可运行，profile 采集需要 ~3 小时真实 GPU 时间
- **工作负载范围**：Qwen3-4B/8B/14B, Llama-3.1-8B, RTX8000, A40

### 适用场景

1. **容量规划**：无 GPU 模拟不同负载速率下的系统行为
2. **调度算法研究**：快速迭代调度器策略
3. **系统调试**：复现和诊断 vLLM 生产问题
4. **论文复现**：配套的复现流水线使结果可审计
5. **CI/CD 测试**：无 GPU 的 CI 环境中测试 vLLM 部署配置

### 最终评价

**一个工程实现极其扎实的学术项目。** CUDA mock 系统的完整性、oracle 算法的设计、插件的低侵入性架构，展示了作者对 vLLM 内部机制和 PyTorch 底层接口的深刻理解。对于任何需要在无 GPU 环境下保真仿真 LLM 推理系统的团队，这是一个值得投入的解决方案。

---

## 相关笔记

