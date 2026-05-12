---
tags:
  - 论文分析
  - llm-inference
  - system-design
  - prefill-decode-disaggregation
arxiv: 2311.18677
authors:
  - Pratyush Patel (University of Washington / Microsoft)
  - Esha Choukse (Microsoft)
  - Chaojie Zhang (Microsoft)
  - Aashaka Shah (Microsoft)
  - Íñigo Goiri (Microsoft)
  - Saeed Maleki (Microsoft)
  - Ricardo Bianchini (Microsoft)
institutions:
  - University of Washington
  - Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
venue: ISCA 2024
---

# Splitwise: Efficient Generative LLM Inference Using Phase Splitting（Splitwise：利用阶段分离的高效生成式 LLM 推理）

## 一、论文概览

| 项目 | 内容 |
|------|------|
| **标题** | Splitwise: Efficient Generative LLM Inference Using Phase Splitting |
| **作者** | Pratyush Patel (UW / Microsoft), Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini (Microsoft) |
| **发表** | ISCA 2024 |
| **arXiv** | [2311.18677](https://arxiv.org/abs/2311.18677) |
| **核心贡献** | 提出 **Prefill（预处理）与 Decode（解码）阶段分离（PD Disaggregation）** 的原型方案，将 LLM 推理请求的两个阶段拆分到不同机器上执行，实现相特定的资源管理与异构硬件部署 |

### 核心贡献

1. **系统性表征**：基于 Azure 生产环境 trace（coding 和 conversation 两类 LLM 服务），对 A100/H100 上的 prompt computation（预填充）和 token generation（解码）阶段的延迟、吞吐、内存和功耗特征进行了详细刻画。
2. **Splitwise 技术**：提出将模型的预填充和解码阶段拆分到不同的机器上执行，通过优化的 KV-cache 传输（逐层异步传输 + MSCCL++）实现跨机器状态迁移。
3. **集群架构设计**：设计了四种 Splitwise 变体（Splitwise-AA、Splitwise-HH、Splitwise-HA、Splitwise-HHcap），优化吞吐、成本和功耗目标。
4. **真实系统评估**：使用生产 trace + 真实硬件实验 + 集群模拟器验证，在等成本下提升吞吐 1.4×、降低 20% 成本，或在等成本等功耗下提升吞吐 2.35×。

---

## 二、技术方法详解

### 2.1 LLM 推理的两阶段特性

生成式 LLM（以 decoder-only 架构为主）的单个推理请求天然包含两个性质截然不同的计算阶段：

- **Prompt Computation Phase（预填充阶段）**：输入提示的所有 token **并行**通过模型前向传播，生成第一个输出 token。此阶段对计算能力（FLOPs）需求极高，是 **compute-bound（计算密集型）**。
- **Token Generation Phase（解码阶段）**：后续的每个输出 token **串行**生成，每个 step 仅需对最后生成的 token 做前向传播，同时需要读取之前所有上下文 token 的 KV-cache。此阶段主要是 **memory-bound（内存带宽/容量密集型）**，对计算能力并不敏感。

### 2.2 关键洞察（7 个洞察）

论文在 Section III 中通过详尽的实验表征，提炼出 7 个核心洞察：

| 序号 | 洞察 | 意义 |
|------|------|------|
| I | 不同的推理服务（coding vs. conversation）有截然不同的 prompt 和 token 长度分布 | 需要负载感知的资源分配 |
| II | 混合连续批处理在大部分时间里只处理很少的活跃 token（≤20 个 token 占 60–70% 时间） | Token 阶段 GPU 利用率低 |
| III | 绝大多数请求的 E2E 时间主要花在解码阶段 | Token 阶段是瓶颈 |
| IV | 预填充阶段应限制 batch 大小以避免性能退化；解码阶段的 batch 可以尽可能大 | 两阶段需要不同的 batching 策略 |
| V | 预填充阶段受计算能力限制，解码阶段受内存容量限制 | 两阶段对硬件的需求不同 |
| VI | 预填充阶段高效利用 GPU 功耗预算，解码阶段对功耗不敏感 | 解码阶段可以降低功耗/使用低端硬件 |
| VII | 解码阶段可以在计算能力较弱的硬件上运行，实现更好的 Perf/W 和 Perf/$ | 异构部署的动机 |

### 2.3 Splitwise 系统架构

#### 整体设计

Splitwise 维护三个机器池：

- **Prompt Pool（预填充机器池）**：负责处理 prompt computation，生成第一个 token 和 KV-cache
- **Token Pool（解码机器池）**：接收来自 prompt 机器的 KV-cache，继续生成后续 token
- **Mixed Pool（混合机器池）**：弹性伸缩，在负载波动时防止资源碎片化，同时支持预填充和解码

所有机器预先加载模型权重。调度器采用**两级调度架构**：

1. **Cluster-Level Scheduler (CLS)**：负责机器池管理和请求路由。使用 Join the Shortest Queue (JSQ) 算法，为每个请求同时分配一对（prompt 机器、token 机器）。
2. **Machine-Level Scheduler (MLS)**：管理每台机器上的 pending 队列和 batching：
   - **Prompt 机器**：使用 FCFS 调度，限制总 prompt token 数 ≤ 2048
   - **Token 机器**：使用 FCFS，尽可能大的 batch
   - **Mixed 机器**：优先处理 prompt 以满足 TTFT SLO，必要时可抢占 token

#### KV-cache 传输优化

将推理拆分到两台机器后，核心挑战是将 prompt 机器上生成的 KV-cache 高效传输到 token 机器。论文实现了两种传输策略：

1. **Naive Serialized Transfer（朴素串行传输）**：等待完整 prompt 阶段完成后再传输整个 KV-cache，会产生显著延迟。
2. **Optimized Layer-wise Transfer（逐层异步传输）**：每计算出模型某一层的 KV-cache 后，立即通过 **MSCCL++**（GPU-driven 通信库）的 zero-copy one-sided `put` 原语异步传输到 token 机器。传输与下一层的计算重叠，大幅降低可见延迟。
   - 小 prompt（<512 token）使用串行传输以降低干扰
   - 大 prompt 使用逐层传输来隐藏延迟

**传输延迟实测**：在 H100 上优化后不可重叠部分仅约 5ms，在 A100 上约 8ms，仅为 prompt 计算时间的 <7%。对端到端延迟的影响仅为 0.8%。

### 2.4 Splitwise 集群设计变体

论文提出四种变体：

| 变体 | Prompt 机器 | Token 机器 | 特性 |
|------|------------|------------|------|
| Splitwise-AA | DGX-A100 | DGX-A100 | 同构低成本方案 |
| Splitwise-HH | DGX-H100 | DGX-H100 | 高高性能同构方案 |
| Splitwise-HA | DGX-H100 | DGX-A100 | 异构方案：高性能 prompt + 低成本 decode |
| Splitwise-HHcap | DGX-H100 | DGX-H100 (power cap) | 同构 + token 机器降低功耗 50% |

### 2.5 集群资源预配框架

使用事件驱动的离散时间模拟器进行集群规模搜索，输入包括：
1. 目标集群设计（如 Splitwise-HA）
2. LLM 性能模型（piece-wise linear，MAPE < 3%）
3. prompt/token 长度分布 trace
4. SLO 目标（如表所示）
5. 约束（如吞吐量目标）
6. 优化目标（最小化成本/功耗/最大化吞吐）

**SLO 定义**（以 DGX-A100 无争用时为基准）：

| Metric | P50 | P90 | P99 |
|--------|-----|-----|-----|
| TTFT | 2× | 3× | 6× |
| TBT | 1.25× | 1.5× | 5× |
| E2E | 1.25× | 1.5× | 5× |

---

## 三、实验评估

### 3.1 KV-cache 传输开销

- 串行传输延迟随 prompt 大小线性增长
- 逐层传输后，不可重叠的固定开销约 5ms (H100) / 8ms (A100)
- Splitwise 优化传输的 E2E 开销仅 **0.8%**（对比串行传输的 3%）
- 对用户可见的第二 token 延迟影响：Splitwise 增加 **16.5%**（串行传输增加 64%）

### 3.2 等功耗吞吐优化集群

以 40 台 DGX-H100 的功耗为基准，评估各方案：

**Coding 负载**：
- Baseline-A100（70 台）：基准
- Splitwise-AA（55P, 15T）：显著提升吞吐
- Splitwise-HH（35P, 5T）：P50 TBT 优于 Baseline-H100
- Splitwise-HA（H100 prompt + A100 token）：最佳的 TTFT + E2E 折中
- Mixed Pool 在高负载下自动扩展，有效防止碎片化

**Conversation 负载**：
- Splitwise-HHcap 在所有指标上表现最佳

### 3.3 等吞吐成本/功耗优化

| 优化目标 | 最佳方案 | 收益 |
|----------|----------|------|
| **等功耗最大吞吐** | Splitwise-AA | 比 Baseline-A100 提升 **2.15×** 吞吐 |
| **等成本最大吞吐** | Splitwise-AA | 比 Baseline-H100 提升 **1.4×** 吞吐，降低 20% 成本 |
| **等吞吐最低成本** | Splitwise-AA | 比 Baseline-H100 降低 **25%** 成本 |
| **等吞吐最低功耗** | Splitwise-HHcap | 比 Baseline-H100 降低 **25%** 功耗 |
| **等成本等功耗最大吞吐** | Splitwise-HA | 比 Baseline-A100 提升 **1.18×** 吞吐，降低 10% 成本 |
| **Summary 最优** | Splitwise-HH | 同等成本功耗下提升 **2.35×** 吞吐 |

### 3.4 鲁棒性测试

- **Trace 切换**：将 coding 优化的集群运行 conversation 任务——Splitwise-AA/HH 通过 Mixed Pool 自适应调整，吞吐无损失；Splitwise-HA/HHcap 仅损失 7%
- **模型切换**：Llama-70B 替代 BLOOM-176B——所有 Splitwise 方案均优于 Baseline，Splitwise-HH/HHcap 延迟最低

---

## 四、亮点与局限

### 亮点

1. **开创性贡献**：首次系统性地提出并验证了 PD Disaggregation 的思路，是后续大量相关工作（如 DistServe、Mooncake、InferCe LLM 等）的奠基性论文
2. **完备的实验设计**：从微观表征（7 insights）→ 系统设计 → 真实硬件验证 → 大规模模拟评估，完整体现了系统研究的全链路
3. **实用性突出**：直接在生产环境 trace 上验证，考虑了 SLO、成本、功耗等多维度的实际运营约束
4. **开源贡献**：代码在 vLLM 上实现并回馈开源社区（PR #2809），同时开源了集群模拟器

### 局限

1. **集群规模受限**：模拟器评估的规模相对有限（数十台机器），未充分验证超大规模集群（数百/数千节点）下的调度扩展性
2. **"mixed pool" 的设计较粗略**：混合池中的机器本质上退化为非 Splitwise 的常规推理机器，可能导致资源碎片
3. **假设 InfiniBand 连接**：Splitwise-HA 的 A100 ↔ H100 异构连接在实际环境中可能不可行（不同代的 InfiniBand 兼容性、跨机房延迟等）
4. **未考虑 conversation 上下文复用**：论文假设每轮请求独立，未处理多轮对话中的 KV-cache 复用场景
5. **故障恢复未深入**：仅简要提及了重启或 checkpoint 方案，未设计高效的容错机制

---

## 五、个人评价

Splitwise 是 LLM 推理系统领域毫无疑问的里程碑式工作。它提出一个**极其简洁但又极具洞察力**的核心思想：既然预填充和解码阶段对硬件的需求完全不同，为什么要把它们绑在同一台机器上？

这个简单的 insight 开启了一个全新的研究方向——PD Disaggregation（预填充-解码分离），后续涌现了大量工作：

- **DistServe (OSDI'24)**：进一步将 PD 分离与 SLO 感知的调度结合
- **Mooncake (VLDB'24)**：在分离架构上引入 KV-cache 持久化存储
- **FasterTransformer + 分离部署**：工业界的跟进（如字节跳动、Anyscale 等的实践）

论文写得很清楚，实验设计严谨真实，7 个 insight 层层递进、环环相扣。唯一略显不足的是对大规模集群场景的验证深度——但这或许也给了后来者继续探索的空间。

**评分**：⭐⭐⭐⭐⭐ — 开创性工作，读完后"原来这么简单，为什么我之前没想到"的感觉。每个 LLM 推理系统研究者的必读论文。

---

## 六、关键数据速查

| 指标 | A100 | H100 | 比值 |
|------|------|------|------|
| TFLOPs (TF32) | 19.5 | 66.9 | 3.43× |
| HBM 容量 | 80GB | 80GB | 1.00× |
| HBM 带宽 | 2039 GBps | 3352 GBps | 1.64× |
| TDP 功耗 | 400W | 700W | 1.75× |
| NVLink 带宽 | 50 Gbps | 100 Gbps | 2.00× |
| InfiniBand 带宽 | 200 GBps | 400 GBps | 2.00× |
| 成本（AWS 按需） | ~$17.6/hr | ~$38/hr | 2.16× |

**Llama-70B 在 H100 上 P50 指标（无 batch）**：
- Coding: TTFT=95ms, TBT=31ms, E2E=493ms, 成本=$0.52
- Conversation: TTFT=84ms, TBT=28ms, E2E=3387ms, 成本=$3.6

---

## 相关链接

- [[LLM推理系统深度综述]]
- [[vLLM PagedAttention 论文分析]]
- [[DistServe 论文分析]]
- [arXiv 原文](https://arxiv.org/abs/2311.18677)
- [GitHub: vLLM Splitwise PR](https://github.com/vllm-project/vllm/pull/2809)
- [GitHub: Splitwise 集群模拟器](https://github.com/Mutinifni/splitwise-sim)
- [Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)
