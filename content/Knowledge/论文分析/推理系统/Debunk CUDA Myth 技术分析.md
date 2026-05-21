---
tags:
- 论文分析
- llm-inference
- hardware-evaluation
- cuda
- gaudi
- npu
- isca-2025
arxiv: 2501.0021
authors:
- Yunjae Lee
- Juntaek Lim
- Jehyeon Bang
- Eunyeong Cho
- Huijong Jeong
- Taesu Kim
- Hyungjun Kim
- Joonhyung Lee
- Jinseop Im
- Ranggi Hwang
- Se Jung Kwon
- Dongsoo Lee
- Minsoo Rhu
institutions:
- KAIST
- NAVER Cloud
- SqueezeBits
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# Debunking the CUDA Myth: 打破CUDA神话——基于GPU的AI系统公平对比：Intel Gaudi NPU与NVIDIA A100在AI模型推理中的性能与可编程性评估

## 一、论文概览

| 项目 | 内容 |
|------|------|
| **论文标题** | Debunking the CUDA Myth Towards GPU-based AI Systems: Evaluation of the Performance and Programmability of Intel's Gaudi NPU for AI Model Serving |
| **会议/期刊** | ISCA-52 2025 (第52届国际计算机体系结构研讨会) |
| **作者** | Yunjae Lee, Juntaek Lim (共同一作, KAIST), 等 |
| **机构** | KAIST (韩国科学技术院), NAVER Cloud, SqueezeBits |
| **通讯作者** | Minsoo Rhu (KAIST, mrhu@kaist.ac.kr) |
| **arXiv ID** | 2501.00210 |
| **核心主题** | Intel Gaudi-2 NPU vs NVIDIA A100 GPU 在LLM推理中的公平对比——性能与可编程性 |

### 背景与动机

本文站在2025年的节点，呼应了2010年Intel发表的经典论文"Debunking the 100X GPU vs. CPU Myth"（ISCA 2010）——那篇文章指出经过合理优化后，GPU相比CPU的加速比远非宣称的100倍，实际上仅有2.5倍。十五年后的今天，角色发生了戏剧性的对调：**Intel成为了挑战者（underdog），NVIDIA成为了统治地位的"守成者"**。

本文的核心问题是：**Intel Gaudi NPU能否在AI推理市场对NVIDIA GPU形成真正的竞争威胁？CUDA是否真的是不可逾越的"护城河"？**

### 硬件规格对比

| 规格 | NVIDIA A100 | Intel Gaudi-2 | 比率 |
|------|-------------|---------------|------|
| **制程** | TSMC 7nm | TSMC 7nm | — |
| **BF16矩阵算力** | 312 TFLOPS (Tensor Cores) | 432 TFLOPS (MME) | 1.4× |
| **BF16向量算力** | 39 TFLOPS (SIMD Cores) | 11 TFLOPS (TPC) | 0.3× |
| **HBM类型** | HBM2E | HBM2E | — |
| **HBM容量** | 80 GB | 96 GB | 1.2× |
| **HBM带宽** | 2 TB/sec | 2.46 TB/sec | 1.2× |
| **片上SRAM** | 40 MB (L2 Cache) | 48 MB (Shared Memory) | 1.2× |
| **芯片间通信** | 600 GB/s 双向 (NVLink+NVSwitch) | 600 GB/s 双向 (RoCE P2P) | 1.0× (但架构差异巨大) |
| **TDP** | 400 W | 600 W | 1.5× |

---

## 二、技术方法详解

### 2.1 Gaudi硬件架构

Gaudi采用**异构计算**范式，集成两种核心计算单元：

1. **MME (Matrix Multiplication Engines)**：大尺寸输出驻留（output-stationary）脉动阵列，基础配置为两个独立的256×256 MAC单元。**关键创新**：MME可动态重配置——运行时可将两个MME合并为512×256、1024×128等不同形状的单一MAC阵列，以最佳匹配目标GEMM的(M, K, N)形状。

2. **TPC (Tensor Processing Cores)**：完全可编程的VLIW处理器，每个TPC包含标量处理单元、向量处理单元（2048-bit SIMD）和加载/存储单元。Gaudi-2有24个TPC。每个TPC有1KB标量局部存储和80KB向量局部存储。

3. **片上共享内存**：48 MB SRAM作为graph compiler的scatchpad，用于MME、TPC和DMA引擎之间的数据流转。

4. **通信架构**：HLS-Gaudi-2服务器包含8颗Gaudi-2芯片，每颗有24个100GbE RoCEv2端口，其中21个用于芯片间直连（P2P）。**每对Gaudi-2芯片之间通过3条100GbE链路连接**，因此有效带宽随参与通信的芯片数量线性变化。

### 2.2 Gaudi软件架构

**编程模型**：
- GPU: SIMT (Single Instruction Multiple Thread) 模型，依赖大规模寄存器文件和动态分支发散解决
- Gaudi: **单线程编程模型**，优化数据级并行而非线程级并行

**关键差异**：
- CUDA中，开发者可以在一个低层级kernel中同时使用Tensor Cores和SIMD Cores（通过WMMA API）
- Gaudi SDK限制开发者只能直接控制TPC，**MME只能通过PyTorch层级访问**，不能直接在TPC-C kernel中编程

**Graph Compiler**：
- Intel Gaudi SDK包含一个graph compiler，将AI模型转换为Gaudi NPU可执行格式
- 应用高层优化如算子融合（基于MLIR的operation fuser）
- 自动确定MME的最佳配置并实现MME和TPC之间的流水线执行
- **黑盒性质**：开发者无法控制graph compiler的优化过程

**TPC编程最佳实践**：
1. 数据访问粒度必须对齐256字节（全局内存最小访问粒度）
2. 手动展开for循环以利用指令级和内存级并行（因为TPC指令有4周期延迟）

---

## 三、实验评估

### 3.1 微基准测试方法论

论文设计了一套微基准测试套件，覆盖三类原语操作：

| 类别 | 操作 | Gaudi-2实现 | A100实现 |
|------|------|-------------|----------|
| Compute | GEMM | PyTorch API | PyTorch API |
| Compute | 非GEMM (ADD/SCALE/TRIAD) | TPC-C | CUDA |
| Memory | 向量gather-scatter | TPC-C | CUDA |
| Communication | 集合通信 | Intel HCCL | NVIDIA NCCL |

**端到端工作负载**：
- RecSys: DLRM-DCNv2的两个配置（RM1计算密集型、RM2内存密集型）
- LLM: Llama-3.1-8B-Instruct 和 Llama-3.1-70B-Instruct

---

### 3.2 核心发现

#### 发现#1: GEMM操作——Gaudi-2全面领先

- Gaudi-2在所有(M,K,N)形状上均超越A100，当M=K=N=8192时达到429 TFLOPS（峰值利用率的99.3%）
- **平均计算利用率高出4.5%**（最大32%）
- 原因：Gaudi MME的动态重配置能力使其脉动阵列能灵活适应不同形状的GEMM，相比非可配置的脉动阵列，利用率提升最高达15%

> **关键洞察**：Gaudi-2不仅在绝对算力上（432 vs 312 TFLOPS）占优，在**计算效率**（利用率）上也更胜一筹。

#### 发现#2: 非GEMM向量操作——Gaudi-2落后但效率相当

- 由于向量算力差距（11 vs 39 TFLOPS），Gaudi-2绝对吞吐量落后
- **但在计算利用率上两者相当**：ADD/SCALE均达到峰值的50%，TRIAD达到99%（A100类似）
- TPC编程中，256字节对齐和循环展开至关重要——SCALE受益最大（从13 GFLOPS提升至530 GFLOPS）

> **关键洞察**：绝对算力差距可以通过上层框架的优化来弥补，Gaudi的TPC设计效率并不差。

#### 发现#3: 内存操作——细粒度访问是Gaudi的软肋

- 向量大小 ≥256字节时，Gaudi-2达到平均64%内存带宽利用率（A100为72%），竞争力不错
- 向量大小 <256字节时，Gaudi-2骤降至**平均15%**，A100保持36%（**2.4倍差距**）
- 原因：Gaudi全局内存最小访问粒度为**256字节**，而NVIDIA的cache line大小为32字节或采用32字节sector缓存

> **关键洞察**：256字节最小访问粒度是Gaudi架构的根本性限制，对细粒度的embedding lookup影响显著。

#### 发现#4: 集合通信——系统级架构差异而非处理器本身

- 8设备全部参与时，Gaudi-2在6种集合通信中有5种优于A100
- **但随着设备数减少，Gaudi-2的带宽利用率几乎线性下降**（因为P2P直连架构）
- A100的NVSwitch确保无论多少设备参与通信，都能维持全带宽
- Gaudi-2在2设备时通信带宽仅为峰值的1/8

> **关键洞察**：这不是Gaudi芯片本身的问题，而是**系统级网络架构**的差异——Intel需要引入类似NVSwitch的全互联交换机。

---

### 3.3 端到端应用性能

#### RecSys (推荐系统)

| 指标 | Gaudi-2 vs A100 |
|------|----------------|
| 平均性能 | **慢22%** (RM1), **慢18%** (RM2) |
| 功耗 | 平均高12% |
| 能效 | **低28%** |
| 优势场景 | 宽embedding向量+大批量 (最高1.36×加速) |
| 劣势场景 | 向量<256字节 + 小批量 (最高70%性能损失) |

#### LLM (大语言模型)

| 场景 | 性能加速比 | 能效加速比 |
|------|-----------|-----------|
| 单设备 (8B) | **1.47×** | **1.48×** |
| 2设备 TP=2 (70B) | 1.29× | 1.48× |
| 4设备 TP=4 (70B) | 1.32× | 1.51× |
| 8设备 TP=8 (70B) | 1.35× | 1.56× |

- **功耗表现惊喜**：尽管TDP高出50%（600W vs 400W），Gaudi-2在实际LLM推理中平均功耗仅比A100高1%（小批量时甚至更低，因DVFS/功率门控）
- 单设备LLM推理能效平均提升**48%**，多设备提升**52%**

> **关键洞察**：LLM推理以矩阵乘法为主，Gaudi-2的MME优势得以充分发挥。RecSys以向量gather和小MLP层为主，Gaudi-2的256字节对齐限制成为瓶颈。

---

### 3.4 可编程性案例分析

#### Case 1: DLRM Embedding Lookup (低层级TPC-C优化)

**问题**：Intel Gaudi SDK默认的embedding层仅达到GPU优化版本的37%

**优化策略**：
1. **SingleTable**: 每个table独立处理，循环展开4次，利用TPC局部存储
2. **BatchedTable**: 将多个table合并为一个大table（类似FBGEMM），通过偏移量索引区分不同table——显著提升小批量下的内存带宽利用率

**结果**：
| 指标 | Gaudi BatchedTable | A100 FBGEMM |
|------|-------------------|-------------|
| 平均带宽利用率 | 34.2%（峰值70.5%） | 38.7%（峰值81.8%） |
| 大向量(≥256B)吞吐量 | A100的**95%** | 基线 |
| 小向量(<256B)吞吐量 | A100的**47%** | 基线 |

> **关键洞察**：TPC-C提供了足够的低层级编程灵活性来优化性能，但硬件架构限制（256字节粒度）在细粒度操作上仍然存在。

#### Case 2: vLLM PagedAttention (高层级PyTorch优化)

**问题**：Gaudi SDK不支持在低层级TPC-C kernel中编程MME，PagedAttention的实现受限于PyTorch层级和graph compiler

**基线实现 (vLLM_base)**：
- 使用2D BlockTable，因零填充导致冗余的KV cache block gather
- 将所有KV cache block gather到连续内存后再执行FusedSDPA
- 阻止了graph compiler在MME和TPC之间进行有效的流水线执行

**优化实现 (vLLM_opt)**：
- 用1D BlockList替代2D BlockTable，消除零填充引起的冗余gather
- 调整query tensor形状以匹配BlockList格式
- 使graph compiler能更有效将TPC-based gather和MME-based GEMM切分为独立子操作并流水执行

**结果**：
- PagedAttention吞吐量提升：**平均7.4×**（零填充比例0%时），最多55.7×（零填充90%时）
- 但绝对值仍仅为A100的**45%**
- **端到端LLM性能与A100相当**（Amdahl定律：MLP层GEMM优势弥补了Attention层的劣势）
- SLO指标（TTFT和TPOT）与A100相似

> **关键洞察**：尽管graph compiler的黑盒性质带来限制，但通过PyTorch层级的合理编程，graph compiler仍能有效捕获并行性。**Gaudi在Attentioin层落后2.2倍，但GEMM层的优势使整体性能持平**。

---

## 四、亮点与局限

### 亮点

1. **首次全面对比**：本文是第一个从性能和可编程性两个维度系统对比Gaudi NPU与NVIDIA GPU的学术工作
2. **公平对比方法论**：使用同一制程节点（TSMC 7nm）和内存子系统（HBM2E），对比选择合理
3. **微基准+端到端+可编程性三维评估**：方法论完整，层次分明
4. **逆向工程MME重配置**：通过Intel Gaudi Profiler逆向推断了MME的动态重配置策略（图7），对理解Gaudi架构有重要价值
5. **开源实践指导**：TPC-C的编程最佳实践（256字节对齐、循环展开）对开发者有直接参考价值
6. **有争议性的核心论点**：提出"CUDA本身并非不可逾越的护城河，NVIDIA的真正优势在于丰富的软件生态"

### 局限

1. **对比对象选择**：选择A100而非H100/B200——虽然A100和Gaudi-2同为7nm/HBM2E合理，但截至2025年H100已是主流，结论的时效性受限。作者在脚注中提及Gaudi-3与Gaudi-2架构几乎相同但算力更高
2. **Gaudi SDK版本**：基于v1.18.0（PyTorch 2.4），后续版本可能有显著改进
3. **不具备MME低层级编程能力**：论文的多项优化受限于graph compiler的黑盒性质，这是Gaudi架构的根本性限制
4. **RecSys仅限单设备**：Intel Gaudi SDK尚不支持多设备RecSys（TorchRec），限制了对比范围
5. **未涉及AI训练**：对比仅限于推理，未覆盖训练场景
6. **缺乏与AMD等其他竞争平台的对比**（已在未来工作中提及）

---

## 五、个人评价

### 核心价值

这是一篇**极具时代意义**的体系结构论文。它巧妙呼应了15年前Intel的经典论文，在角色反转的背景下探讨了一个关键问题：**NVIDIA的CUDA生态到底有多不可替代？**

### 主要论断分析

论文的核心论断——**"CUDA本身可能不是护城河，NVIDIA的真正优势在于软件生态"**——虽然具有争议性，但分析较为扎实。作者的逻辑：

1. 大多数AI开发者使用PyTorch等高层框架，不直接编写CUDA
2. 只要NPU厂商有效支持这些高层框架并提供优化的后端库，CUDA的优势就会消解
3. Gaudi-2在LLM推理中确实展现了竞争力（能效优势高达56%）

### 需要审慎看待的方面

1. **外部效度**：Gaudi-3的可用性和成熟度尚需验证，vLLM on Gaudi的生产环境表现可能与实验室结果有差距
2. **"软件生态" vs "CUDA"的边界模糊**：作者将TensorRT-LLM、cuBLAS等归入"生态"而非"CUDA本身"，但这种区分在实践中的意义有限——因为这些库正是CUDA生态的组成部分
3. **MME不可编程的限制是根本性的**：这意味着一类需要MME参与的自定义算子（如FlashAttention变体）在Gaudi上的实现会受到严重限制
4. **256字节对齐的硬件限制难以通过软件完全弥补**

### 对行业的启示

- 对于AI芯片创业公司：**高层框架兼容性比低层编程接口的灵活性更重要**——只要接入PyTorch/TensorFlow生态并优化后端库，就有机会在特定场景下挑战NVIDIA
- 对于Intel/Gaudi：需要解决 (1) 细粒度内存访问效率，(2) 芯片间全互联网络，(3) MME编程接口开放化 三大短板
- 对于学术界：本文的评估方法论（微基准+端到端+可编程性案例）可作为NPU/GPU对比研究的参照标准

### 评分：⭐⭐⭐⭐ (4/5)

- **加分项**：方法论严谨、对比公平、具有时代意义、可编程性分析有深度
- **扣分项**：A100而非H100的对比选择、Gaudi SDK版本限制、未涉及训练

---

## 相关链接

- [[Knowledge/论文分析/推理系统/LLM推理系统深度综述]]
- GPU vs NPU架构对比分析
- PagedAttention原理与实现
- vLLM推理引擎深度分析
- 推荐系统深度学习模型综述
