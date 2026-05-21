---
tags:
- 论文分析
- llm-training
- memory-optimization
- zero
source: https://github.com/microsoft/DeepSpeed
arxiv: 1910.02054
authors: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
institutions: Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models |
| **arXiv** | 1910.02054 |
| **机构** | Microsoft |
| **代码** | https://github.com/microsoft/DeepSpeed (⭐37K+) |
| **影响力** | DeepSpeed 框架基石，定义模型训练显存优化的里程碑 |

### 核心贡献

1. **ZeRO-DP 三阶段分片**：将模型状态（参数、梯度、优化器状态）分片到各 GPU，避免冗余
2. **ZeRO-R 内存冗余消除**：将激活检查点、临时缓冲区等运行时内存也优化
3. **突破单 GPU 显存限制**：支持训练万亿参数模型
4. **与 DP 兼容**：保持数据并行接口和计算模式，无需修改模型代码

---

## 二、技术方法详解

### 2.1 显存消耗分析

传统 DP 中，每 GPU 存储完整的模型副本 → 显存冗余严重。

**模型状态分类（SC 维度）**：
- **P** (Parameters): 参数，每个 GPU 都存一份完整的 (Ψ 个元素)
- **O** (Optimizer States): 优化器状态（如 Adam 的 momentum+variance），2× 参数大小
- **G** (Gradients): 梯度，每 GPU 一份完整副本
- **R** (Residual Memory): 激活值、临时缓冲区、碎片空间

### 2.2 ZeRO-DP 三阶段

| 阶段 | 分片内容 | GPU 显存 (以 Ψ=100B 为例) | 通信量 |
|------|---------|------------------------|--------|
| Stage 1 (P_os) | 优化器状态分片 | 16Ψ → 4Ψ + 16Ψ/N_d | 相同 |
| Stage 2 (P_g) | 梯度+优化器状态分片 | 16Ψ → 2Ψ + 2Ψ/N_d | ~1.5× |
| Stage 3 (P_p) | 参数+梯度+优化器状态全分片 | 16Ψ → 16Ψ/N_d | ~1.5× |

- **Stage 1**：优化器状态跨 GPU 分片，每 GPU 只需维护 1/N_d 的优化器
- **Stage 2**：梯度也分片，AllGather 后按需获取
- **Stage 3**（ZeRO-3）：参数也分片，前向/反向时 broadcast，用完即丢

### 2.3 ZeRO-R 优化

- **Activation Checkpointing**: 选择性存储激活值（约 1.5× 计算换 5× 显存）
- **Constant Size Buffer**: 使用固定大小缓冲区避免动态分配
- **Memory Defragmentation**: 分片后自动整理显存碎片

### 2.4 通信优化

- ZeRO-DP 的通信量只有标准 DP 的 ~1.5×，但显存节约 N_d 倍
- 计算-通信 overlap：前向/反向中嵌入分片后的通信操作

---

## 三、实验评估

| 模型 | GPU | 方法 | 吞吐 (TFLOPs/GPU) | 最大模型 |
|------|-----|------|-------------------|---------|
| GPT-2 1.5B | 64 V100 | DP (baseline) | 19.4 | OOM |
| GPT-2 1.5B | 64 V100 | ZeRO-2 | **27.8** | 17.8B |
| GPT-100B | 400 V100 | ZeRO-2 | 15.1 | 170B |
| 1 Trillion | 32000 V100 | ZeRO-2 | 15+ | 1T |

---

## 四、亮点与局限

### 亮点
- **显存效率突破**：64 V100 上从 1.5B → 17.8B（10×+ 模型规模提升）
- **零代码修改**：DeepSpeed ZeRO 只需几行配置，不要求修改模型
- **生态影响**：FSDP 是 PyTorch 对 ZeRO-3 的官方实现

### 局限
- **Stage 3 通信开销**：参数 AllGather 增加 1.5× 通信量，在小批次时影响更大
- **对 CPU/NVMe 卸载的延迟敏感**：低速存储导致训练吞吐下降

---

## 五、个人评价

ZeRO 的工作在 DP 框架内优雅地解决了显存瓶颈，是 LLM 训练领域引用最高的论文之一。三阶段分片思想平衡了显存-通信-吞吐，成为所有现代 LLM 训练框架的标配。唯一遗憾是尚未完全解决 AllGather 通信在超大规模集群中的开销问题。

## 相关链接
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
- [[Knowledge/论文分析/训练系统/ZeRO-Offload 技术分析]]
- [[Knowledge/论文分析/训练系统/ZeRO-Infinity 技术分析]]
- [[Knowledge/论文分析/训练系统/FSDP 技术分析]]
