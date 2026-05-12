---
tags:
  - 论文分析
  - llm-training
  - parallelism
  - pytorch
arxiv: 2304.11277
authors: Yanli Zhao, Andrew Gu, Rohan Varma, Min Xu, Sayed Hadi Hashemi, et al.
institutions: Meta (PyTorch Team)
created: 2026-05-11
rating: ⭐⭐⭐⭐⭐
---

# FSDP: Experiences on Scaling Fully Sharded Data Parallel

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel |
| **arXiv** | 2304.11277 |
| **机构** | Meta (PyTorch Team) |
| **代码** | PyTorch Distributed (内置) |

### 核心贡献

1. **PyTorch 原生 ZeRO-3**：FSDP 将 ZeRO-3 思想集成到 PyTorch 生态
2. **混合分片策略**：支持 full shard / shard-grad-op / no-shard 三种模式
3. **反向预取（Backward Prefetch）**：反向中预取下一层参数，隐藏通信延迟
4. **生产级验证**：在 Meta 的 512+ GPU 集群上验证

---

## 二、技术方法

### 分片策略

| 模式 | 分片对象 | 适用场景 |
|------|---------|---------|
| FULL_SHARD | 参数+梯度+优化器 | 大模型，高显存瓶颈 |
| SHARD_GRAD_OP | 梯度+优化器 | 中等模型，适度通信 |
| NO_SHARD | 不切片（等同 DDP） | 小模型，最优化通信 |

### 反向预取

在反向传播中，`fsdp_pre_backward_hook` 提前预取下一层的参数 AllGather，与当前层的计算 overlap。相比 ZeRO-3 的实现，反向预取约减少 20% 的通信等待时间。

---

## 三、实验

在 Meta 的生产集群上验证：
- 174B 参数模型在 256 A100 上训练
- 相比 DDP 把模型容量从 41B 提升到 174B
- 与 ZeRO-3 性能相当，零额外代码依赖

---

## 四、个人评价

FSDP 的重要意义在于将 ZeRO-3 的显存优化带入 PyTorch 主流生态，向所有 PyTorch 用户开放了大规模训练能力。它在训练小模型时引入的开销比 ZeRO 更可控（通过混合分片策略），但在极致大模型场景下，Megatron-LM 的 TP+PP 仍不可替代。

## 相关链接
- [[ZeRO 技术分析]]
- [[LLM训练系统深度综述]]
