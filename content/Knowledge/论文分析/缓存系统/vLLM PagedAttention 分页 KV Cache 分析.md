---
tags:
- 论文分析
- kv-cache
- memory-management
- paging
- llm-serving
- oss
arxiv: '2309.06180'
source: https://arxiv.org/abs/2309.06180
github: https://github.com/vllm-project/vllm
authors:
- Woosuk Kwon
- Zhuohan Li
- Siyuan Zhuang
- Ying Sheng
- Lianmin Zheng
- Cody Hao Yu
- Joseph E. Gonzalez
- Hao Zhang
- Ion Stoica
institutions:
- UC Berkeley
- Stanford University
created: 2026-05-15
rating: ⭐⭐⭐⭐⭐
permalink: vllm-pagedattention-kv-cache
---

# vLLM / PagedAttention — 分页 KV Cache 管理与推理系统

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| **arXiv** | 2309.06180 |
| **GitHub** | https://github.com/vllm-project/vllm（⭐60K+） |
| **发表** | SOSP 2023 |
| **核心贡献** | 将 KV Cache 分页化（类似 OS 虚拟内存），消除内部碎片，支持共享 prefix |
| **标签** | KV Cache 管理, 内存分页, LLM 推理系统, 前缀共享 |

---

## 二、核心贡献

### 2.1 问题：KV Cache 的内存碎片化

传统 LLM 推理系统中，KV Cache 被预分配为连续内存区域，导致两大问题：

| 问题 | 描述 | 影响 |
|------|------|------|
| **内部碎片** | 预先分配最大可能序列长度的内存，实际生成远短于此 | 浪费 60-80% 的 GPU 显存 |
| **外部碎片** | 不同请求的连续内存块无法灵活复用 | 内存利用率低，限制并发度 |

### 2.2 PagedAttention 核心思想

> 将 KV Cache 划分为**固定大小的 Page（块）**，类似 OS 虚拟内存中的分页机制。每个请求的 KV Cache 以非连续方式存储在多个 Page 中，通过**页表（Block Table）** 维护映射关系。

```
传统方式（连续分配）:
|---- Request A KV Cache (预分配最大长度) ----|  ← 内部碎片
              |---- Request B KV Cache ----|      ← 外部碎片

PagedAttention（分页方式）:
| Page 0 | Page 1 | Page 3 | ← Request A (离散存储)
| Page 2 | Page 4 | ← Request B (灵活复用)
  ↑ Block Table 维护虚拟地址 → 物理页映射
```

### 2.3 创新设计

| 组件 | 说明 |
|------|------|
| **KV Block Manager** | 全局物理页分配器，类似 OS 的内存管理单元 |
| **Block Table** | 每个请求维护一张页表，映射逻辑 Token 位置到物理 Page |
| **Copy-on-Write** | Prefix 共享时使用写时复制机制，避免数据拷贝 |
| **Prefix Caching** | 自动缓存公共前缀（如 System Prompt）的 KV Page，跨请求复用 |

---

## 三、技术方法详解

### 3.1 分页机制

PagedAttention 在 Attention 计算层面修改了 kernel，使其支持非连续内存访问：

```python
# PagedAttention 的注意力计算（简化）
def paged_attention(query, block_table, page_size):
    """
    query: [num_heads, head_dim]
    block_table: [num_blocks] 每个元素指向一个物理页
    page_size: 每个页存储的 token 数
    """
    output = zeros(num_heads, head_dim)
    for block_idx in range(num_blocks):
        phys_page = block_table[block_idx]
        # 从物理页加载 KV Cache（非连续访问）
        key_block = load_phys_page(phys_page, "key")
        value_block = load_phys_page(phys_page, "value")
        # 计算注意力
        attn_scores = query @ key_block.T
        attn_weights = softmax(attn_scores)
        output += attn_weights @ value_block
    return output
```

关键参数：

| 参数 | 默认值 | 影响 |
|------|--------|------|
| **Page Size** | 16 tokens | 大页 → 减少页表大小但增加内部碎片 |
| **Max Num Blocks** | 由 GPU HBM 决定 | 限制最大并发请求数 |
| **GPU Memory Fraction** | 0.9 | 控制 KV Cache 可用的 HBM 比例 |

### 3.2 前缀共享（Prefix Caching）

vLLM 自动检测请求间的公共前缀（如 System Prompt、Few-shot Examples），共享其 KV Page：

```
请求 1: "请将以下内容翻译成英文：Hello World"
请求 2: "请将以下内容翻译成英文：你好世界"
                          ↓
共享前缀: "请将以下内容翻译成英文：" → KV Page 被两个请求共享
不同部分: "Hello World" / "你好世界"  → 各自独立的 Page
```

**Copy-on-Write 机制**：当某个请求需要修改共享 Page 时（如生成新 token），先复制再修改。

### 3.3 调度与批处理

vLLM 实现了**连续批处理（Continuous Batching）**：

- 在每个 decode 步骤，动态选择可用请求组成批次
- 新请求可随时加入（如果 GPU 内存允许）
- 完成的请求立即释放 KV Page 供其他请求使用

---

## 四、性能模型分析

### 4.1 经验性性能特征

vLLM 虽无显式端到端性能模型，但其设计隐含以下性能关系：

| 关系 | 描述 |
|------|------|
| **吞吐量 vs Page Size** | Page Size 过小 → 页表过大，管理开销增加；Page Size 过大 → 内部碎片增多，有效内存减少 |
| **TTFT vs 前缀共享** | 前缀命中 → 跳过预填充 → TTFT 降低 30-70% |
| **TPOT vs 物理页分布** | 非连续页的 TLB 类开销在 Page Size > 8 时趋于稳定 |

### 4.2 内存效率估算

```
有效内存利用率 ≈ (∑ sequence_length_i) / (num_blocks × page_size)
                  ↓
实际使用 tokens / 分配的 tokens

典型值: 无需预先分配 → 利用率从 20-40% 提升至 90%+
```

### 4.3 局限

vLLM 本身**不提供** KV Cache 触发的端到端延迟预测模型。它作为一个实际系统，其性能通过真实运行测量而非建模预测。

---

## 五、局限与未来方向

### 局限

1. **Page Size 调优困难**：最佳 Page Size 依赖工作负载和硬件，缺乏自动化调优
2. **连续批处理复杂度**：调度决策（如何选择请求批次）是 NP-hard 问题，当前使用启发式策略
3. **无显式卸载支持**：所有 KV Cache 必须在 GPU HBM 中，不支持 CPU DRAM 或 SSD 卸载
4. **共享粒度受限**：前缀共享仅适用于完全相同的 token 序列，不支持近似匹配

### 未来方向

- **自适应 Page Size**：根据序列长度分布动态调整 Page Size
- **异构卸载集成**：将 FlexGen 或 LLM in a Flash 的卸载策略集成到 vLLM 中
- **ML 驱动的调度**：使用强化学习优化连续批处理的请求选择

---

## 六、个人评价

PagedAttention 是 KV Cache 管理领域的里程碑式工作。它将 OS 虚拟内存的分页思想引入 LLM 推理，从根本上解决了 KV Cache 的内存碎片问题。vLLM 系统也因此成为目前最广泛使用的开源 LLM 推理框架之一。

**影响力**：几乎后续所有 KV Cache 研究工作（卸载、量化、共享、逐出）都将 vLLM 作为基准系统或其研究成果的集成目标。

**适用场景**：任何需要高效 GPU 内存管理的 LLM 推理部署，尤其是高并发场景。

## 相关链接

- [[FlexGen IO 感知 KV Cache 卸载分析]] — 将 KV Cache 卸载到 CPU DRAM + SSD
- [[LLM in a Flash KV Cache Flash 存储分析]] — Flash 存储模型权重和 KV Cache
- [[缓存系统性能建模洞察分析]] — KV Cache 研究全景图谱中的位置
