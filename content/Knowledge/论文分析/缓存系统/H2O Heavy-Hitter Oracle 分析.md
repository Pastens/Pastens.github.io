---
tags:
- 论文分析
- kv-cache
- eviction
- heavy-hitter
- attention
- neurips-2023
source: https://arxiv.org/abs/2306.14048
github: https://github.com/FMInference/H2O
created: 2026-05-15
rating: ⭐⭐⭐⭐⭐
---

# H2O (Heavy-Hitter Oracle)：KV Cache 逐出的奠基之作

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models |
| **arXiv** | 2306.14048 |
| **GitHub** | https://github.com/FMInference/H2O |
| **发表** | NeurIPS 2023 |
| **引用** | 800+（截至 2026） |
| **标签** | KV Cache 逐出, Attention 分析, 推理优化 |

---

## 二、核心贡献

H2O 提出了一个简单但极具影响力的发现——**Heavy Hitter（重击者）现象**，并基于此设计了 KV Cache 逐出策略：

### 2.1 Heavy Hitter 现象

> 在注意力计算中，**极少数 token（约占 20%）贡献了绝大部分（>90%）的注意力分数**，这些 token 被称为 Heavy Hitters。

```
注意力分数分布示例（以 LLaMA-7B 为例）:
┌────────────────────────────────────────┐
│ Top-20% token 的注意力分数之和: ~90%   │  ← Heavy Hitters
│ Bottom-80% token 的注意力分数之和: ~10% │  ← 可逐出部分
└────────────────────────────────────────┘
```

### 2.2 H2O 逐出策略

| 组件 | 说明 |
|------|------|
| **保留集** | 始终保留 Heavy Hitter token（累积注意力分数最高的 token） |
| **逐出集** | 非 Heavy Hitter token 按需逐出 |
| **策略名称** | Heavy Hitter Oracle — 将"未来会获得高注意力"的 token 提前保留 |
| **实现方式** | 在推理过程中持续跟踪每个 token 的累积注意力分数，保留 Top-k |

### 2.3 性能效果

| 指标 | 效果 |
|------|------|
| **KV Cache 压缩比** | 5×~10× 压缩（保留 20% token） |
| **精度损失** | < 1% 困惑度（perplexity）损失 |
| **推理加速** | 最高 2× 端到端加速（因减少 KV Cache I/O） |
| **模型兼容** | OPT, LLaMA, GPT-NeoX 等主流架构 |

---

## 三、技术方法详解

### 3.1 Heavy Hitter 的发现方法

H2O 通过大量实验发现了注意力中的幂律分布特征：

```python
# 伪代码：Heavy Hitter 识别
def identify_heavy_hitters(attention_scores, head_idx, token_idx):
    # attention_scores: [num_heads, num_tokens]
    # 对每个 head 计算 token 的累积注意力权重
    cum_attn = cumulative_attention(attention_scores)
    # 按累积注意力排序，取 Top-k
    top_k_indices = argsort(-cum_attn)[:k]
    return top_k_indices
```

**关键发现**：
1. Heavy Hitter 模式在**所有层和所有 head** 中都存在
2. 不同 head 的 Heavy Hitter 集合**高度重叠**（即相同的 token 在所有 head 中都很重要）
3. Heavy Hitter 模式在**推理早期就稳定形成**（前几步就能确定哪些 token 是 Heavy Hitter）

### 3.2 逐出算法的核心逻辑

H2O 的逐出策略可以形式化为：

```
缓存状态: Cache = {token_1, token_2, ..., token_n}
每个 token 的累积注意力分数: score(token_i) = sum(attn_scores over all steps)

当 Cache 满时:
  1. 找出 score 最低的 token: victim = argmin(score)
  2. 逐出 victim
  3. 新生成的 token 始终加入 Cache（即使分数低，因为可能成为未来的 Heavy Hitter）
```

> 注意：新 token **总是被保留**（即使初始注意力分数很低），因为它们尚未获得累积注意力。

### 3.3 与 Baseline 的对比

| 策略 | 压缩比 4× (PPL) | 压缩比 8× (PPL) | 压缩比 16× (PPL) |
|------|:---:|:---:|:---:|
| **H2O** | **最低 PPL** | **最低 PPL** | **最低 PPL** |
| Random | 高 2-5% | 高 10-15% | 退化 |
| StreamingLLM | 高 1-2% | 高 5-8% | 退化 |
| 完整缓存 | 基准 | 基准 | 基准 |

---

## 四、个人评价

### 亮点

- **开创性发现**：Heavy Hitter 现象是 KV Cache 逐出的理论基础，启发了后续几乎所有工作（ScissorHands, AhaKV, SnapKV, KIVI 等）
- **极简实现**：仅需维护一个累积注意力分数列表，计算开销极小
- **效果显著**：5× 压缩下几乎无损，10× 压缩下仍保持可用精度
- **影响力强**：NeurIPS 2023 论文，GitHub 2K+ stars，被大量后续工作引用

### 局限

- **注意力分数追踪开销**：需要在推理过程中计算和累积注意力分数，带来少量额外开销
- **静态 Top-k 保留**：保留比例是超参数，无法根据序列内容动态调整
- **未考虑跨层相关性**：每层独立做逐出决策，未利用层间重要性传递（后来被 ScissorHands 改进）
- **对长上下文退化**：超长序列（>32K）中 Heavy Hitter 比例可能需要调整

### 后续影响

H2O 开启了 KV Cache 逐出的研究方向，直接催生了：
- [[ScissorHands 分析]] — 引入 Persistence of Importance 假设，利用跨层相关性
- [[AhaKV 分析]] — 自适应 holistic attention-guided 逐出策略
- SnapKV — 基于观察窗口的逐出
- Keyformer — Top-k 缓存的自适应版本

## 相关链接
- [[ScissorHands 分析]]
- [[AhaKV 分析]]
- [[缓存系统性能建模洞察分析]] — KV Cache 研究全景图谱
