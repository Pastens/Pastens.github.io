---
tags:
  - 论文分析
  - kv-cache
  - cache-sharing
  - multi-tenant
  - parameter-remapping
  - prefix-sharing
arxiv: "2507.11507"
created: 2026-05-15
rating: ⭐⭐⭐⭐
---

# Oneiros — 多租户场景下通过参数重映射优化 KV Cache 共享

## 一、论文基本信息

| 属性 | 内容 |
|------|------|
| **标题** | Oneiros: KV Cache Sharing via Parameter Remapping for Multi-Tenant LLM Serving |
| **arXiv** | 2507.11507 |
| **核心贡献** | 多租户场景下通过参数重映射优化 KV Cache 共享 |
| **关键词** | KV Cache 共享、参数重映射、多租户、前缀共享、内存优化 |

## 二、核心思想

### 问题

GORGO/KVShare 解决的是**相同前缀**的跨用户共享，但如果用户使用了**相似但不同**的参数化 prompt 呢？

例如：
- 用户 A 的 prompt: `"Translate to French: {text_1}"`
- 用户 B 的 prompt: `"Translate to French: {text_2}"`

传统 prefix caching 认为这两个 prompt 的前缀只有 `"Translate to French: "` 部分共享，但 Oneiros 进一步观察到：**即使不同参数值（text_1 vs text_2）经过模型编码后，它们与共享前缀的交互模式也是相似的**。

Oneiros 提出：通过**参数重映射（parameter remapping）**，让不同用户之间即使 prompt 不完全相同，也能有效地共享 KV Cache。

### Oneiros 的解决方案

```
用户 A 的 prompt: "Translate to French: Hello"
                        ↓
用户 B 的 prompt: "Translate to French: World"
                        ↓
共享前缀 "Translate to French: " → KV Cache 已缓存
不同部分 "Hello" / "World" → 参数重映射 → 映射到"空槽位"复用 KV 计算
```

核心思想：
1. 识别 prompt 中的**可参数化部分**（不同用户输入差异的部分）
2. 将这些部分通过**重映射函数**映射到一个规范的 KV Cache 槽位集合
3. 不同用户的"不同输入"实际上共享了同一套 KV Cache 表示

## 三、技术方法详解

### 3.1 参数重映射的核心机制

Oneiros 不直接共享原始 KV Cache，而是共享一个**经过参数重映射的 Cache 池**：

```
原始输入: S = [Prefix, Param_A_specific_tokens]
                          ↓
参数重映射函数 f: Token → Slot
                          ↓
共享 Cache: Slot-based KV Cache pool
                          ↓
用户 B 的输入: S' = [Prefix, Param_B_specific_tokens]
                          ↓
参数重映射 g: Token → Slot  (映射到相同或相邻 slot)
                          ↓
Cache 命中！→ 直接使用共享 KV Cache
```

### 3.2 重映射函数的构建

Oneiros 设计了**三类重映射策略**：

| 策略 | 方法 | 适用场景 | 共享收益 |
|------|------|---------|---------|
| **Identity Remap** | 相同 token 直接匹配 | 完全相同的 prompt | 最高 |
| **Hash Remap** | 不同 token 哈希到同一 slot | 语义相似的参数值 | 中等 |
| **Learnable Remap** | 可学习映射矩阵 | 复杂语义关系 | 灵活但需训练 |

**Hash Remap 示例**：
- `"Hello" → slot 1`, `"World" → slot 1`（如果只关心是否"有 token"而不关心具体值）
- `"cat" → slot 2`, `"dog" → slot 2`（如果只在语义层面区分动物 vs 非动物）

### 3.3 精度分析

参数重映射的核心挑战是：**如何在精度损失和共享效率之间取得平衡**。

Oneiros 的理论分析：

设原始 KV Cache 为 $K, V$，重映射后的 Cache 为 $\tilde{K}, \tilde{V}$，则 Attention 输出的误差为：

$$
\Delta = \text{Attn}(Q, K, V) - \text{Attn}(Q, \tilde{K}, \tilde{V})
$$

Oneiros 证明：当重映射满足**局部性假设**（即同一 slot 内的 token 在注意力计算中行为相似）时，$\Delta$ 可以被控制在一个小的范围内。

### 3.4 多租户隔离

Oneiros 通过**slot 分组**来实现多租户隔离：
- 每个租户分配一组 slot
- 跨租户的 slot 共享需显式授权
- slot 的逐出策略：按租户的最近使用时间（LRU per tenant）

## 四、与 GORGO/KVShare 的对比

| 维度 | GORGO/KVShare | Oneiros |
|------|-------------|---------|
| **共享条件** | 相同前缀 | 相似前缀（参数化差异） |
| **核心机制** | 网络感知调度 + 全局索引 | 参数重映射 + slot 池 |
| **技术方法** | 哈希匹配 + 节点选择 | 重映射函数 + 共享槽位 |
| **网络模型** | ✅ 显式建模传输延迟 | ❌ 关注缓存命中而非网络 |
| **适用场景** | 相同模板的不同用户 | 模板略有差异的不同用户 |
| **覆盖范围** | 完全匹配的前缀 | 相似但不完全匹配的前缀 |
| **精度影响** | 无（完全匹配） | 有（重映射引入近似误差） |

**核心差异**：
- GORGO 解决的是"**能否更快地找到并传输**"相同 Cache 的问题
- Oneiros 解决的是"**如何让更多用户共享即使 prompt 不完全相同**"的 Cache 的问题
- 二者在技术路线上互补，可以结合使用

## 五、核心发现

| 对比 | KV Cache 节省 | 精度影响 |
|------|-------------|---------|
| 无共享（基准） | 0% | 基准 |
| 传统 prefix caching | 30-50% (仅共享相同前缀) | 无损失 |
| **Oneiros 参数重映射** | **50-70%** | **<1% 精度损失** |
| GORGO（完全匹配） | 40-60% | 无损失 |
| Oneiros + GORGO（结合） | **60-80%**（估计） | **<1%** |

**关键洞察**：
- 参数重映射的核心收益来自**扩大了共享的范围**：从"完全匹配"扩展到"相似即可"
- 精度损失与重映射粒度直接相关：越粗粒度的映射共享越多，但精度损失也越大
- 在 Few-shot 场景（各用户的 example 不同但格式相同）收益最大

## 六、局限与未来方向

### 局限
1. **重映射近似误差**：不同 token 映射到同一 slot 必然引入信息损失，在长上下文任务中误差可能累积
2. **适用场景限制**：适用于 prompt 结构高度模板化的场景，对完全自由形式的 prompt 收益有限
3. **映射函数选择**：Identify/Hash/Learnable 三种策略各有利弊，缺乏自适应选择机制
4. **Slot 竞争**：多个 token 映射到同一 slot 时的冲突处理策略（GREM/add/noisy）影响精度

### 未来方向
- **自适应重映射**：根据 prompt 的具体内容动态选择重映射策略
- **与量化结合**：Oneiros + KVTuner 的组合：压缩后的重映射共享
- **语义感知映射**：使用 embedding 相似度而非哈希来决定映射关系
- **增量蒸发移除**：结合逐出策略管理 slot 池
- **端到端系统实现**：Oneiros 目前主要是方法研究，缺乏完整的 serving 系统实现

## 七、个人评价

Oneiros 提出了一种**新颖的跨用户 KV Cache 共享视角**——不再局限于完全相同的 prompt 前缀匹配，而是通过参数重映射扩展了共享的范围。这在多租户场景中非常实用，因为实际部署中用户 prompt 往往是"模板一致但参数不同"的模式。

相比 GORGO 侧重网络延迟优化，Oneiros 更侧重于**缓存命中率的提升**。两者结合（Oneiros 扩大共享范围 + GORGO 优化共享传输）能覆盖完整的多租户 Cache 共享场景。

**适用场景**：多租户 SaaS 平台、企业内部推理集群，其中不同租户使用相似但参数化的 prompt 模板。

## 相关链接
- [[GORGO KVShare 跨用户 KV Cache 共享分析]] — 另一种跨用户 Cache 共享方案
- [[KVTuner 混合精度 KV Cache 量化分析]] — 可与 Oneiros 结合的压缩技术
- [[缓存系统性能建模洞察分析]] — Cache 共享策略在全景图中的位置
