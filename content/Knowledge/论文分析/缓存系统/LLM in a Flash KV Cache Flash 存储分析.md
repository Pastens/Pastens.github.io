---
tags:
- 论文分析
- kv-cache
- flash-storage
- offloading
- dataflow
- page-hit-model
arxiv: '2312.11514'
source: https://arxiv.org/abs/2312.11514
authors:
- Keivan Alizadeh
- Iman Mirzadeh
- Dmitry Belenko
- S. Karen Khatamifard
- Minsik Cho
- Carlo C. Del Mundo
- Mohammad Rastegari
- Mehrdad Farajtabar
institutions:
- Microsoft Research
- University of Washington
created: 2026-05-15
rating: ⭐⭐⭐⭐
---

# LLM in a Flash — 将模型权重与 KV Cache 存储在 Flash 存储中

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | LLM in a Flash: Efficient Large Language Model Inference with Limited Memory |
| **arXiv** | 2312.11514 |
| **核心贡献** | 将模型权重和 KV Cache 存储在 Flash（SSD），设计数据流以最小化 Flash 读取 |
| **标签** | Flash 存储, KV Cache 卸载, 数据流优化, 预测性加载, 内存受限部署 |

---

## 二、核心贡献

### 2.1 问题背景

FlexGen 将 KV Cache 卸载到 CPU DRAM + SSD，但**模型权重仍然必须放在 GPU HBM 或 CPU DRAM 中**。LLM in a Flash 更进一步——假设整个系统的**可用 DRAM 极低**（如边缘设备仅 4-8GB），模型权重和 KV Cache 都必须存储在 Flash（SSD/NAND）中。

### 2.2 核心思路

> 将 LLM 的**所有参数**（权重 + KV Cache）存储在 Flash 中，仅在推理时按需加载到 DRAM。核心挑战是最小化从 Flash 读取的数据量，因为 Flash 读取的能效和延迟远高于 DRAM。

```
传统 GPU 推理:
GPU HBM → (权重 + KV Cache 常驻) → 计算

FlexGen 卸载:
GPU HBM (KV 活跃) → CPU DRAM (KV 缓存) → SSD (KV 备份)

LLM in a Flash（最激进）:
Flash (权重 + KV Cache) → DRAM (当前层权重 + 活跃 KV) → 计算
 ↑ 按需加载，最大化数据复用
```

### 2.3 主要贡献

| 贡献 | 说明 |
|------|------|
| **数据流设计** | 最小化 Flash 读取次数的计算-加载流水线 |
| **Page 命中率模型** | 预测性加载策略的决策依据 |
| **权重+KV 联合调度** | 同时考虑权重加载和 KV Cache 加载的协同调度 |
| **低内存适配** | 在 4-8GB DRAM 设备上运行 7B-13B 模型 |

---

## 三、技术方法详解

### 3.1 数据流设计

LLM in a Flash 的数据流核心原则：**尽可能避免从 Flash 重复读取相同数据**。

#### 3.1.1 分块（Blockwise）加载

将模型权重和 KV Cache 分块存储在 Flash 中，每次推理一个层时，仅加载该层所需的权重和 KV：

```
┌─────────────────────────────────────────────┐
│ Flash Storage                               │
│ ┌─────────────────────────────────────────┐ │
│ │ Layer 0: 权重块 | KV 块 | KV 块 | ...   │ │
│ │ Layer 1: 权重块 | KV 块 | KV 块 | ...   │ │
│ │ Layer N: 权重块 | KV 块 | KV 块 | ...   │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
         ↓ 按层按需加载
┌─────────────────────────────────────────────┐
│ DRAM (有限容量)                              │
│ 当前层权重 | 当前活跃 KV Block               │
└─────────────────────────────────────────────┘
```

#### 3.1.2 计算-加载流水线

```
Step 1: 加载 Layer 0 权重到 DRAM
Step 2: 加载 Layer 0 的 KV Cache 到 DRAM
Step 3: 计算 Layer 0 (同时后台加载 Layer 1 的权重)
Step 4: 计算 Layer 1 (同时后台加载 Layer 2 的权重)
...
```

### 3.2 预测性加载策略

由于 Flash 读取延迟远高于 DRAM（~10μs vs ~100ns），LLM in a Flash 采用**预测性加载**策略：在计算当前 token 时，预测下一个 token 可能需要的 KV Page 并提前加载。

#### 3.2.1 Page 命中率模型

核心模型：**Flash 读取的 Page 命中率模型**

```
Page 命中率 = P(next_token 所需的 KV Page 已在 DRAM 中)

影响命中率的因素:
- DRAM 中缓存的 Page 数量 (cache_size)
- 注意力分布模式 (局部注意力 vs 全局注意力)
- 序列长度 (越长 → 总 Page 越多 → 命中率越低)
```

**预测性加载决策**：

```python
def should_prefetch(kv_page, hit_rate_model):
    """
    hit_rate_model 根据历史访问模式预测
    该 Page 在下一步被访问的概率
    """
    predicted_prob = hit_rate_model.predict(kv_page)
    
    # 如果概率超过阈值，则提前加载
    if predicted_prob > PREFETCH_THRESHOLD:
        queue_prefetch(kv_page)
```

### 3.3 权重与 KV 的联合调度

LLM in a Flash 将权重加载和 KV Cache 加载视为**联合优化问题**：

```python
# 联合调度策略
def joint_schedule(layer_idx, active_kv_pages):
    # 第一步：检查该层权重是否已在 DRAM 中
    if weights[layer_idx] not in DRAM:
        schedule_flash_read(weights[layer_idx], priority=HIGH)
    
    # 第二步：检查需要加载哪些 KV Page
    for page in active_kv_pages:
        if page not in DRAM:
            priority = MEDIUM if page in predicted_hot_set else LOW
            schedule_flash_read(page, priority)
    
    # 第三步：执行调度（先高优先级，后低优先级）
    execute_schedule()
```

**关键洞察**：权重加载是必须的（每层计算前必须加载），而 KV Cache 加载是可选的（可以通过重计算避免）。因此权重加载优先级 > KV Cache 加载优先级。

---

## 四、性能模型：Flash 读取的 Page 命中率模型

### 4.1 形式化模型

LLM in a Flash 的性能由 Flash 读取的总数据量决定：

```
T_inference = T_compute + ∑(data_loaded / Flash_BW)

其中 data_loaded 受命中和未命中影响：
data_loaded = weights_size × num_layers    # 权重必须加载
            + KV_size × (1 - hit_rate) × num_steps  # KV 按需（命中可跳过）
```

### 4.2 命中率的影响因素

| 因素 | 对命中率的效应 |
|------|--------------|
| **DRAM 容量** | 越大 → 可缓存更多 Page → 命中率越高 |
| **Flash 读取带宽** | 越高 → 可更频繁预取 → 命中率提升 |
| **序列局部性** | 注意力集中在局部窗口 → 高命中率 |
| **预测准确率** | 预测器精度直接影响预取的有效性 |

### 4.3 模型精度

LLM in a Flash 的性能模型在以下条件下较为准确（误差 < 15%）：
- DRAM 容量固定（如 8 GB）
- 模型架构已知（层数、隐藏维度、注意力头数）
- Flash 读取带宽稳定（无其他 IO 竞争）

---

## 五、局限与未来方向

### 局限

1. **Flash 写入寿命限制**：KV Cache 需要频繁写入 Flash（每个生成步骤写入新 token），加速 Flash 磨损
2. **预测器精度天花板**：预测性加载的命中率受限于预测器精度，长距离注意力模式预测困难
3. **多批次支持不足**：设计主要针对单请求场景，多批次并发时 DRAM 竞争加剧
4. **延迟敏感场景不适用**：Flash 读取延迟（~10μs）导致首次 token 时间（TTFT）显著增加
5. **权重+KV 竞争 DRAM**：模型权重和 KV Cache 都在 DRAM 中竞争有限空间，调度复杂度高

### 未来方向

- **与 CXL 内存结合**：CXL 内存扩展可以提供比 Flash 更低延迟的卸载目标
- **学习型预取器**：使用更先进的序列模型（如小型 Transformer）预测 KV Page 访问模式
- **磨损均衡调度**：将 Flash 写入寿命纳入调度决策，延长硬件寿命
- **多请求联合调度**：扩展支持多请求并发的 KV Cache Flash 管理

---

## 六、个人评价

LLM in a Flash 将 KV Cache 卸载推向了**最极端的存储层级**——直接使用 Flash 存储。相比于 FlexGen（保留 CPU DRAM 作为中间层），LLM in a Flash 假设整个系统的 DRAM 都极度有限，这是一个全新的挑战维度。

**最大贡献**：提出了 Page 命中率模型来指导预测性加载策略，这是 KV Cache 卸载领域第一个**显式建模 Flash 访问模式**的工作。

**与 FlexGen 的关系**：两者是递进关系。FlexGen 探索了 CPU DRAM + SSD 的两级卸载，LLM in a Flash 则是将整个推理过程适配到 Flash-heavy 的存储架构。如果 FlexGen 覆盖的是"中端 GPU + 丰富 DRAM"场景，LLM in a Flash 覆盖的是"低端/边缘设备 + 极有限 DRAM"场景。

**适用场景**：边缘设备推理、物联网场景、内存受限的移动设备、低成本推理集群。

## 相关链接

- [[Knowledge/论文分析/缓存系统/FlexGen IO 感知 KV Cache 卸载分析]] — CPU DRAM + SSD 的两级卸载
- [[Knowledge/论文分析/缓存系统/vLLM PagedAttention 分页 KV Cache 分析]] — 高效的 HBM 管理方案
- [[Knowledge/论文分析/缓存系统/缓存系统性能建模洞察分析]] — KV Cache 研究全景图谱中的位置
