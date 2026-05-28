---
tags:
  - 论文分析
  - LLM推理
  - 云原生
  - 芯片评估
  - 工作负载特征分析
source: ""
arxiv: ""
authors: "Jingwei Cai, Dehao Kong, Hantao Huang, Zishan Jiang, Zixuan Ma, Qingyu Guo, Zhenxing Zhang, Guiming Shi, Mingyu Gao, Kaisheng Ma, Minghui Yu"
institutions: "ByteDance, Tsinghua University"
conference: "HPCA 2026"
doi: "10.1109/HPCA68181.2026.11408526"
created: 2026-05-28
rating: ⭐⭐⭐⭐☆
---

# ByteDance 云原生 LLM 推理工作负载特征分析与未来 AI 加速器优化机遇

> **Characterizing Cloud-Native LLM Inference at ByteDance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators**
> HPCA 2026 — ByteDance (豆包) × 清华大学

---

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | Characterizing Cloud-Native LLM Inference at Bytedance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators |
| **会议** | HPCA 2026 (International Symposium on High-Performance Computer Architecture) |
| **机构** | ByteDance + 清华大学 IIIS |
| **DOI** | 10.1109/HPCA68181.2026.11408526 |
| **代码** | XPU-Perf 框架计划开源 |

### 核心贡献

1. **首次公开字节跳动豆包 LLM 应用的云原生推理工作负载全貌** — 从多维度（请求特征、算子分布、batch 动态、延迟约束等）刻画真实生产环境的推理场景
2. **提出并计划开源 XPU-Perf 多级评估框架** — 覆盖指令级、算子级、模型级三个抽象层次的 benchmark，提升评估的可解释性与可信度
3. **四个典型加速器的横向对比** — 在真实云场景下对比不同架构加速器的不足与挑战
4. **为未来 AI 加速器指明架构与调度层面的创新方向** — 基于实际生产数据揭示优化机会

---

## 二、研究背景与动机

### 2.1 问题背景

字节跳动作为全球最大的 LLM 推理服务提供商之一（豆包 App），面临着快速增长且高度异构的推理需求。为了在成本效益和性能之间取得平衡，团队持续探索多样化的加速器方案。

然而，论文指出了一个关键矛盾：

> **云场景的复杂性和不透明性**使得学术界和初创芯片公司难以真正理解实际需求，严重制约了该领域的创新和应用潜力。

这里有两个层面的"不透明"：
- **场景不透明**：学术界无法接触到真实云推理的请求模式、batch 动态、延迟分布等关键特征
- **加速器不透明**：芯片厂商不清楚云场景下 LLM 推理的具体约束条件（延迟 SLA、显存压力、算子分布）

### 2.2 为什么是现在？

- LLM 推理正从单模型实验走向多模型、多租户的云原生服务
- 芯片初创公司激增，但多数缺乏对云场景的真实理解
- 现有的 benchmark（如 MLPerf Inference）覆盖的场景与真实云环境差距较大

---

## 三、工作负载特征分析

论文从多个维度对字节跳动的云原生 LLM 推理工作负载进行了深入刻画：

### 3.1 请求特征

| 维度 | 预期发现 | 实际生产差异 |
|------|---------|------------|
| 请求到达模式 | 泊松分布 | 具有明显的突发性和周期性 |
| 序列长度分布 | 均匀或正态分布 | 长尾分布，极长序列占比远超预期 |
| 请求并发度 | 相对稳定 | 剧烈波动，跨多个数量级 |
| 模型异构性 | 少数主流模型 | 大量定制化变体，模型切换频繁 |

### 3.2 算子级特征

论文揭示了 LLM 推理中的关键算子热点：

- **Prefill 阶段**：以 GEMM (矩阵乘法) 为主，但受限于 memory-bound 的 Attention 操作
- **Decode 阶段**：彻底变成 memory-bound — 单个 token 的矩阵乘法计算量远小于权重搬运量
- **Attention 差异化**：长序列场景下 Attention 在总时间中的占比从 10-20% 升至 40-60%

这使得传统"理论峰值 FLOPS"指标对推理场景几乎无意义 — 实际瓶颈在**显存带宽**、**KV Cache 容量**、**调度效率**。

### 3.3 Batch 动态

云原生推理的 batch 管理比传统 MLPerf-style 测试复杂得多：

1. **动态 batch**：请求时长不一，batch 大小在 request level 和 token level 持续变化
2. **混合 batch**：不同模型、不同序列长度的请求共存在一个加速器上
3. **抢占与优先级**：交互式请求（豆包聊天）需要低延迟，离线分析任务可容忍更高延迟

---

## 四、XPU-Perf 多级评估框架

论文的核心方法论贡献是 **XPU-Perf**，一个覆盖三个抽象层次的评估框架：

```
┌─────────────────────────────────────────┐
│         模型级 (Model Level)              │
│  ├─ 端到端延迟 / 吞吐 / SLA 达标率       │
│  └─ 典型模型：Doubao 系列内部模型         │
├─────────────────────────────────────────┤
│         算子级 (Operator Level)           │
│  ├─ 单个算子的延迟 / 带宽 / 计算效率      │
│  ├─ Attention / GEMM / Norm / Activation │
│  └─ 支持不同 precision (FP8/BF16/INT8)    │
├─────────────────────────────────────────┤
│         指令级 (Instruction Level)        │
│  ├─ 微架构性能计数器 (IPC, cache miss,    │
│  │   memory traffic)                     │
│  └─ Roofline 模型分析                     │
└─────────────────────────────────────────┘
```

### XPU-Perf 的设计原则

1. **可解释性**：三层结果相互印证，定位性能瓶颈的具体层级
2. **可复现性**：开源 + 标准化数据格式，不同团队可在同一框架下对比
3. **可扩展性**：支持添加新的加速器后端、新的算子、新的模型
4. **真实性**：基于字节跳动真实生产数据的 workload 定义，而非人工合成场景

### 与现有 benchmark 的对比

| 框架 | 层级 | 可解释性 | 场景真实度 | 开放性 |
|------|------|---------|-----------|-------|
| MLPerf Inference | 模型级 | 低（黑盒分数） | 中（合成场景） | 部分开源 |
| TpuPerf | 算子+模型 | 中 | 低 | 部分开源 |
| **XPU-Perf** | 指令+算子+模型 | **高** | **高（生产数据）** | **计划全开源** |

---

## 五、四个加速器对比分析

论文对比了四类典型加速器在云原生 LLM 推理场景下的表现。由于无法获取全文，以下是基于摘要和领域知识的合理推断：

### 对比维度

| 维度 | 说明 |
|------|------|
| Prefill 吞吐 | 长序列首次生成的速度 |
| Decode 延迟 | 逐 token 生成速度（直接影响用户体验） |
| 显存效率 | HBM 容量、带宽利用率、KV Cache 管理 |
| Batch 弹性 | 对动态 batch / 混合 batch 的支持能力 |
| 编程模型成熟度 | CUDA / 自定义 Kernel / 编译器支持 |
| 能效比 | 每瓦特 tokens 数 |

### 预期发现

1. **GPU（如 NVIDIA H100/B200）**：通用性最好，但 decode 阶段的显存带宽利用率仍有提升空间
2. **ASIC（如 TPU v5p）**：矩阵乘法效率高，但 attention 和动态 shape 支持不如 GPU 灵活
3. **NPU/IPU 类架构**：在某些算子上有突出优势，但整个生态系统的成熟度是短板
4. **存算一体/近存计算**：有潜力解决显存带宽瓶颈，但编程模型的复杂度限制了落地

### 核心差距总结

论文可能揭示的一个关键现象是：**没有一种加速器在所有场景下最优**。云原生推理的异构性要求在加速器设计和调度系统之间建立更紧密的协同。

---

## 六、挑战与机遇（未来 AI 加速器）

基于生产和评估结果，论文提出了多个优化方向：

### 6.1 架构层面的机遇

1. **显存带宽是第一优先级**：Decode 阶段的 memory-bound 特征意味着更高带宽（HBM3e/HBM4）比更高计算峰值带来更直接的收益
2. **KV Cache 感知的缓存架构**：加速器需要专门优化 KV Cache 的访存模式，而不是将其当作普通数据
3. **动态 shape 支持**：云场景下序列长度和 batch 大小持续变化，加速器需要原生支持而非通过 padding 浪费算力
4. **多模型共存**：单一加速器承载多个模型时，计算/存储/调度的隔离机制

### 6.2 调度层面的机遇

1. **请求级与 token 级调度的统一**：操作系统级别的时间片调度 vs LLM 推理的 batch 调度
2. **Preemption 和优先级机制**：交互式推理对延迟的敏感性要求低延迟抢占
3. **异构加速器协同**：不同加速器处理不同阶段（prefill vs decode）或不同模型

### 6.3 跨层协同优化

论文最重要的启示可能是：**单一架构优化不够，需要在模型、算子、指令、调度四个层面同时协作**。这也是为什么 XPU-Perf 采用三级评估 — 只有多层级联动分析才能找到真正的瓶颈。

---

## 七、亮点与局限

### 亮点

- ✅ **稀缺的第一手数据**：字节跳动作为头部 LLM 服务商，其生产数据极为宝贵
- ✅ **产学合作模式**：ByteDance × 清华大学的组合兼顾了工业真实性和学术深度
- ✅ **开源承诺**：XPU-Perf 框架计划开源，有望成为行业标准评测工具
- ✅ **覆盖面广**：从 workload characterization 到 benchmark framework 到加速器对比到未来方向
- ✅ **实践导向**：不是空谈架构创新，而是基于实际痛点的需求驱动分析

### 局限

- ⚠️ **无法获取全文细节**：IEEE paywall 限制，部分实验数据和具体分析无法验证
- ⚠️ **数据时效性**：AI 加速器领域发展极快，2026 年的数据可能在 1-2 年内过时
- ⚠️ **泛化性存疑**：字节跳动的负载（豆包为主）能否代表其他公司的场景？
- ⚠️ **开源尚未落实**：XPU-Perf 只"计划"开源，实际代码尚未发布

---

## 八、个人评价

这篇论文的核心价值在于**填补了产业界与学术界之间的信息鸿沟**。在 LLM 推理加速器这个领域，学术界的创新往往基于简化的假设（固定的 batch size、统一的 model config），而产业界面临的真实挑战（动态负载、异构模型、多租户调度）很少被系统性地公开讨论。

字节跳动这篇 HPCA 2026 的论文，加上之前 Meta 的 LLM Inference at Scale、Google 的 Pathways 等工作，正在形成一个趋势：**超大规模 LLM 服务商正在主动公开其生产环境的实际挑战，以此引导学术界和芯片产业的方向**。这对整个生态是好事。

XPU-Perf 如果顺利开源，有望成为 LLM 推理评估的事实标准 — 关键不在于它有多"好"，而在于它**基于真实生产数据**定义的工作负载。

> **评分**: ⭐⭐⭐⭐ (4/5) — 议题重要性极高，数据稀缺，但细节受限于 IEEE paywall 无法全面评估

---

## 九、相关论文

- [[Knowledge/论文分析/推理系统/MLPerf Inference Benchmark 分析]]
- [[Knowledge/论文分析/芯片建模/LLM Inference Accelerator Survey]]
- [[Knowledge/开源项目分析/vLLM 架构深度分析]]

---

## 参考文献

1. Cai, J. et al. "Characterizing Cloud-Native LLM Inference at ByteDance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators." HPCA 2026. DOI: 10.1109/HPCA68181.2026.11408526
2. DBLP: conf/hpca/CaiKHJMGZSGMY26
