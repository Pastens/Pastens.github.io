---
tags:
- 论文分析
- llm-training
- communication
- collective
arxiv: 2111.04867
authors: Aashaka Shah, Vijay Chidambaram, Meghan Cowan, Saeed Maleki, Madan Musuvathi,
  Todd Mytkowicz, Jacob Nelson, Olli Saarikivi, Rachee Singh
institutions: UT Austin, Microsoft
created: 2026-05-11
rating: ⭐⭐⭐⭐
---

# TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches

## 一、论文概览

| 属性 | 内容 |
|------|------|
| **标题** | TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches |
| **arXiv** | 2111.04867 |
| **机构** | UT Austin, Microsoft Research |
| **顶会** | OSDI 2022 |

### 核心贡献

1. **拓扑感知的集合通信合成**：自动生成最优 AllReduce/AllGather/ReduceScatter 通信算法
2. **通信草图（Sketch）**：用户提供通信模式的高层描述（"节点间 AllReduce + 节点内 Ring"）
3. **优于手动调优**：在多种拓扑上比 NCCL 默认方案快 2-5×

---

## 二、技术方法

### 通信草图（Communication Sketch）

用户无需指定具体通信序列，只需描述通信意图（"将节点 A 的数据合并到节点 B"）。TACCL 自动搜索最优实现：
- 考虑：拓扑连接、链路带宽、数据量
- 搜索：LTL（线性时序逻辑）约束的通信序列

在 64 GPU/8 节点的 InfiniBand 集群上，TACCL 合成的 AllReduce 比 NCCL 默认快 1.5×。

---

## 相关链接
- [[Knowledge/论文分析/训练系统/LLM训练系统深度综述]]
