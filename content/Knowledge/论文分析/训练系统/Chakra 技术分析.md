---
tags:
  - 论文分析
  - 训练系统
  - 性能建模仿真
arxiv: 2305.14516
title: "Chakra: 基于标准化执行轨迹推进性能基准测试与协同设计"
authors: Srinivas Sridharan, Taekyung Heo, Louis Feng, Zhaodong Wang, Matt Bergeron, Wenyin Fu, Shengbao Zheng, Brian Coutinho, Saeed Rashidi, Changhai Man, Tushar Krishna
institutions: Meta, Georgia Institute of Technology, Hewlett Packard Enterprise
created: 2026-05-12
rating: ⭐⭐⭐⭐
---

# Chakra: 基于标准化执行轨迹推进性能基准测试与协同设计

## 一、论文概览

### 1.1 背景与动机

现代深度学习系统高度依赖分布式训练，涉及数据并行、模型并行、流水线并行及其混合策略（3D/4D并行）。然而，当前ML系统的性能建模与协同设计面临三大核心挑战：

1. **缺乏统一的执行轨迹交换标准**：不同ML框架（PyTorch、TensorFlow等）产出的执行轨迹格式各异，阻碍了跨组织的信息交换与协同设计。
2. **性能建模工具链不完善**：缺乏系统化的工具链来识别瓶颈并高效调试。
3. **缺乏执行轨迹合成能力**：企业因知识产权保护无法共享详尽的训练轨迹数据；同时需要能够生成面向未来系统配置（不同NPU数量、网络拓扑）的轨迹以进行性能预测。

### 1.2 核心贡献

Chakra 提出了一套完整的开源基础设施，核心创新点包括：

1. **Chakra ET 图模式（Schema）**：一种开放、最小化且可扩展的图模式，用于标准化描述ML执行轨迹，捕获关键操作与依赖关系。
2. **执行轨迹采集与转换工具链**：支持从PyTorch、FlexFlow等框架采集轨迹并转换为Chakra统一格式。
3. **基于生成式AI的轨迹合成**：利用生成模型学习大量生产轨迹的潜在统计特性，合成能保留分布特征但混淆原始信息的轨迹。
4. **开源模拟器集成**：与ASTRA-sim等开源训练系统模拟器打通，形成端到端的性能评估流程。

---

## 二、技术方法详解

### 2.1 Chakra ET 模式设计

每个节点包含以下核心字段：`id`、`name`、`type`（枚举：COMP, MEM_LOAD, MEM_STORE, COMM_SEND, COMM_RECV, COMM_COLL）、`parent`（父节点ID列表）、`attribute`（扩展属性）。

- **属性字段**受ONNX模式启发，支持float/int/string及其重复类型
- **节点类型枚举**覆盖计算、内存、点对点通信和集合通信四大类
- **每个NPU一张轨迹图**：假设每个NPU对应一张独立的执行轨迹

### 2.2 执行轨迹采集与合成

**真实轨迹采集**：通过Execution Graph Observer扩展PyTorch Profiler，无需侵入式修改模型代码。

**生成式AI合成轨迹**（技术亮点）：
- **"主轨迹"概念**：将N条轨迹（每个rank一条）无损压缩为一条主轨迹，使其可无损重构
- **分层生成模型**：通信类型生成器 → 消息大小生成器（GMM拟合） → 依赖关系建模
- 层次化设计相比单一黑盒生成模型更易调试，结果可解释

### 2.3 开源工具链

| 工具 | 功能 |
|------|------|
| **ET Converter** | 将PyTorch、FlexFlow等格式执行轨迹转换为Chakra统一模式 |
| **ET Visualizer** | 可视化Chakra ET的节点依赖关系 |
| **Timeline Visualizer** | 时间轴展示各NPU任务执行情况 |
| **Test Case Generator** | 允许用户定义任意执行轨迹图 |
| **Trace Feeder** | 供模拟器集成使用的C++库 |

---

## 三、实验评估

**目标系统**：2D-Torus（8×8，64 NPU）和 DGX2（层次化拓扑）

**ML负载**：MLP, Transformer (1.65亿参数), DLRM

**关键结果**：
- NPU规模扩展：Transformer计算密集型表现最佳；MLP-MP通信暴露时间占主导
- 网络带宽灵敏度：2D-Torus整体优于DGX2
- 在Dreamshard、GPU性能建模等场景有落地验证

---

## 四、亮点与局限

### 亮点

1. **填补关键空白**：首个系统化的、开源的执行轨迹标准化方案
2. **模式设计的实用主义**：受ONNX启发的最小化+可扩展设计
3. **生成式AI合成轨迹**：首次将层次化生成模型引入执行轨迹合成
4. **完整的工具链生态**

### 局限

1. 单NPU单轨迹假设限制了全局调度优化
2. 合成轨迹验证深度不足
3. 实验规模仅到64 NPU
4. 生成模型主要聚焦通信部分

---

## 五、个人评价

Chakra 是一篇典型的"系统基础设施"论文，其贡献不在于提出新的算法或模型，而在于**设计了一套可能改变行业实践的标准和工具**。这种"使能型"工作在AI系统研究领域具有独特价值——正如指令迹对计算机体系结构研究的推动作用，Chakra ET有望对ML系统研究产生类似影响。
