---
title: OpenAI MRC 超级计算机网络协议深度分析
date: 2026-05-21
tags:
- 网络
- AI基础设施
- 超级计算机
- 协议
- OpenAI
- Stargate
- RDMA
- RoCE
- SRv6
- 多路径传输
description: OpenAI 联合 AMD、Broadcom、Intel、Microsoft、NVIDIA 发布的 MRC (Multipath Reliable
  Connection) 多路径可靠连接协议深度分析
source: https://www.openai.com/index/mrc-supercomputer-networking/
---

# OpenAI MRC 超级计算机网络协议深度分析

> **发布日期**: 2026-05-05 | **来源**: OpenAI Engineering Blog  
> **涉及机构**: OpenAI, AMD, Broadcom, Intel, Microsoft, NVIDIA, Oracle Cloud Infrastructure  
> **相关项目**: Stargate, RoCE, Ultra Ethernet Consortium, SRv6, Open Compute Project

---

## 1. 背景

### 1.1 大模型训练的网络瓶颈

随着 AI 前沿模型（Frontier Model）的规模持续增长，训练集群中 GPU 数量已从数千扩展到数十万甚至更多。在同步预训练（Synchronous Pretraining）中，所有 GPU 以锁步方式协同工作，**单一数据传输延迟**即可导致全局停滞。OpenAI 指出：

> 训练一个模型，单个步骤（step）可能涉及数百万次数据传输。一次传输的迟到会像涟漪般波及整个作业，导致 GPU 空转。

网络拥塞、链路故障和设备故障是延迟和不稳定的最主要来源——且**集群规模越大，问题越严重、越难解决**。

### 1.2 Stargate 的挑战

在规划 Stargate（星门）超级计算机时，OpenAI 遇到了两个核心挑战：

1. **最小化网络拥塞**：除了不可避免的"多 GPU 同时发往同一目标"场景外，应通过设计主动避免拥塞。
2. **最小化网络故障对训练的影响**：在足够大的规模下，即使最好的网络也会持续出现链路/交换机故障。传统方案中，**单点故障就可能导致训练作业崩溃**，需要从检查点恢复，或停滞数秒等待路由收敛。

OpenAI 将这种效应称为 **"故障放大器"（failure amplifier）**——作业越大，单链路抖动的影响就越致命。

### 1.3 传统方案的不足

传统方案（如经典的 RoCE 部署）使用**单路径流**（single-path flows），存在两个根本问题：
- 不同流在相同链路上碰撞，产生拥塞
- 每个流只能使用可用网络平面中的一个，路径多样性无法被利用

当网络故障发生时：
- 动态路由协议（如 BGP）需要**数秒甚至数十秒**来收敛
- 交换机本身的软件复杂度引入了额外的故障模式
- 链路修复通常需要协调停机，影响训练持续性

---

## 2. 解决方案：MRC 协议架构

MRC（Multipath Reliable Connection，多路径可靠连接）是 OpenAI 联合 AMD、Broadcom、Intel、Microsoft、NVIDIA 历时两年开发的新型网络协议，已内置到最新的 **800Gb/s 网络接口**中。

### 2.1 核心技术理念

MRC 的核心理念是：**不要追求绝对无故障的网络，而是构建在故障面前仍能稳定工作的网络**。它通过三个支柱实现：

1. **多平面（Multi-Plane）网络拓扑**——提供天然冗余
2. **自适应包喷洒（Adaptive Packet Spraying）**——消除核心拥塞
3. **静态源路由（Static Source Routing / SRv6）**——微秒级故障绕行

### 2.2 多平面网络拓扑

#### 设计思路

将单个高速网络接口（如 800Gb/s）**分割为多个较小的链路**。例如，一个 800Gb/s 的接口可以连接到 **8 台不同的交换机**，每台 100Gb/s，形成 8 个独立的并行网络平面。

#### 拓扑优势

| 指标 | 传统单平面 800Gb/s | MRC 多平面 8×100Gb/s |
|------|-------------------|-------------------|
| 交换机端口密度 | 64 端口 × 800Gb/s | **512 端口** × 100Gb/s |
| 集群规模（全连接） | 需 3-4 级交换机 | **仅需 2 级交换机**连接约 13.1 万 GPU |
| 功耗 | 高 | **显著降低** |
| 组件数（故障点） | 多 | **大幅减少** |
| 路径多样性 | 低 | **极高** |

这使得流量更多地保留在 Tier-0 交换机本地，进一步提升了性能。

### 2.3 自适应包喷洒（Adaptive Packet Spraying）

#### 核心创新

传统 TCP/RoCE 要求同一个传输（transfer）的所有包走同一条路径，以保证顺序到达。MRC 彻底改变了这一模型：

> **MRC 将单个传输的包**喷洒到数百条路径上，跨越所有不同的网络平面。包可以乱序到达——因为每个 MRC 包都携带了目标内存地址，接收端可以直接写入内存。

#### 工作机制

1. **负载均衡**：将流量均匀分布到所有可用路径，避免热点
2. **拥塞感知切换**：每个 MRC 连接为使用的多条路径维护少量状态。检测到某路径拥塞时，自动切换到另一条
3. **故障容错**：一旦丢包，MRC 会**保守处理**——立即停止使用该路径，重传可能丢失的包
4. **路径探针**：退休路径后发送探针包，确认是否存在真实故障、是否已恢复

#### 包裁剪（Packet Trimming）

MRC 还引入了**包裁剪**机制来处理目标端拥塞导致的丢包：

- 当交换机因拥塞即将丢弃包时，改为**裁剪掉载荷**，仅将头部转发到目标
- 目标收到裁剪后的头，触发**显式重传请求**
- 这减少了"误以为路径故障"的假阳性（false positive）

### 2.4 静态源路由（SRv6 源路由）

#### 放弃动态路由

传统数据中心网络依赖 BGP 等动态路由协议，交换机运行复杂的软件来计算和更新路由。这些复杂性本身就引入故障模式。

MRC 的策略：**直接禁用动态路由**，改用 IPv6 段路由（SRv6）。

#### SRv6 工作方式

```
包的路径信息编码在目标地址中，格式为 [sw1, sw2, sw3, ...]
交换机查找自己的标识 → 移除标识 → 偏移目标地址 → 暴露下一跳标识
静态路由表（交换机上线时配置，永不改变）决定物理转发方向
```

#### 核心优势

- **故障绕行由发送端自主决定**——检测到丢包后，立即在发送端选择新路径
- **交换机只需盲目跟随静态路由**——无需收敛、无需重算、无需协商
- 消除了**整类动态路由故障行为**

#### 故障响应时间对比

| 方案 | 故障响应时间 |
|------|------------|
| 传统动态路由 | 秒到数十秒 |
| MRC + SRv6 | **微秒级** |

---

## 3. 部署与效果

### 3.1 部署范围

MRC 已部署在 OpenAI 所有的 **NVIDIA GB200 超级计算机**上，用于训练前沿模型。部署站点包括：

- **Oracle Cloud Infrastructure (OCI)** 得克萨斯州阿比林 Stargate 站点
- **Microsoft Fairwater 超级计算机**

MRC 已用于训练 **多个 OpenAI 模型**，硬件来自 NVIDIA 和 Broadcom。

### 3.2 生产环境实测数据

#### 链路故障容错

- 训练网络中**数百万条链路**
- 观测到 **每分钟多次** Tier-0 到 Tier-1 链路的抖动（link flap）
- **MRC 确保其对同步预训练作业没有可测量的影响**
- 影响小到**无需优先修复那些链路**

#### 交换机维护

- 训练过程中需要重启 **4 台 Tier-1 交换机**
- 之前：需要操作团队非常小心地协调，不能中断训练
- 现在：**甚至不需要跟训练作业的运行团队协调**

#### 链路维修

- 之前：需要操作团队在维护前停用链路
- 现在：**可以在链路仍在服务时进行修复**——如果链路工作得足够好，MRC 就会使用它；如果不行，MRC 会避开它直到修好

#### GPU 网口故障

- 一个 8 端口网络接口失去 1 个端口 → 最大速率降低 1/8
- MRC 检测后立即重算路径，避开故障平面
- 同时通知对端**不要使用该平面发送入站流量**
- 大多数故障链路在 **1 分钟内恢复**，MRC 自动将其重新纳入使用
- 实际训练速度损失**远小于物理容量损失的比例**

---

## 4. 三大关键优势总结

| 优势 | 描述 | 直接影响 |
|------|------|---------|
| **简化拓扑** | 2 级交换机连接 10 万+ GPU，替代传统 3-4 级 | 更低功耗、更少故障点、更低成本 |
| **消除核心拥塞** | 自适应包喷洒使得网络核心**基本没有拥塞** | 训练中消除异常值延迟，多作业共享集群互不干扰 |
| **极速故障绕行** | SRv6 源路由 + 微秒级路径切换 | 链路故障对训练无感，维护不再需要协调停机 |

---

## 5. 行业影响与开放性

### 5.1 MRC 的开放

MRC 规范已通过 **Open Compute Project (OCP)** 向社区开放：
- 规范地址：[OCP MRC 贡献](https://www.opencompute.org/)
- 合作论文：*"Resilient AI Supercomputer Networking using MRC and SRv6"*（已发表）

### 5.2 技术脉络

MRC 的技术继承关系：
- **底层**：扩展 RoCE（RDMA over Converged Ethernet）标准
- **中间**：借鉴 Ultra Ethernet Consortium (UEC) 的技术
- **上层**：引入 SRv6 源路由适配大规模 AI 网络

### 5.3 对行业的意义

1. **网络设计范式的转变**：从"追求无故障网络"转向"在故障中仍能稳定运行"
2. **跨行业协作典范**：AMD、Broadcom、Intel、Microsoft、NVIDIA 五家巨头联手
3. **开放生态**：通过 OCP 开放，可供整个行业使用和构建

---

## 6. 关键启示

- **同步训练是故障放大器**——集群越大，单点故障影响越大，容错设计不再是可选项而是必选项
- **简化即可靠**——通过放弃复杂的动态路由协议，采用静态源路由，反而获得了更强的故障弹性
- **路径多样性是性能的保障**——多平面拓扑 + 包喷洒的组合，从架构层面解决了拥塞问题
- **网络接口的外延创新**——在 800Gb/s 网卡层面的内置协议增强，比单纯改进交换机更灵活

---

## 7. OCP MRC 1.0 规范要点解析

> 完整标题: **Multipath Reliable Connection (MRC) Specification Revision 1.0**  
> 发布日期: 2026-03-21 | 页数: 76 页  
> 贡献机构: AMD, Broadcom, Intel, Microsoft, NVIDIA, OpenAI  
> 主要作者: Rip Sohan (AMD), Eric Spada (Broadcom), Eric Davis (Broadcom), Mark Handley (Broadcom), 以及 Idan Burstein, Tony Hurson, Jithin Jose, Vivek Kashyap, Rong Pan, Sayantan Sur

### 7.1 规范定位与范围

MRC 规范扩展了 InfiniBand Reliable Connection (RC) 传输模型，增加了**显式多路径支持、拥塞控制、路径健康跟踪和故障恢复**。规范运行在标准 best-effort Ethernet 之上，其 opcode [7:5] = 0b110（与标准 RC 区分）。

### 7.2 支持的传输操作（大幅精简）

MRC 仅支持 **6 种 opcode**，全部是 Write/WirteIMM 操作：

| Opcode | 名称 | 说明 |
|--------|------|------|
| 0xC6 | RDMA WRITE First | 多包消息首包 |
| 0xC7 | RDMA WRITE Middle | 多包消息中间包 |
| 0xC8 | RDMA WRITE Last | 多包消息尾包 |
| 0xC9 | RDMA WRITE Last with Immediate | 多包消息尾包 + 立即数 |
| 0xCA | RDMA WRITE Only | 单包消息 |
| 0xCB | RDMA WRITE Only with Immediate | 单包消息 + 立即数 |

**不支持**：RDMA Read、Send、Atomic 操作，不支持 RNR-NAK 流控。

### 7.3 三种路由模式

| 模式 | 路径决定机制 | 适用场景 |
|------|------------|---------|
| **ECMP** | 基于 UDP 源端口 + IPv6 Flow Label 哈希 | 传统 CLOS 网络 |
| **Structured EV** | IPv6 Flow Label + UDP 源端口组成 32 位熵值，分段编码各级转发决策 | 需确定性路径规划的 NIC |
| **SRv6 uSID** | 使用 uSID 地址栈编码完整路径，交换机执行 pop-and-left-shift | 最高灵活度 |

#### `Structured EV` 详解
- IPv6 中：低 16 位 Flow Label + 16 位 UDP Source Port = 32 位 Structured EV
- 被分割为多个 hop-specific 子字段，每个子字段编码特定网络级的转发决策
- 示例：3 跳结构，宽度分别为 10b、8b、4b

#### `SRv6` 详解
- 使用 uSID 容器，LID + 最多 6 个 uSID 描述 GPU 间路径
- 支持 shortcut 路径（T0 和 T1 shortcut，不经过 T2 根交换机）
- 接收端支持：保留外 SRv6 封装（End/End.X）或解封装（End.DX/End.DT）

### 7.4 协议 header 栈关键变更

#### RETH 变化
- 对于多包消息，**每个包都包含 RETH header**（与传统 IBTA 仅在首包放 RETH 不同）
- VA 地址每个包递增 MTU
- 需要做 MR boundary check

#### 新增 header
- **METH（Message Extension Transport Header）**：携带 RQMSN（16 位 WriteIMM 的 Receive Queue MSN）和 MSN（16 位 Requestor 消息序列号）
- **TSETH（Timestamp Extension Header）**：16 位 `tx_timestamp`，分辨率 128ns，在 SACK 中反射用于 RTT 计算

#### BTH 修改
- 新增 `RTX` 位：标记重传包
- 新增 `TS`（TSETH）位：标记存在 TSETH header

### 7.5 可靠交付机制

#### ACK 体系分离
MRC 将 **Reliability SACK** 与 **Transport ACK** 逻辑分离：
- Transport ACK → 标准 RDMA 交付确认
- Reliability SACK → 提供 Responder bitmap 状态，支持快速丢包检测与拥塞控制
- 一个包可被 Reliability SACK 确认但被 Transport NAK 拒绝

#### Responder 乱序处理窗口

```
窗口 = [cack_psn + 1, cack_psn + max_psn_range)
```

- `max_psn_range (mpr)`：128 的整数倍，默认可达 32 × 128 = 4096 个包
- 落入窗口的包有效；PSN 低于 `cack_psn - 2^23` 或高于 `cack_psn + mpr` 的包静默丢弃
- 支持跨消息边界的乱序接收和跟踪

#### 包裁剪（Packet Trimming）

交换机在拥塞时截取载荷，仅转发头部到 Responder。DSCP 映射体系：

| DSCP 类 | 必需性 | 用途 |
|---------|--------|------|
| DSCP_TRIMMABLE | 可选 | 可被 TRIM 的数据 |
| DSCP_NO_TRIM | **必需** | 不可 TRIM |
| DSCP_TRIMMABLE_RETX | 可选 | 重传 TRIM 类 |
| DSCP_TRIMMED | **必需** | 已 TRIM 的包 |
| DSCP_TRIMMED_LASTHOP | 可选 | 最后一跳 TRIM |
| DSCP_CONTROL | **必需** | 控制流量 |

### 7.6 拥塞控制：NSCC 算法

采用 UET Network Signal Congestion Control 算法，**每 QP 运行**，发送端、SACK 时钟驱动、基于窗口。

**两个强制信号**：
1. **RTT**（滞后指标）：估算排队延迟
2. **ECN**（领先指标）：网络拥塞信号

**窗口调整逻辑**：

| ECN | RTT vs target_Qdelay | 推断状态 | 调整 |
|-----|---------------------|---------|------|
| 未设置 | RTT < target | 不拥塞 | 比例增加 (AI) |
| 未设置 | RTT ≥ target | 接近拥塞 | 公平增加 |
| 设置 | RTT ≥ target | 拥塞 | 乘性减窗 (MD) |
| 设置 | RTT < target | 拥塞已缓解 | 不调整 |

**发送调度器四状态**：IDLE → READY → ACTIVE → PENDING，循环转换。

### 7.7 路径感知 EV 状态机（核心创新）

EV（Entropy Value）是 MRC 中描述**一条路径**的抽象概念，每个包通过 EV 选择路径。

#### EV 四状态

| 状态 | 集合 | 说明 | 触发条件 |
|------|------|------|---------|
| **GOOD** | Active | 可正常发送 | 初始状态 |
| **SKIP** | Inactive | 临时跳过，自动恢复 | ECN 标记 / TRIM 拥塞信号 |
| **ASSUMED_BAD** | Inactive | 路径不可达 | 超时 / 持续故障 |
| **DENIED** | Inactive | 控制面禁用 | 管理员配置 |

#### EV 状态转换机制

- SACK 的 `M` 标志为 `SKIP_ONCE`（0b01）→ 该 EV 进入 SKIP
- SACK 的 `M` 标志为 `ALWAYS_SKIP`（0b10）→ 进入 ASSUMED_BAD
- NACK 的 TRIMMED（非末跳）→ 进入 SKIP
- PROBE 响应 → 可恢复为 GOOD

**关键的"保守策略"**：一旦丢包，立即停止使用该路径（移除出 Active），这与传统方案中等待路由收敛形成根本对比。

### 7.8 负载均衡推荐

- Active EV 集合大小建议：**1-2 个拥塞窗口**之间最佳
- 超过 100 个后收益递减
- 主动负载均衡（active load-balancing）显著优于 oblivious spraying

### 7.9 软件架构

两个独立接口：

| 接口 | 头文件 | 使用者 | 功能 |
|------|-------|-------|------|
| **MRC 应用 API** | `mrc.h` | NCCL 等上层库 | 设备发现、QP/CQ 管理、发送 WR |
| **MRC 控制器 API** | `mrc_ctl.h` | 特权进程 | 设备管理、CC/EV profile、EV 事件处理、EV 探测 |

每个厂商实现自己的 MRC provider driver，独立于 libibverbs。

### 7.10 与标准 RoCE RDMA 的关键差异总表

| 维度 | RoCE RC | MRC |
|------|---------|-----|
| 路径 | 单路径 | 多路径（ECMP/Structured EV/SRv6） |
| 支持操作 | Read/Write/Send/Atomic | 仅 Write 和 WriteIMM |
| 每个包 RETH | 仅在首包 | 所有包都有 |
| ACK 体系 | Transport ACK only | Transport ACK + SACK/NACK 分离 |
| 乱序处理 | 视为错误 | **标准行为** |
| 拥塞控制 | DCQCN 等 | NSCC（UET 算法） |
| 流控 | RNR-NAK | 无端到端流控 |
| 重传粒度 | 整个消息 | **选择性逐包重传** |
| 路径感知 | 无 | EV 状态机 + 主动负载均衡 |
| 链路故障响应 | 上层感知（秒级） | EV 自动降级 + Probe 恢复（微秒级） |
| SRv6 支持 | 无 | 原生支持 |
| QP 类型 | RC/UC/UD | 仅 MRC（扩展 RC） |

---

## 8. 华为 Unified Bus (UB) 与 MRC 的对比分析

> 华为 Unified Bus (UB) 是华为 CloudMatrix 架构中的核心互连技术，本文基于 CloudMatrix384 论文（arXiv:2506.12708）和 UB-Mesh 论文（arXiv:2503.20377）进行分析。

### 8.1 UB 技术概述

华为 Unified Bus (UB) 是专门为 AI 超级节点设计的**超高速、低延迟 Scale-Up 网络**。UB 是华为 CloudMatrix 架构的核心基石，其核心理念是实现"一切皆可池化、平等对待、自由组合"的全对等（peer-to-peer）硬件架构。

#### 架构体系

CloudMatrix384 超级节点拥有**三个独立的网络平面**：

| 网络平面 | 互连技术 | 范围 | 连接对象 | 功能 |
|---------|---------|------|---------|------|
| **UB 平面** | Unified Bus (自有协议) | 超级节点内 Scale-Up | 384 NPU + 192 CPU | TP/EP 通信、池化内存访问、KV 缓存共享 |
| **RDMA 平面** | RoCE | 超级节点间 Scale-Out | NPU ↔ NPU | 跨节点训练推理、PD 分离传输 |
| **VPC 平面** | 标准以太网 / UBoE | 数据中心 | NPU/CPU ↔ 存储 | 管理面、持久化存储、外部服务 |

#### UB 拓扑

- **L1 级**：每个计算节点内，7 个板载 UB 交换机芯片连接 8 NPU + 4 CPU
- **L2 级**：12 个计算机柜 → 4 个通信机柜，L2 交换机分为 **7 个独立子平面**
- 每个子平面 16 个 L2 UB 交换机，每芯片 48 端口
- **无带宽超分**（non-blocking），确保全对等全带宽互连
- 跨节点带宽衰减 < **3%**，延迟增加 < **1 μs**

#### 软件栈

- **CANN**（Compute Architecture for Neural Networks）作为中间层
- 上层支持 PyTorch、TensorFlow、MindSpore
- 应用级通信通过华为的 HCCS（Huawei Cache Coherence System）接口

### 8.2 UB-Mesh：nD-FullMesh 拓扑

UB-Mesh（arXiv:2503.20377）是 UB 的网络拓扑演进：

- 采用 **nD-FullMesh** 拓扑，利用 LLM 训练的数据局部性
- 优先使用短距离直连互连，最小化数据移动距离
- 引入 **All-Path-Routing (APR)** 路由机制，高效管理数据流量
- **64+1 备份设计**：64 个活跃端口 + 1 个冷备端口，提升可用性 7.2%
- 比传统 Clos 架构性价比高 **2.04 倍**
- 各种 LLM 训练任务线性度 **95%+**

### 8.3 MRC vs UB 核心对比

#### 8.3.1 定位差异

| 维度 | MRC (OpenAI) | UB (Huawei) |
|------|-------------|-------------|
| **定位** | **Scale-Out** 网络协议 | **Scale-Up** 网络架构 |
| **目标规模** | 10 万+ GPU（跨超级节点） | 384 NPU/超级节点（节点内） |
| **开放性** | **OCP 开放标准** | 华为自有/专有 |
| **协议层级** | **传输层协议**（扩展 RoCE） | **物理/交换层架构** + 自有协议 |
| **标准基础** | 扩展 RoCE + UEC + SRv6 | 华为自有硬件 + 自有协议 |

#### 8.3.2 技术方案对比

| 维度 | MRC | UB |
|------|-----|----|
| **网络拓扑** | 多平面（Multi-Plane）以太网，2 级交换机 | 2 级 UB 交换机（L1+L2），7 子平面 |
| **路由方式** | SRv6 静态源路由 / Structured EV / ECMP | All-Path-Routing (APR) |
| **包调度** | **自适应包喷洒** + EV 状态机 × 数百条路径 | 基于拓扑的 APR 路由 |
| **拥塞控制** | **NSCC**（UET 算法，ECN + RTT 双信号） | 未公开详细算法 |
| **可靠性** | 每个包带 RETH，EV 状态自动降级 + Probe 恢复 | **64+1 备份**，APR 自动故障绕行 |
| **故障响应** | **微秒级**（发送端自主决策路径切换） | **微秒级**（APR 路由） |
| **乱序处理** | **原生支持**（每个包带内存地址） | 未公开（HCCS 层面处理） |
| **丢包检测** | SACK/NACK + 选择性逐包重传 | 未公开 |
| **硬件需求** | 标准以太网交换机 + 支持 MRC 的 800Gb/s NIC | 华为专用 UB 交换机（LRS/HRS）+ Ascend NPU |

#### 8.3.3 处理的问题对比

| 问题 | MRC 方案 | UB 方案 |
|------|---------|---------|
| **网络拥塞** | 包喷洒 + 多平面 + NSCC 拥塞控制 | 非阻塞全对等拓扑 + APR |
| **链路/交换机故障** | EV 状态机微秒级路径切换 | 64+1 备份 + APR 自动故障切换 |
| **维护中断** | 无需停机，MRC 自动避开故障链路 | 备份设计允许热更换 |
| **拥塞控制假阳性** | TRIM（包裁剪）减少误判 | 未公开 |
| **集群规模扩展** | 多平面拓扑 + 标准以太网交换机，10 万+ | 384 NPU 节点内 + RDMA Scale-Out 到 16.5 万 |
| **多作业隔离** | 喷洒消除核心拥塞，作业互不干扰 | 非阻塞拓扑天然隔离 |
| **标准兼容** | 扩展 RoCE，兼容现有 RDMA 生态 | 专有 UB 协议 + 独立 RoCE 平面 |

### 8.4 关键区别总结

#### 1. 开放生态 vs 垂直整合

MRC 是**多厂商标准**——由 AMD、Broadcom、Intel、Microsoft、NVIDIA、OpenAI 联合制定，通过 OCP 开放，运行在标准以太网上，生态兼容性最强。

UB 是**华为垂直整合方案**——从 NPU（Ascend 910）、CPU（Kunpeng）、UB 交换机（LRS/HRS）、到协议栈（UB/CANN/HCCS）全栈自研，性能最优但锁定华为生态。

#### 2. Scale-Up vs Scale-Out 的范式差异

| 特征 | MRC | UB |
|------|-----|----|
| **互联范围** | **超级节点之间**（跨机柜、跨集群） | **超级节点之内**（板间、机柜内） |
| **延迟要求** | 容忍微秒级延迟（10-100μs） | 追求纳秒级延迟（< 1μs 跨节点增量） |
| **传输类型** | 以太网包（可变长、ECN、头压缩） | 定制链路（更高带宽密度） |
| **交换机** | 商用标准以太网交换机 | 华为定制低/高基数交换机（LRS/HRS） |

UB 本质上解决的是**如何把 384 个 NPU 变成一台"超级计算机"**的问题，而 MRC 解决的是**如何把成千上万台这样的"超级计算机"连接起来高效训练**的问题。两者是**互补而非竞争**关系。

#### 3. 故障处理的哲学差异

- **MRC**：采用**"先避开，再确认"**的保守策略——丢包立即停止使用该路径，用 Probe 确认是否真的故障，适合万卡规模的故障普遍性
- **UB**：采用**"硬件冗余 + 拓扑弹跳"**策略——非阻塞拓扑天然避免拥塞，64+1 备份提供容错，APR 提供路由级故障绕行

### 8.5 值得关注的趋势

1. **UBoE（UB over Ethernet）**：华为在 VPC 平面提出了 UBoE，尝试将 UB 的低延迟特性扩展到以太网，这可能是未来 UB 走向开放的第一步
2. **UB-Mesh 的超级节点扩展**：当前 CloudMatrix384 是 384 NPU，论文提出未来可扩展到更大规模
3. **统一网络平面**：华为在 CloudMatrix 长远愿景中计划将 RDMA 和 VPC 平面合并为单一统一网络平面，与 MRC 的"统一以太网"理念一致
4. **MRC 也可用于 Scale-Up**：虽然 MRC 当前定位是 Scale-Out，但其多平面喷洒 + SRv6 的设计也适用于超级节点内部的高带宽互连

---

## 参考文献

1. OpenAI. "Supercomputer networking to accelerate large scale AI training." *OpenAI Engineering Blog*, May 5, 2026. https://www.openai.com/index/mrc-supercomputer-networking/
2. "Resilient AI Supercomputer Networking using MRC and SRv6." OpenAI, et al., 2026.
3. Sohan, R., Spada, E., Davis, E., Handley, M., et al. "Multipath Reliable Connection (MRC) Specification Revision 1.0." *Open Compute Project*, March 21, 2026. https://www.opencompute.org/documents/ocp-mrc-1-0-pdf
4. Ultra Ethernet Consortium. "UEC Specification." https://ultraethernet.org/
5. Zuo, P., Lin, H., Deng, J., et al. "Serving Large Language Models on Huawei CloudMatrix384." *arXiv:2506.12708*, June 2025. https://arxiv.org/abs/2506.12708
6. Liao, H., Liu, B., Chen, X., et al. "UB-Mesh: a Hierarchically Localized nD-FullMesh Datacenter Network Architecture." *arXiv:2503.20377*, March 2025. https://arxiv.org/abs/2503.20377
