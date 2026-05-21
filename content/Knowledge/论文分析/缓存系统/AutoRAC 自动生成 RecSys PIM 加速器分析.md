---
tags:
- 论文分析
- recsys
- pim
- processing-in-memory
- embedding
- accelerator
- design-space-search
arxiv: '2505.10748'
created: 2026-05-15
rating: ⭐⭐⭐⭐
permalink: autorac-recsys-pim
---

# AutoRAC — 自动生成推荐系统的 PIM 加速器设计

## 一、论文基本信息

| 属性 | 内容 |
|------|------|
| **标题** | AutoRAC: Automatic PIM Accelerator Generation for Recommender Systems |
| **arXiv** | 2505.10748 |
| **核心贡献** | 自动生成推荐系统的 PIM（Processing-in-Memory）加速器设计，将 Embedding 查找的访存模式映射到近存储计算单元 |
| **关键词** | PIM、推荐系统、Embedding 查找、近存储计算、设计空间搜索、自动生成 |

## 二、核心思想

### 问题

推荐系统中的 Embedding 查找（Embedding Lookup）是典型的**内存密集型**操作，存在严重的**内存墙**问题：

- Embedding 表规模可达数 TB，远超片上缓存容量
- Embedding 查找表现为稀疏、随机的内存访问模式，访存带宽成为瓶颈
- 传统冯·诺依曼架构下，数据搬运开销远超计算开销

### 解决方案

AutoRAC 提出一种 **PIM（Processing-in-Memory）加速器自动生成框架**，核心思路是将 Embedding 查找的访存模式映射到 PIM 近存储计算单元：

1. **访存模式建模**：将 Embedding 查找的访存行为形式化为可映射的计算图
2. **PIM 设计映射**：将 Embedding 聚合操作下推到近存储处理单元（PIM Bank）中执行
3. **设计空间搜索**：自动化地搜索 PIM 设计参数（如 Bank 数量、处理单元配置、数据布局策略）
4. **代码自动生成**：根据搜索结果自动生成完整的加速器 RTL 代码

### 关键技术

- **访存-计算联合建模**：联合分析 Embedding 查找的访存模式与 PIM 计算单元的匹配度
- **数据布局优化**：将频繁访问的 Embedding 行放置在就近的 PIM Bank 中，减少跨 Bank 通信
- **流水线调度**：优化 PIM Bank 间的计算流水线，提高并行度
- **设计空间枚举**：对 PIM 架构参数（带宽、Bank 数量、近存储计算粒度）进行系统化搜索

## 三、方法详解

### 3.1 Embedding 查找的 PIM 映射

推荐系统中的 Embedding 查找包含以下步骤：

1. **索引查询**：根据用户/物品 ID 查询 Embedding 表
2. **向量读取**：从 HBM/DRAM 中读取对应 Embedding 向量
3. **聚合计算**：对读取的 Embedding 向量进行求和/平均/拼接等聚合操作

AutoRAC 将步骤 (2) 和 (3) 下推到 PIM 近存储单元中执行，避免数据跨片搬运。

### 3.2 自动设计空间搜索

AutoRAC 搜索的关键设计参数包括：

| 参数 | 描述 | 影响 |
|------|------|------|
| **PIM Bank 数量** | 近存储计算单元的数量 | 并行度与面积开销的权衡 |
| **Bank 内计算能力** | 每个 Bank 支持的最大向量操作数 | 单 Bank 吞吐率 |
| **数据分布策略** | Embedding 行到 Bank 的映射方式 | 跨 Bank 通信开销 |
| **聚合粒度** | 单次聚合操作处理多少 Embedding 向量 | 计算延迟与硬件利用率 |
| **存储带宽分配** | HBM/DRAM 带宽在 PIM Bank 间的分配 | 访存瓶颈的缓解程度 |

### 3.3 自动代码生成

AutoRAC 将搜索结果转化为可综合的 RTL 代码，包含：

- PIM Bank 控制逻辑
- 数据路由与互连网络
- Embedding 聚合计算单元
- 与主机的接口逻辑

## 四、优势与局限性

### 优势

- **自动化程度高**：从 PIM 设计空间搜索到 RTL 生成全自动
- **访存效率大幅提升**：消除 Embedding 查找的数据搬运开销
- **可扩展性强**：支持不同规模的 Embedding 表配置
- **设计空间覆盖全面**：系统化搜索 PIM 设计参数

### 局限性

- **PIM 硬件依赖**：需要定制化的近存储计算硬件支持
- **访存模式假设**：假设 Embedding 查找的访存模式可被 PIM 单元有效利用
- **生成代码质量**：自动生成的 RTL 可能不如手动微调的加速器高效
- **动态稀疏性处理**：推荐的稀疏 Embedding 访问模式变化时，静态的数据分布策略可能失效

## 五、与缓存系统的关联

虽然 AutoRAC 主要聚焦于 PIM 加速器生成，但其与缓存/KV Cache 系统有深层的交叉点：

1. **访存模式类比**：Embedding 查找的稀疏随机访存与 LLM 推理中 KV Cache 的访问模式有相似之处（随机稀疏）
2. **近存储计算范式的迁移**：PIM 的思路可迁移到 KV Cache 系统中，即"在 KV Cache 存储附近执行计算"
3. **设计空间搜索方法论**：AutoRAC 的设计空间搜索方法适用于缓存系统参数调优（如缓存策略、预取策略）

## 六、总结与展望

AutoRAC 开创性地将 **PIM 加速器的设计自动化**引入推荐系统领域，为 Embedding 查找的访存瓶颈提供了体系结构层面的解决方案。其自动设计空间搜索和 RTL 生成方法具有通用性，未来可扩展到：

- 大语言模型的 KV Cache PIM 加速
- 图神经网络的近存储处理
- 多模态检索系统的 PIM 加速
- 更复杂的跨 Bank 数据流优化策略
