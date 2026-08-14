---
tags:
  - 业界动态
  - Agent基础设施
  - 编程范式
  - 动态组合
source: https://deepseek-harness.github.io/deepseek-harness/
date: 2026-08-14
---

# DeepSeek Harness 架构深度分析

> **发布日期**: 持续更新（VitePress 文档站） | **来源**: deepseek-harness.github.io（GitHub org `deepseek-harness`，仓库私有，Pages 由私有仓库构建）
> **涉及机构**: DeepSeek-AI
> **关联**: [[Knowledge/论文分析/软件系统/Cordis 时空可组合性编程范式 深度技术分析]]（Cordis 论文分析）

---

## 1. 背景

DeepSeek Harness（下称 dsh）是 DeepSeek 自研的 **agent harness 平台**——即运行 AI agent 的"宿主框架"：管理模型接入、工具执行、会话状态、权限沙箱、子代理编排、用户交互等。其文档站约 70 个页面（中英双语，中文为主），覆盖从架构概念到每个子系统的生成式 API 参考。

**为什么会关注它**：在分析 *A Programming Paradigm for Spatiotemporal Composability*（Cordis 论文，北大 + DeepSeek-AI）时，发现 Cordis v4 的 README 文档链接指向 `deepseek-harness.github.io/deepseek-harness/reference/cordis-primer`。顺着该链接研究后发现：**dsh 就是 Cordis 论文 §1.2.2 动机场景（自进化 agent harness）与结论 future work 的生产落地**——Cordis 以 vendor 方式引入 dsh 底层（npm scope `@deepseek-ai/cordis`），论文形式化的"可逆效应 + 反应式余效应"被逐条兑现为产品机制。

## 2. 平台架构

### 2.1 总体设计：一切皆插件，无特权内核

```
┌─────────────────────────────────────────────────────────┐
│  Profile（命名组装，如 web / headless）                    │
│  ├── 组合包 Bundle（dsh-base, dsh-web-app, dsh-headless） │
│  ├── cordis.patch.yml（profile 级 / home 级 / --patch）   │
│  └── 树外插件（用户安装）                                  │
├─────────────────────────────────────────────────────────┤
│  Cordis 插件树（@deepseek-ai/cordis，vendor）              │
│  └── 每个产品功能都是一个插件：                            │
│      模型适配器 / 工具注册表 / 会话日志 / agent loop 本身   │
└─────────────────────────────────────────────────────────┘
```

文档原话（架构页）：*"不存在需要打补丁的特权内核：扩展 dsh 的方式是把插件挂载到其他插件旁边，而各项注册都是副作用，会在其插件卸载时撤销。"*

- **Profile 与组合包**：运行中的 dsh 是一棵插件树，由启动时按序叠加的各层组合而成。`dsh.profile` 列出组合包，`dsh.bundle` 指向组合包的 patch 文件；patch 按 id 定位条目并替换其整个 config。`dsh --profile web --dump-config` 可查看实际配置树。
- **三层发行形态**：`dsh-base`（每个 profile 的第一层：模型适配器、工具、持久化、沙箱与审批策略、设置、凭据、遥测）、`dsh-web-app`（浏览器应用）、`dsh-headless`（一次性运行器，完全不带服务器）。

### 2.2 核心包与事件域

| 包 | 职责 | ctx 键 |
|----|------|--------|
| core/session | 仅追加的 SessionEvent 日志和内存存储（唯一真源） | ctx.sessions |
| core/system-prompt | 提示词片段与工具 schema 的组装 | ctx.systemPrompt |
| core/tools | 作用域化的工具注册表 + 带把关的执行流水线 | ctx.tools |
| core/agent | Agent 接口、活跃 agent 注册表、agent/* 事件 | ctx.agents |
| core/agent-loop | 实现该接口的默认驱动器 | ctx.agentLoop |
| core/scope | 按 agent 划分作用域的注册原语（库，无 ctx 键） | — |
| llm/llm | 消息与流式词汇表 + 适配器 seam | ctx.llm |

**事件就是扩展点**，分三个域：
- **会话事件**：追加到日志并经 `session/event` 广播的**持久事实**（reload 后仍存在）
- **Agent 事件**（agent/*）：携带活跃 Agent 句柄（inbox、步骤、状态、请求、验证、续跑）
- **能力事件**（fs/*、tools/*、telemetry/*）：无需导入循环即可向 seam 附加策略和适配器

### 2.3 轮次流程（Turn/Step 生命周期）

一个**步骤** = 一次模型请求 + 它调用的工具；一个**轮次**包含零个或多个步骤（领取首条输入时打开，不再欠工作时关闭）。

```
turn/start → agent/pre-step（可拒绝/改写）→ step/start
  → agent/request → llm/stream → assistant/chunk* → assistant/message
  → tool/call* → tools/pre-execute → tools/execute → tools/post-execute → tool/result*
  → step/end → agent/turn-stopping → turn/end
```

- `agent/pre-step`、`agent/request`、`llm/stream`、三个 `tools/*` 事件是 **waterfall**（监听器必须调用 next() 委托下去）；`agent/turn-stopping` 是 serial 事件
- 输入通过同一个 inbox 到达驱动器；注入的上下文留在 inbox 中直到另一条消息唤醒
- **模型可见即已记录**：抵达模型请求的一切都必须能从会话日志重建，由运行时不变量断言——新增模型可见输入就必须新增会话事件（扩展 SessionEventMap 并从日志渲染）

### 2.4 能力 Seam：可替换能力三角色

一个 **seam** = 声明接口的 Service Definition + 实现它的 Service Provider + 使用它的 Consumer（通常是面向模型的工具）。能力表列出约 60 个 ctx 键，标注角色（seam/core/bundle）、实现包、直接消费方：

| 代表性 seam | 实现包 | 说明 |
|------------|--------|------|
| ctx.llm | llm-deepseek, llm-pi-ai, llm-replay | 适配器注册提供方；agent loop 调用提供方无关的流服务 |
| ctx.sessionPersistence | session-persistence-jsonl, -sqlite | 各后端持久化同一套 SessionEvent 词汇，组合时选后端 |
| ctx.subprocess | subprocess-local, -e2b | Bash/PTY/LSP/ACP/Codex/Claude Code subagent 后端都通过它 spawn |
| ctx.shell | bash-local, bash-sandbox, pwsh-local | 沙箱/远程/PowerShell 执行器可替换 bash-local，消费方不改 |
| ctx.sandbox | sandbox-local + bash-sandbox/fs-sandbox | 消费方交出 argv；按每次调用的策略包装 |
| ctx.skills | skill-badge, skill-filesystem | 合并提供方技能目录；tool-skill 渲染会话前缀 |
| ctx.credentials | credentials-local | 配置携带机密引用；按操作解析，轮换后下一次请求即生效 |

**seam 的意义**（文档原话）："seam 正是替换一个提供方就能改变整个产品的原因。文件系统与进程提供方共享同一个执行世界，因此把它们指向远程沙箱，也就把 Bash、PTY 和 LSP 一并搬了过去，无需提供方专用 fork。"

### 2.5 Typert：类型化远程调用

- 业务对象包通过**声明合并**扩展两个空 map（TypertLookupMap / 作用域上下文声明），把 Host 对象类型与 wire identity 关联
- 生成的 Remote 描述符引用这些 key，运行时提供方提供活对象解析行为
- API 网关（ctx.typertGateway）把生成的 Remote 描述符与**实时 Cordis 服务**关联，通过共享的 Connection RPC 载体提供一元调用

### 2.6 运行时不变量注册表

- `ctx.invariants`：包自有运行时不变式检查的可配置注册表服务（选择逻辑、名称保留、子 fiber 生命周期、归因到包的失败）
- 每个工作区包发布一个 `./invariant` 配套插件，以自己确切的 npm 包名注册检查；无检查项的包必须导出以 "No runtime invariant:" 开头的空安装器并解释原因
- `pnpm run verify-package-invariants` 机械性拒绝：生成文件标记、无解释空安装器、遗漏报告器、错误注册名、不完整接线
- 注册与 disposer 绑定到 effect：卸载任一侧都移除监听器/状态/保留项，HMR 重载后无残留

### 2.7 子系统全景（30+）

| 分组 | 子系统 |
|------|--------|
| 内核与作用域 | 核心（packages/core）、作用域（按 agent 划分的注册原语）、运行时不变式 |
| 会话与持久化 | 会话、会话查询（sqlite 全文/排序/摘要）、会话引用、会话标题（LLM 生成）、会话投影、会话持久化、Spill 存储 |
| 遥测 | session-telemetry（otel 后端，脱敏后出进程） |
| 模型与上下文 | LLM 流式响应、Token 计量（按会话隔离折叠区）、系统提示词、上下文压缩（剪枝 + 摘要，可回放） |
| 执行与工具 | 工具、Bash 执行、子进程、PTY 会话、后台任务、文件系统（先读后编辑检查）、LSP 导航、代码运行时（Code Mode / run_code）、Web 访问 |
| 技能/工作流/子代理 | 技能（skill 目录合并）、工作流、子代理（inprocess / ACP / Codex / Claude Code 后端） |
| 策略与交互 | 审批（tools/pre-execute 前询问）、权限预设、沙箱（local / E2B 远程 Linux）、计划模式、用户交互（ask）、命令（无需模型轮次）、目标、定时提醒 |
| 平台与接入 | HTTP 服务器、Typert、客户端模块、存储（json/sqlite，领域优先 KV）、工作区、用户设置、用户凭据 |

## 3. 与 Cordis 论文的关系（关键发现）

| 论文概念 | dsh 生产实践 |
|---------|-------------|
| **无特权内核**（一切皆插件，可从配置替换） | "扩展 dsh 的方式是把插件挂载到其他插件旁边"——模型适配器、工具注册表、会话日志、agent loop 本身都是插件 |
| **可逆效应**（ctx.effect / 逆由组合推导） | 插件注册的一切（事件监听、工具、定时器）卸载时自动清理；手动资源用 `ctx.effect(() => ... return cleanup)`，无需手写 removeListener/clearInterval |
| **反应式 coeffects**（inject 声明依赖，依赖就绪才激活） | 插件用 `inject: ['tools', 'llm']` 声明服务依赖，框架保证依赖就绪后才 apply；加载顺序由服务依赖表达而非手动编排 |
| **fiber 生命周期 + HMR** | profile/组合包/cordis.patch.yml 分层组装插件树，热重载 |
| **组件树 / 层级组合**（Γ∞） | "运行中的 dsh 是一棵插件树，由启动时按序叠加的各层组合而成" |
| **Isolation（realm 隔离）** | agent preset 中的服务行需要 isolate realm 才能让不同会话拥有不同能力集合 |
| **Confluence（动态 = 静态）** | patch 按 id 定位条目并替换整个 config——最终配置树可由 --dump-config 确定性输出 |

**证据链**：
1. Cordis v4 README → 文档链接指向 dsh 文档站的 cordis-primer
2. cordis-primer 原话："Cordis 是 DeepSeek Harness 底层以 vendor 方式引入的插件框架"；vendor 源码与同步流程见私有仓库 `vendor/README.md`
3. dsh 开发教程 import `@deepseek-ai/cordis`——DeepSeek 将 Cordis 发布在自己的 npm scope 下
4. dsh 架构页：*"Cordis 是 dsh 底层的框架：插件向共享上下文贡献服务、类型化事件和可逆的副作用。产品的每一部分都是插件……因此每一部分都可以从配置替换。"*
5. 文档建议"使用 agent（智能体）探索代码库并理解其架构"——dogfooding 自家平台

**结论**：论文的 Koishi 案例（4000+ 聊天插件）是采用性/存在性证据；DeepSeek Harness 则是论文 §1.2.2 动机场景（自进化 agent harness）的直接生产实现。DeepSeek 用 Cordis 把"agent 自我修改的可恢复地基"落成了产品——这也解释了论文作者来自 DeepSeek-AI、以及 Cordis v4 独立于 Koishi 演进的原因。

## 4. 行业影响

- **Agent harness 的架构范式**：dsh 把"插件系统 = 可逆副作用 + 反应式依赖"的模型应用到整个 agent 平台，与主流 harness（如 OpenAI Codex CLI、Anthropic Claude Code 的钩子系统）的"硬编码功能 + 事件钩子"路线形成对比——dsh 的每个能力都是可替换 seam
- **子代理生态开放**：subagent 后端支持 inprocess / ACP / Codex / Claude Code——不锁定自家模型，反而桥接竞品 agent 作为子代理执行器
- **沙箱即 seam**：local 与 E2B 远程 Linux 两种后端共享同一接口，文件系统/进程/终端统一搬移，降低了"本地开发 → 远程沙箱"的迁移成本
- **运行时不变量工程化**：每个包强制配套 invariant 检查 + 机械验证脚本，是大型插件平台质量保证的可借鉴模式

## 5. 关键启示

1. **理论论文可以反哺产品架构**：Cordis 论文（2026 年中发布）的形式化与 dsh 文档站的架构语言高度同构（可逆副作用、反应式依赖、fiber 生命周期），说明 DeepSeek 是"先有工程直觉，论文补形式地基，再用形式化指导 v4 重构"的闭环。
2. **事件溯源是 agent 日志的正确抽象**："模型可见即已记录"这条不变量把会话日志变成唯一真源——回放、fork、恢复、transcript、遥测全部从事件流派生，这与论文对可逆性/可重建性的强调一脉相承。
3. **能力 seam 是 harness 可扩展性的关键设计**：替换一个提供方即可改变整个产品（本地→远程沙箱、换 LLM 适配器、换持久化后端），把"配置驱动"做到系统级。
4. **Cordis 的学术定位需要更新**：它不再是"给 Koishi 写理论"，而是 DeepSeek agent 平台的地基——对 Cordis 论文的理解应加上 dsh 这个生产参照物。

---

## 参考资料

- DeepSeek Harness 文档站：https://deepseek-harness.github.io/deepseek-harness/（中文，含 /en/ 英文版）
- Cordis 入门：https://deepseek-harness.github.io/deepseek-harness/reference/cordis-primer
- 架构：https://deepseek-harness.github.io/deepseek-harness/reference（架构 + 能力 Seams + Agent 生命周期 + Tool 执行）
- 关联论文分析：[[Knowledge/论文分析/软件系统/Cordis 时空可组合性编程范式 深度技术分析]]
