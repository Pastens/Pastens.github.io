# npu_drv_camodel 驱动仿真层深度技术分析

> 项目：https://gitcode.com/cann/runtime/tree/master/src/cmodel_driver
> 分析日期：2026-05-12
> 许可证：CANN Open Software License Agreement v2 (CANN OSL v2)

## 一、概述

`libnpu_drv_camodel.so` 是 Ascend NPU **设备驱动的纯软件仿真实现**。它模拟了真实 NPU 驱动层的核心功能——设备管理、资源分配、内存管理、任务队列调度——全部在 host 侧内存中用普通 POSIX 操作完成，**不接触任何 `/dev/davinci*` 或其他硬件设备节点**。

### 代码位置

```
cann/runtime/src/cmodel_driver/
├── cmodel_driver.h       ← 对外 API 头文件
├── driver_api.c          ← 驱动 API 实现 (964行)
├── driver_impl.c         ← 设备初始化/销毁、资源 ID 分配 (176行)
├── driver_impl.h         ← 资源管理常量和类型定义 (66行)
├── driver_mem.c          ← 设备内存模拟 (378行)
├── driver_mem.h          ← 内存管理常量和类型定义 (84行)
├── driver_queue.c        ← SQ/CQ 队列模拟 + 中断模拟 (236行)
├── driver_queue.h        ← 队列常量/结构体定义 (与 driver_api.c 同文件)
├── CMakeLists.txt
└── module.mk
```

### 依赖链

```
libruntime_camodel.so
  └─ 链接 npu_drv_camodel.so ←（我们在这里）
       ├─ 闭源: libcamodel.so (芯片指令级仿真引擎)
       └─ 闭源: libtsch_camodel.so (任务调度仿真)
```

---

## 二、核心功能分析

### 2.1 设备管理 — 纯虚拟操作

所有设备操作都是"空操作"或返回固定值：

```c
// 打开设备 → 执行初始化，返回成功
drvError_t drvDeviceOpen(void **devInfo, uint32_t deviceId)
{
    (void)drvDriverStubInit();    // 初始化全部数据结构
    return DRV_ERROR_NONE;
}

// 获取设备号 → 固定返回 1
drvError_t drvGetDevNum(uint32_t *num_dev)
{
    *num_dev = MAX_DEV_NUM;       // = 1
    return DRV_ERROR_NONE;
}

// 获取平台信息 → 固定返回 RUN_MODE_ONLINE = 1
drvError_t drvGetPlatformInfo(uint32_t *info)
{
    *info = RUN_MODE_ONLINE;
    return DRV_ERROR_NONE;
}
```

关键设计点：
- **设备 ID 只用索引号**，`DEVICE_HANDLE_TO_ID(X) = X`，无真实设备文件
- **P2P 全部空操作**：`halDeviceEnableP2P` / `halDeviceDisableP2P` / `halDeviceCanAccessPeer` 全部返回 `DRV_ERROR_NONE`

### 2.2 资源管理 — 内存位图分配

NPU 的资源（Stream、Event、Task、SQ/CQ）被实现为**静态内存数组 + 位图**：

```c
// driver_impl.h — 资源池大小定义
#define MAX_EVENT_NUM  1024
#define MAX_STREAM_NUM 1024
#define MAX_TASK_NUM   60000   // 云端芯片；非云端 32760
#define MAX_SQCQ_NUM   1024

// 四种资源的空闲 ID 位图
int8_t g_drvEventIdList[MAX_DEV_NUM][MAX_EVENT_NUM];
int8_t g_drvStreamIdList[MAX_DEV_NUM][MAX_STREAM_NUM];
int8_t g_drvTaskpoolIdList[MAX_DEV_NUM][MAX_TASK_NUM];
int8_t g_drvSqCqIdList[MAX_DEV_NUM][MAX_SQCQ_NUM];
```

**分配算法**（`__drvIdAlloc`）：
```
1. 遍历 g_drvXxxIdList[device][0..n]，找到第一个 0 值
2. 置为 1，返回索引作为 ID
3. 没找到就返回 DRV_ERROR_OUT_OF_MEMORY
```

**释放算法**（`__drvIdFree`）：
```
1. 验证 ID 范围合法性
2. 验证该 ID 已被分配（resList[id] == 1）
3. 置为 0
```

`halResourceIdAlloc` 是统一入口，通过 `in->type` 分发：
- `DRV_STREAM_ID` → `__drvIdAlloc(&id, devId, DRV_RES_STREAM)`
- `DRV_EVENT_ID` → `__drvIdAlloc(&id, devId, DRV_RES_EVENT)`
- `DRV_MODEL_ID` → 固定返回 1
- `DRV_NOTIFY_ID` → 固定返回 1

### 2.3 内存管理 — host malloc 模拟 HBM

"设备内存"（HBM, High Bandwidth Memory）完全在 host 侧用 `malloc` 模拟。

**模拟地址空间**：

```
HBM_BASE    = 0x10000000      (256 MB)
HBM_MAX_ADDR= 0x40000000      (1 GB)   V100 版本
              0x30000000      (768 MB)  V200 版本
MAX_ALLOC   = HBM_MAX_ADDR - HBM_BASE
```

**分配策略**（`drvMemAllocDeviceHBM`）— 双向链表 + 首次适应：

```c
typedef struct tagDrvMemNode {
    drvMemAttribute_t drvMemMgmtData;  // {size, address, status}
    struct tagDrvMemNode *prior;
    struct tagDrvMemNode *next;
} drvMemNode_t;
```

分配时遍历空闲块链表：
1. **精确匹配**：空闲块大小 == requestSize → 标记 BUSY，返回 `HBM_BASE + address`
2. **分割匹配**：空闲块大小 > requestSize → 新建节点占用前半部分，剩余部分保留
3. **512 字节对齐**（`tSize = ((tSize >> 9) + 1) << 9`）
4. 返回的是**模拟地址** `HBM_BASE + offset`，不是真实 host 指针

释放时合并相邻空闲块（`drvMergeDeviceHBM`）：
```
drvMergeDeviceHBM:
  1. 与前驱空闲块合并（地址连续）
  2. 与后继空闲块合并
  3. 若后继是 tail 节点 → 将空闲空间归还给 tail
```

**memcpy** — H2D/D2H 通过 `busDirectWrite` / `busDirectRead` 调用 `libcamodel.so` 的接口：

```c
drvError_t drvModelMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t size, drvMemcpyKind_t kind)
{
    switch (kind) {
        case DRV_MEMCPY_HOST_TO_DEVICE:
            // 将数据"写入"仿真的设备地址
            busDirectWrite(address, size, src, 0);   // → libcamodel.so
            break;
        case DRV_MEMCPY_DEVICE_TO_HOST:
            // 从仿真的设备地址"读取"数据
            busDirectRead(dst, size, address, 0);    // → libcamodel.so
            break;
        case DRV_MEMCPY_HOST_TO_HOST:
            memcpy_s(dst, destMax, src, size);        // 普通 memcpy
            break;
    }
}
```

**memset** — 类似逻辑，HBM 地址通过 `busDirectWrite` 逐字节写入，host 地址通过 `memset_s`。

### 2.4 队列管理 — 内存环形缓冲模拟 SQ/CQ

NPU 的硬件 SQ (Submission Queue) 和 CQ (Completion Queue) 被模拟为**内存环形缓冲**：

```c
// driver_queue.h — 队列结构体
typedef struct tagDrvQosQueue {
    uint32_t headIndex;
    uint32_t tailIndex;
    uint64_t taskCommand[DRV_QOS_QUEUE_SIZE];  // 任务命令数组
} drvQosQueue_t;

// 全局队列池
drvQosQueue_t  g_drvQosQueue[MAX_DEV_NUM][TS_TASK_CMD_QUEUE_PRIORITIES_LEVEL];
drvQosMgmt_t   g_drvQosQueueMgmt[MAX_DEV_NUM][TS_TASK_CMD_QUEUE_PRIORITIES_LEVEL];
drvReportQueue_t g_drvReportQueue[MAX_DEV_NUM];
```

**任务提交流程**（`halSqMsgSend` → `drvSetTaskCommand`）：
```
1. halSqMemGet → 分配 SQ 槽位
   - NORMAL 类型：从 g_drvQosQueue 取下一个 tailIndex 槽位，标记 IsOccupy
   - CALLBACK 类型：特殊回调队列

2. halSqMsgSend → 提交到调度器
   - 从 queue 中获取上一次放入的 command
   - 调用 drvQosHandleToId 获取 QoS ID
   - 调用 drvSetTaskCommand(deviceId, qos, queue, qMgmt)
   - drvSetTaskCommand 内部调用 libtsch_camodel.so 的 ts接口提交任务
```

**CQ 轮询**（`drvMoveTsReport`）：
```
1. 从 libtsch_camodel.so 获取任务报告队列 (ts_get_task_report_queue)
2. 将报告从 tsReport 拷贝到 drvReportQueue
3. 用信号量通知上层等待者 (drvSemPost)
```

### 2.5 中断模拟

`drvDriverStubInit` 中注册中断回调：

```c
drvError_t drvDriverStubInit(void)
{
    // ... 初始化各资源位图 ...

    // 注册中断触发器
    tsRegDrvReportIrqTriger(drvReportIrqTrigger);

    // 启动仿真引擎
    startModel(camodelLogPath, NULL, CHIP_NUM);  // → libcamodel.so

    // 启动任务调度器
    start_task_scheduler();                       // → libtsch_camodel.so
}
```

`drvReportIrqTrigger` 在仿真器完成一个 task 时被回调，触发 CQ 报告处理。

### 2.6 芯片型号适配

编译时通过 `-DPLATFORM_xxx` 宏选择芯片变体：

| 宏 | 芯片型号 | Model 版本 | 平台配置值 |
|----|---------|-----------|-----------|
| `PLATFORM_MINI_V1` | Ascend310, Ascend950 | V100 | `0x0` |
| `PLATFORM_MINI_V2` | Ascend610, Ascend310P | V200 | `0x10400` |
| `PLATFORM_CLOUD_V1` | Ascend910 | V100 | `0x100` |
| `PLATFORM_LHISI_ES` | hi3796cv300es/cs | V200 | `0x10301` |
| `PLATFORM_ADC_LITE` | Ascend610Lite | V310 | `0xB0C00` |
| `PLATFORM_MC62CM12A` | mc62cm12a | V200 | `0xE1000` |

这些值由 `halGetDeviceInfo` 返回给上层 Runtime。

---

## 三、完整执行流程

### 3.1 初始化阶段

```
drvDriverStubInit()
  ├─ drvEventIDListInit()       → 清空 Event 位图
  ├─ drvStreamIDListInit()      → 清空 Stream 位图
  ├─ drvTaskpoolIDListInit()    → 清空 Taskpool 位图
  ├─ drvSqCqIDListInit()        → 清空 SQ/CQ 位图
  ├─ drvQueueInit()             → 初始化所有队列数据
  ├─ drvMemMgmtInit()           → 初始化 HBM 内存链表
  ├─ tsRegDrvReportIrqTriger()  → 注册中断回调 → libtsch_camodel
  ├─ startModel(path, NULL, 1)  → 启动仿真引擎 → libcamodel
  └─ start_task_scheduler()     → 启动任务调度器 → libtsch_camodel
```

### 3.2 内存分配流程

```
rtMalloc(ptr, size, type)
  → runtime → npu_driver.cc → driver_api.c
    → halMemAlloc(pp, size, DRV_MEMORY_HBM)
      → drvMemAlloc(pp, size, DRV_MEMORY_HBM, 0)
        → drvMemAllocDeviceHBM(pp, tSize, deviceId)
           ① 512字节对齐
           ② 遍历 g_drvMemMgmtHead 链表找空闲块
           ③ 精确匹配或分割
           ④ 返回 HBM_BASE + address（模拟设备地址）
```

### 3.3 任务提交流程

```
rtKernelLaunch(stubFunc, blockDim, args, ...)
  → runtime → task_submit → npu_driver_queue.cc → driver_api.c
    → halSqMemGet(devId, in, out)
      ① 分配 SQ 槽位 (tailIndex)
      ② out->cmdPtr = &queue->taskCommand[tailIndex]
    
    → halSqMsgSend(devId, info)
      ① 获取提交的 command
      ② drvQosHandleToId → 获取 QoS ID
      ③ drvSetTaskCommand(deviceId, qos, queue, qMgmt)
         → 通过 libtsch_camodel 将 task 提交给仿真引擎
```

### 3.4 任务完成与结果读取

```
仿真引擎执行完毕
  → libcamodel 触发"中断"
    → drvReportIrqTrigger(deviceId)
      → drvSemPost(&sem)  // 通知 host 侧线程

Host 侧轮询
  → drvMoveTsReport(deviceId)
    ① 从 libtsch_camodel 读取完成报告 (ts_get_task_report_queue)
    ② 拷贝到 g_drvReportQueue
    ③ drvSemPost → 通知上层
  
  → rtStreamSynchronize(stm)
    返回执行结果
```

---

## 四、与真实驱动的对比

| 能力 | 真实驱动 | `npu_drv_camodel` |
|------|---------|-------------------|
| 设备接口 | `ioctl(/dev/davinciN)` | 纯函数调用，返回固定值 |
| 资源分配 | 硬件寄存器配置 | 内存位图 |
| 设备内存 | 真实 HBM 硬件 | host `malloc` |
| H2D/D2H | DMA 引擎 | `busDirectWrite` / `busDirectRead` → `libcamodel` |
| P2P | PCIe BAR 映射 | 空返回 |
| 中断 | MSI-X 硬件中断 | 信号量 + 回调函数 |
| SQ/CQ | 硬件队列寄存器 | 内存环形缓冲 |
| 缓存一致性 | 硬件 cache 一致性协议 | `drvFlushCache` 空操作 |
| 多设备 | 多卡真实硬件 | `MAX_DEV_NUM=1` 只支持单设备 |

---

## 五、关键观察

1. **511 行验证代码** — `driver_api.c` 中每函数前 3-5 行都是 `COND_RETURN_CMODEL` 参数校验，这是标准的内核驱动编程风格。

2. **内存模型是真实的** — HBM 地址范围、512 字节对齐、first-fit 分配策略与真实硬件行为完全一致，这让上层 Runtime 的无硬件仿真高度精确。

3. **中断和调度通过回调委托给闭源库** — `tsRegDrvReportIrqTriger`、`startModel`、`start_task_scheduler` 都是 `libcamodel.so` / `libtsch_camodel.so` 的接口。这个开源层负责"注册"和"编排"，实际仿真执行在闭源层。

4. **队列实现包含 QoS 优先级** — `g_drvQosQueue[MAX_DEV_NUM][TS_TASK_CMD_QUEUE_PRIORITIES_LEVEL]` 表明支持多优先级任务队列，优先级由 `libtsch_camodel.so` 的调度器实现。

5. **所有 UNUSED 参数都在说同一件事**—"这个参数在真实硬件上有意义，但在仿真模式下不需要"。例如 `drvFlushCache`、`halHostRegister`、`halDeviceEnableP2P` 等。

## 六、关键代码位置

| 功能 | 文件 | 函数 |
|------|------|------|
| 驱动初始化 | `driver_impl.c` | `drvDriverStubInit()` |
| 设备打开 | `driver_api.c` | `drvDeviceOpen()` |
| 资源 ID 分配 | `driver_impl.c` | `__drvIdAlloc()` |
| 设备内存分配 | `driver_mem.c` | `drvMemAllocDeviceHBM()` |
| 设备内存释放 | `driver_mem.c` | `drvFreeDeviceHBM()` |
| 内存拷贝 | `driver_mem.c` | `drvModelMemcpy()` |
| 内存清零 | `driver_mem.c` | `drvModelMemset()` |
| SQ 槽位获取 | `driver_api.c` | `halSqMemGet()` |
| SQ 消息提交 | `driver_api.c` | `halSqMsgSend()` |
| CQ 报告轮询 | `driver_queue.c` | `drvMoveTsReport()` |
| 芯片信息查询 | `driver_api.c` | `halGetDeviceInfo()` |
| 资源 ID 分配入口 | `driver_api.c` | `halResourceIdAlloc()` |
