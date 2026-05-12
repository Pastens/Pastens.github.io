# CANN Runtime CAModel 深度技术分析

> 项目：https://gitcode.com/cann/runtime
> 分析日期：2026-05-12
> 标签：#昇腾 #CANN #runtime #camodel #仿真器

## 一、整体架构概览

CANN (Compute Architecture for Neural Networks) Runtime 是华为昇腾 NPU 的运行时组件。其 CAModel（Computer Architecture Model）变体 `libruntime_camodel.so` 是 Runtime 的**纯软件仿真实现**，可以在无 NPU 硬件的机器上完整模拟 Runtime API 的行为。

### 构建关系

```
libruntime_camodel.so (output name: libruntime_camodel)
  ├── runtime_model (OBJECT library — 核心 Runtime 业务逻辑)
  │    编译条件: CFG_DEV_PLATFORM_PC, STATIC_RT_LIB=1, RUNTIME_API=0
  ├── runtime_platform_others / runtime_platform_910B
  └── 链接依赖:
        ├── npu_drv_camodel.so       ← 驱动仿真层
        ├── mmpa                      ← 跨平台抽象层
        ├── platform                  ← 平台适配
        ├── libcamodel.so             ← 指令级仿真引擎（闭源）
        ├── libtsch_camodel.so        ← 任务调度仿真（闭源）
        └── c_sec / json / ...        ← 通用库
```

### 与真实 Runtime 的差异

| 方面 | `libruntime.so` (上板) | `libruntime_camodel.so` (仿真) |
|------|------------------------|-------------------------------|
| 编译宏 | `CFG_DEV_PLATFORM_LINUX` | `CFG_DEV_PLATFORM_PC` |
| 驱动后端 | 真实 NPU 驱动 (ioctl / /dev/davinci*) | `npu_drv_camodel.so` (纯软件) |
| 芯片模型 | 无 | `libcamodel.so` (指令级模拟) |
| 运行时行为 | 下发 SQ → 硬件执行 | `libtsch_camodel.so` 调度模拟 |
| 运行要求 | 需要 NPU 卡 | 仅需 x86/arm Linux |

---

## 二、源代码构成分析

### 2.1 核心 Runtime 层 — `runtime_model` 对象库

源代码分布在 `src/runtime/` 下，按功能模块组织：

#### Runtime API (`src/runtime/api/`)
| 文件 | 职责 |
|------|------|
| `api.cc` | API 入口，`rtKernelLaunch` / `rtSetDevice` 等函数实现 |
| `api_impl.cc` | API 实现核心，调用 driver/device/kernel 等下层 |
| `api_impl_creator.cc` | API 实现工厂，创建具体实现对象 |
| `api_decorator.cc` | API 装饰器，用于 AclGraph 等扩展 |
| `api_error.cc` | 错误码管理 |

#### Device 管理 (`src/runtime/device/`)
| 文件 | 职责 |
|------|------|
| `device.cc` | 设备管理核心接口 |
| `raw_device.cc` | 底层设备，封装 driver 操作 |
| `raw_device_res_camodel.cc` | **CAModel 专用**：设备资源管理（在 PC 平台分配流/事件/任务 ID） |
| `ctrl_msg.cc` / `ctrl_sq.cc` | 控制消息/队列 |

#### Driver 层 (`src/runtime/driver/`)
| 文件 | 职责 |
|------|------|
| `driver.cc` | 通用 driver 接口，选择 v100/v200 后端 |
| `v100/npu_driver.cc` | Davinci V100 系列的 NPU 驱动 |
| `npu_driver_mem.cc` | 内存操作 |
| `npu_driver_queue.cc` | SQ/CQ 队列操作 |
| `npu_driver_res.cc` | 资源分配与释放 |
| `npu_driver_tiny.cpp` | Tiny 模式驱动 |
| `npu_driver_dcache_lock_*.cpp` | Dcache 锁管理 |

#### Kernel 管理 (`src/runtime/kernel/`)
| 文件 | 职责 |
|------|------|
| `kernel.cc` | Kernel 对象管理 |
| `module.cc` | 模块加载（二进制 .o 的加载与 symbol 解析） |
| `program.cc` | Program 对象（已注册的 kernel 集合） |
| `elf.cc` | ELF 二进制解析 |
| `v100/kernel.cc` | V100 特定 kernel 操作 |

#### 其他关键模块
| 模块 | 路径 | 职责 |
|------|------|------|
| Stream | `src/runtime/stream/` | Stream 管理（`rtStreamCreate` 等） |
| Event | `src/runtime/event/` | Event 管理（`rtEventCreate` 等） |
| Notify | `src/runtime/notify/` | Notify 机制 |
| Pool | `src/runtime/pool/` | 内存池（`rtMalloc` 等） |
| Task | `src/runtime/task/` | 任务调度（`rtKernelLaunch` 的任务下发） |
| Model | `src/runtime/feature/model/` | 模型管理 |
| Soma | `src/runtime/feature/soma/` | Stream 内存池 |

### 2.2 驱动仿真层 — `npu_drv_camodel.so`

源代码位于 `src/cmodel_driver/`：

| 文件 | 职责 |
|------|------|
| `driver_api.c` | 对外 API 声明，包括 `cmodelDrvMemcpy`、`drvDeviceOpen` |
| `driver_impl.c` | 驱动实现核心：设备打开/关闭、资源分配（流/事件/任务/SQ-CQ ID 分配） |
| `driver_mem.c` | 内存管理：设备内存的 malloc/free、memcpy（H2D/D2H）、memset |
| `driver_queue.c` | 队列管理：SQ 提交、CQ 轮询、调度循环 |

**关键实现** — 纯软件的驱动仿真：

- **设备打开** (`drvDeviceOpen`)：返回虚拟设备句柄，不操作 `/dev/davinci*`
- **资源分配** (`__drvIdAlloc`)：在内存中维护空闲 ID 列表（g_drvStreamIdList、g_drvEventIdList 等），纯内存操作
- **内存管理**：在 host 内存中模拟 device 内存布局，malloc/free/memcpy 全部在 host RAM 中完成
- **队列提交**：SQ 中的 task 不是通过硬件执行，而是通过 `libcamodel.so` 中的 `model_api.h` 接口发送给仿真引擎
- **中断模拟**：通过 `tsRegDrvReportIrqTriger` 注册中断回调，仿真引擎模拟硬件中断

### 2.3 指令级仿真引擎 — `libcamodel.so` / `libtsch_camodel.so`

这是**闭源部分**，但在构建系统中以依赖形式存在：

```
npu_drv_camodel.so
  ├── libcamodel.so        ← 芯片指令集模型 (AICore/AIVec 指令模拟)
  └── libtsch_camodel.so   ← 任务调度器模型 (SQ/CQ 调度模拟)
```

`libcamodel.so` 在旧版 module.mk 中被称为 `lib_pvmodel.so`，是真正的指令级仿真引擎，实现了昇腾芯片的指令解码、执行流水线、内存访问模拟等核心功能。

---

## 三、关键数据结构

### 3.1 cmodel_driver 的资源管理

```c
// 资源池大小
#define MAX_STREAM_NUM   (128)
#define MAX_EVENT_NUM    (256)
#define MAX_TASK_NUM     (1024)
#define MAX_SQCQ_NUM     (512)

// 空闲资源 ID 列表（纯软件位图）
int8_t g_drvStreamIdList[MAX_DEV_NUM][MAX_STREAM_NUM];
int8_t g_drvEventIdList[MAX_DEV_NUM][MAX_EVENT_NUM];
int8_t g_drvTaskpoolIdList[MAX_DEV_NUM][MAX_TASK_NUM];
int8_t g_drvSqCqIdList[MAX_DEV_NUM][MAX_SQCQ_NUM];
```

### 3.2 cmodel_driver 的内存管理

```c
// 设备内存池（在 host 内存中模拟）
typedef struct tagDrvMemManager {
    DvDataInfo devMemHead;     // 设备内存链表头
    uint64_t phyBase;          // 模拟物理地址基址
    pthread_mutex_t mutex;     // 线程安全
} DrvMemManager;
```

### 3.3 Runtime 层的 LaunchEvent

`LunchEvent` / `LaunchConfig` 结构体在 task 模块中定义，包含 kernel 启动所需的所有参数。

---

## 四、核心执行流程

### 4.1 `rtKernelLaunch` 在 CAModel 中的完整路径

```
rtKernelLaunch(stubFunc, blockDim, args, argsSize, smDesc, stm)
  │
  ├─ api.cc → api_impl.cc → 解析参数
  │
  ├─ kernel/v100/kernel.cc → 通过 stubFunc 查找 kernel 对象
  │
  ├─ task/task_info/davinci/davinci_kernel_task.cc
  │     → 构造 DaVinciKernelTask，填充 task 描述符
  │     → task.blockDim = blockDim
  │     → task.kernelHandle = stubFunc 对应 handle
  │     → task.argsAddr = args
  │     → task.smDesc = smDesc
  │
  ├─ task/task_submit/v100/task_submit.cc
  │     → 将 task 提交到 SQ (Submission Queue)
  │     → 调用 driver/npu_driver_queue.cc → 写入 SQ 槽位
  │
  ├─ driver/v100/npu_driver.cc → npu_drv_camodel.driver_queue.c
  │     → sq: 在内存中追加 task 描述符
  │     → 调用 libtsch_camodel 触发调度
  │
  └─ libcamodel.so (闭源)
        → 从 SQ 读取 task
        → 加载 kernel 二进制到模拟内存
        → 设置 blockDim 个 AICore 模拟核
        → 执行指令级仿真
        → 写回 CQ 完成状态
        → 触发模拟中断 → runtime 轮询到完成
```

### 4.2 `rtSetDevice` 在 CAModel 中

```
rtSetDevice(devId)
  │
  ├─ api.cc → 检查设备号
  │
  ├─ driver/v100/npu_driver.cc
  │     → npu_drv_camodel.driver_impl.c:drvDeviceOpen
  │        → 验证 devId (仅支持 0)
  │        → 分配设备上下文 (纯内存结构体)
  │        → 返回虚拟设备句柄 (仅用作索引)
  │
  └─ Return RT_ERROR_NONE
```

### 4.3 `rtMalloc` 在 CAModel 中

```
rtMalloc(devPtr, size, type)
  │
  ├─ api_impl.cc → 验证参数
  │
  ├─ pool/memory_pool.cc → 检查内存池
  │
  ├─ driver/v100/npu_driver.cc → npu_drv_camodel.driver_mem.c
  │     → drvMemAlloc: 在 host 申请 size 字节的虚拟内存
  │     → 返回模拟的设备地址 (基于 phyBase 的偏移)
  │     → 地址记录到 DrvMemManager 链表
  │
  └─ *devPtr = 模拟地址 (host 指针偏移)
```

### 4.4 `rtMemcpy` 在 CAModel 中

H2D、D2H、D2D 全部调用 `cmodelDrvMemcpy`：

```c
DVresult cmodelDrvMemcpy(DVdeviceptr dst, size_t destMax, 
                         DVdeviceptr src, size_t size, drvMemcpyKind_t kind)
{
    // H2D: src 是 host 指针，dst 是模拟 device 地址
    // D2H: src 是模拟 device 地址，dst 是 host 指针
    // 全部转换为普通 memcpy
    void *dstAddr = CmodelDevToVir(dst);  // 模拟地址 → 真实 host 指针
    void *srcAddr = CmodelDevToVir(src);  // 同上
    (void)memcpy(dstAddr, srcAddr, size); // 普通 memcpy!
    return DRV_ERROR_NONE;
}
```

---

## 五、三种产品形态对比

| 组件      | 真实硬件                    | PvModel (旧)                | CAModel (新)                |
| ------- | ----------------------- | -------------------------- | -------------------------- |
| 驱动      | `npu_drv.so` → 硬件 ioctl | `npu_drv_pvmodel.so` → 纯软件 | `npu_drv_camodel.so` → 纯软件 |
| 芯片模型    | 无                       | `lib_pvmodel.so` (性能模型)    | `libcamodel.so` (指令级模拟)    |
| 调度      | 硬件调度器                   | `libtsch.so` (模拟)          | `libtsch_camodel.so` (模拟)  |
| Runtime | `libruntime.so`         | `libruntime_cmodel.so`     | `libruntime_camodel.so`    |
| 适用场景    | 上板执行                    | 快速性能评估                     | 精确指令级仿真                    |
| 精度      | 真实执行                    | 近似                         | 指令级精确                      |

三种驱动模式由同一个源代码 `cmodel_driver/` 编译，通过不同的链接参数区分：
- `npu_drv` — 链接 `libtsch` + `lib_pvmodel`
- `npu_drv_pvmodel` — 同 `npu_drv` (CMakeLists.txt 编译同一个 SRC 但不同宏)
- `npu_drv_camodel` — 链接 `libtsch_camodel` + `libcamodel`

---

## 六、与 msOpProf 的集成方式

`libruntime_camodel.so` 被 msOpProf 的劫持注入层（`libmsopprof_injection.so`）作为 Runtime API 的**原始实现**加载：

```cpp
// BindOpprof.cpp - libmsopprof_injection.so 启动时
bool isSimulator = (GetEnv(IS_SIMULATOR_ENV) == "true");
if (isSimulator) {
    soName = "runtime_camodel";  // ← 指向 libruntime_camodel.so
} else {
    soName = "runtime";          // ← 指向 libruntime.so (真实硬件)
}
REGISTER_LIBRARY(soName);
REGISTER_FUNCTION(soName, rtKernelLaunch);
// ... 30+ Runtime API 全部绑定到 camodel 版本
```

而 `npu_drv_camodel.so` 在 `libruntime_camodel.so` 的加载时自动链接，提供纯软件的驱动仿真。

**完整的 4 层调用链**：

```
用户程序 → 被劫持的 rtKernelLaunch (injection)
  → 跳转 libruntime_camodel.so::rtKernelLaunch (CAModel Runtime)
    → npu_drv_camodel.so::driver_queue.c (软件队列)
      → libcamodel.so (指令级仿真引擎) + libtsch_camodel.so (调度器)
```

---

## 七、关键代码位置

| 模块 | 文件 |
|------|------|
| Runtime API 入口 | `src/runtime/api/api.cc` |
| Runtime API 实现 | `src/runtime/api_impl/api_impl.cc` |
| 设备管理 | `src/runtime/device/device.cc` |
| CAModel 设备资源 | `src/runtime/device/raw_device_res_camodel.cc` |
| Driver 抽象层 | `src/runtime/driver/driver.cc` |
| V100 驱动 | `src/runtime/driver/v100/npu_driver.cc` |
| Kernel 模块 | `src/runtime/kernel/kernel.cc` |
| Kernel ELF 加载 | `src/runtime/kernel/elf.cc` |
| Davinci Task | `src/runtime/task/task_info/davinci/davinci_kernel_task.cc` |
| Task 提交 | `src/runtime/task/task_submit/v100/task_submit.cc` |
| 内存池 | `src/runtime/pool/memory_pool_manager.cc` |
| Stream 管理 | `src/runtime/stream/` |
| Event 管理 | `src/runtime/event/` |

**CAModel 仿真层：**

| 模块 | 文件 | 许可证 |
|------|------|--------|
| cmodel_driver API | `src/cmodel_driver/driver_api.c` | CANN OSL v2 |
| cmodel_driver 实现 | `src/cmodel_driver/driver_impl.c` | CANN OSL v2 |
| cmodel_driver 内存 | `src/cmodel_driver/driver_mem.c` | CANN OSL v2 |
| cmodel_driver 队列 | `src/cmodel_driver/driver_queue.c` | CANN OSL v2 |
| CAModel 构建配置 | `src/runtime/cmake/cmodel.cmake` | CANN OSL v2 |

**开源与否总表：**

| 组件 | 开源 | 许可证 |
|------|------|--------|
| `libruntime_camodel.so` 源代码 | ✅ | CANN OSL v2 |
| `npu_drv_camodel.so` 源代码 | ✅ | CANN OSL v2 |
| `libcamodel.so` | ❌ | 闭源 |
| `libtsch_camodel.so` | ❌ | 闭源 |
