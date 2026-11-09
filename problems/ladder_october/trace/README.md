# Trace 算子性能迭代

目标是 CANN 9.0.0 的 Ascend 910B。算子读取连续 ND 矩阵的主对角线，以 FP32 归约，输出形状 `(1,)`，并保持输入 dtype。三个版本只收录相对于 CANNJudge Trace 空工程的源码覆盖文件。

| 版本 | 实现 | 设备结果 |
| --- | --- | --- |
| `v1` | 单核，128 元素分块。跨步 DMA 后 Gather，每块归约后同步到标量端 | CANNJudge 15/15，15 点合计 371.61 μs |
| `v2` | 单核，512 元素分块。向量指令生成 Gather 偏移，块和留在 UB，最后只做一次 V→S 等待 | 待 NPU 编译和计时 |
| `v3` | 对四个及以上的 tile 按连续区间分核。AIV 核在互不重叠的 GM 槽写入块和，核 0 按与 `v2` 相同的顺序汇总 | 待 NPU 编译和计时 |

`v3` 通过 `GetCoreNumAiv()` 限制核数，短输入和只有一个可用核的设备走单核路径。并行路径使用纯 Vector 的 `KERNEL_TYPE_MIX_AIV_1_0`、硬件 `SyncAll` 和 `SetScheduleMode(1)`。每个 tile 预留独立的 32 字节槽，最多使用 1 KiB 用户工作区。上板后需要验证调度和栅栏的行为。[SyncAll](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html) · [SetScheduleMode](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/basicdataapi/atlasopapi_07_00248.html)

三个版本在 KernelOpt 的 CPU 模拟中均通过 36/36 个用例。`v2` 和 `v3` 的所有输出逐位一致。16384×16384 FP32 用例的实际模拟调用计数如下。

| 版本 | AIV 数量 | 输入 DMA 调用总数 | 单核最多输入 DMA 调用 | V→S 事件 | 用户工作区 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v1` | 1 | 128 | 128 | 128 | 0 B |
| `v2` | 1 | 32 | 32 | 1 | 0 B |
| `v3` | 16 | 32 | 2 | 0 | 1024 B |

`v3` 的并行路径另有写入块和、跨核栅栏以及核 0 读回块和的开销。这些计数来自 CPU 模拟，设备耗时以实机评测为准。`sim/results.json` 保存逐例结果和计数。

## 复现

下载并解压 Trace 空工程。[CANNJudge Trace 答题页](https://cannjudge.cn/public/ladder_october/trace/submit)

在本目录执行以下命令。

```text
python -m pip install ziglang==0.15.2
python sim/run.py /path/to/template/code --report trace-results.json
```

测试用替身编译实际 Host Tiling、形状推导及设备入口。每组重复五次，检查 FP16/FP32 数值、位级重复性、对角线元素恰好读取一次、工作区守卫、尾块和核数回退。跨核同步用线程栅栏模拟，计数用于比较执行路径。若已安装 Zig，可以设置 `KERNELOPT_TRACE_ZIG` 指向可执行文件。

在 NPU 环境里验证时，先复制空工程，再把选定的 `versions/v*/` 下的 `op_host/` 和 `op_kernel/` 覆盖到副本的 `code/`。使用该工程的 CANN 构建流程。
