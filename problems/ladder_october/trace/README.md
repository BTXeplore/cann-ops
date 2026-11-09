# Trace 算子性能迭代

目标是 CANN 9.0.0 的 Ascend 910B。算子读取连续 ND 矩阵的主对角线，以 FP32 归约，输出形状 `(1,)`，并保持输入 dtype。两个版本只收录相对于 CANNJudge Trace 空工程的源码覆盖文件。

| 版本 | 实现 | 设备结果 |
| --- | --- | --- |
| `v1` | 单核，128 元素分块。跨步 DMA 后 Gather，每块归约后同步到标量端 | CANNJudge 15/15，15 点合计 371.61 μs |
| `v2` | 单核，512 元素分块。向量指令生成 Gather 偏移，块和留在 UB，最后只做一次 V→S 等待 | 待 NPU 编译和计时 |

两个版本在 KernelOpt 的 CPU 模拟中均通过 36/36 个用例。16384×16384 FP32 用例的模拟计数如下。

| 版本 | AIV 数量 | 输入 DMA 调用 | V→S 事件 |
| --- | ---: | ---: | ---: |
| `v1` | 1 | 128 | 128 |
| `v2` | 1 | 32 | 1 |

`sim/results.json` 保存逐例结果和计数。设备耗时以实机评测为准。

## 复现

下载并解压 Trace 空工程。[CANNJudge Trace 答题页](https://cannjudge.cn/public/ladder_october/trace/submit)

在本目录执行以下命令。

```text
python -m pip install ziglang==0.15.2
python sim/run.py /path/to/template/code --report trace-results.json
```

测试用替身编译实际 Host Tiling、形状推导及设备入口。每组重复五次，检查 FP16/FP32 数值、位级重复性、对角线元素恰好读取一次、工作区守卫、尾块和核数回退。若已安装 Zig，可以设置 `KERNELOPT_TRACE_ZIG` 指向可执行文件。

在 NPU 环境里验证时，先复制空工程，再把选定的 `versions/v*/` 下的 `op_host/` 和 `op_kernel/` 覆盖到副本的 `code/`。使用该工程的 CANN 构建流程。
