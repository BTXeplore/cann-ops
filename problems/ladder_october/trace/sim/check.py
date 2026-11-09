import json
import os
from pathlib import Path
import subprocess


build = Path("build").resolve()
binary = build / ("trace-sim.exe" if os.name == "nt" else "trace-sim")
meta = json.loads((build / "compile.json").read_text(encoding="utf-8"))
output = subprocess.check_output([str(binary)], text=True, timeout=30)
scenarios = []
checks = []
for line in output.splitlines():
    (sid, rows, columns, dtype, correct, stable, ref, actual, reads, off_diagonal, kernel_ok, shape_ok,
     tiling_ok, output_bits, blocks, tile, workspace, input_dma, max_input_dma, vector_to_scalar,
     *samples) = line.split("\t")
    scenarios.append({"id": sid, "device": "cpu", "samples_ns": [int(x) for x in samples]})
    checks.append({"id": sid, "shape": [int(rows), int(columns)], "dtype": dtype, "output_shape_is_1": shape_ok == "1",
                   "host_tiling_passed": tiling_ok == "1", "kernel_passed": kernel_ok == "1",
                   "passed": correct == "1", "bitwise_repeatable": stable == "1", "expected": ref,
                   "actual": actual, "output_bits": int(output_bits), "reads": int(reads),
                   "read_off_diagonal": off_diagonal == "1", "aiv_blocks": int(blocks), "tile": int(tile),
                   "workspace_bytes": int(workspace), "input_dma_calls": int(input_dma),
                   "max_input_dma_calls_per_block": int(max_input_dma), "vector_to_scalar_events": int(vector_to_scalar)})
print(json.dumps({
    "measurement_kind": "simulation", "correct": bool(checks) and all(x["passed"] for x in checks),
    "environment": {"toolchain": f"zig {meta['zig']}", "device": "CPU (Ascend C API test doubles)",
                    "target_backend": "ascend910b", "timer": "host steady_clock (simulation only)"},
    "host_compile_ns": meta["host_compile_ns"], "scenarios": scenarios, "checks": checks,
}, allow_nan=False))
