import json
import os
from pathlib import Path
import subprocess
import sys
import time


project = Path(sys.argv[1]).resolve(strict=True)
base = Path(__file__).resolve().parent
build = Path("build").resolve()
build.mkdir(exist_ok=True)
compiler = [os.environ["KERNELOPT_TRACE_ZIG"]] if os.environ.get("KERNELOPT_TRACE_ZIG") else [sys.executable, "-m", "ziglang"]
env = {**os.environ, "ZIG_LOCAL_CACHE_DIR": str(build / "zig-local")}
env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(build / "zig-global"))
binary = build / ("trace-sim.exe" if os.name == "nt" else "trace-sim")
version = subprocess.check_output([*compiler, "version"], env=env, text=True).strip()
start = time.perf_counter_ns()
subprocess.run([
    *compiler, "c++", "-std=c++20", "-O2", "-fno-strict-aliasing", "-ffp-contract=off",
    "-I", str(base), "-I", str(project / "op_kernel"), "-I", str(project / "op_host"),
    str(base / "check.cpp"), "-o", str(binary),
], env=env, check=True)
(build / "compile.json").write_text(json.dumps({"zig": version, "host_compile_ns": time.perf_counter_ns() - start}), encoding="utf-8")
