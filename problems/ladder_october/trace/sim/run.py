"""Reproduce Trace correctness and logical work counters with the official template."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run_variant(template: Path, overlay: Path, sim: Path, work: Path, env: dict[str, str]) -> dict:
    project = work / "code"
    for name in ("op_host", "op_kernel"):
        shutil.copytree(template / name, project / name)
        shutil.copytree(overlay / name, project / name, dirs_exist_ok=True)
    subprocess.run([sys.executable, str(sim / "compile.py"), str(project)], cwd=work, env=env, check=True)
    result = subprocess.run([sys.executable, str(sim / "check.py")], cwd=work, check=True, capture_output=True, text=True)
    checked = json.loads(result.stdout)
    return {"version": overlay.name, "correct": checked["correct"], "cases": checked["checks"]}


def main() -> None:
    sim = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="code directory from the CANNJudge Trace template")
    parser.add_argument("--variant", action="append", choices=("v1", "v2", "v3"), help="repeat to select specific versions")
    parser.add_argument("--report", type=Path, help="save the correctness and operation-count report as JSON")
    args = parser.parse_args()
    template = args.template.resolve(strict=True)
    if not (template / "op_kernel" / "trace_tiling_key.h").is_file():
        parser.error("template must be the Trace code directory containing op_host and op_kernel")

    variants = sim.parent / "versions"
    selected = args.variant or sorted(p.name for p in variants.iterdir() if p.is_dir())
    reports = {}
    with tempfile.TemporaryDirectory(prefix="trace-") as work:
        root = Path(work)
        env = os.environ.copy()
        env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(root / "zig-cache"))
        for version in dict.fromkeys(selected):
            if not (variants / version).is_dir():
                parser.error(f"{version} is not present in this checkout")
            reports[version] = run_variant(template, variants / version, sim, root / version, env)
            cases = reports[version]["cases"]
            print(f"{version}: {sum(case['passed'] for case in cases)}/{len(cases)} CPU simulation cases passed")

    identical = None
    if "v2" in reports and "v3" in reports:
        def output_bits(version: str) -> dict[str, int]:
            return {case["id"]: case["output_bits"] for case in reports[version]["cases"]}

        identical = output_bits("v2") == output_bits("v3")
        print(f"v2/v3 output bits match: {identical}")
    if args.report:
        data = {"measurement_kind": "simulation", "versions": list(reports.values()), "v2_v3_bitwise_equal": identical}
        args.report.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if identical is False or not all(report["correct"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
