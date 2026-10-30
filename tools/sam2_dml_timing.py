"""Time an exported SAM2 image encoder under ONNX Runtime's DirectML provider, the runtime the
MikanXR editor runs its models on.

    .venv_sam2_dml/Scripts/python tools/sam2_dml_timing.py <encoder.onnx> [...] [--iters 50] [--warmup 10]
        [--cpu] [--check] [--out out/sam2_timing]

Runs each graph on a random image of its exported size, the outputs copied back to the host as a
frame would need: median and p90 ms per run under DmlExecutionProvider, and under the CPU provider
too with --cpu. --check also runs the CPU provider once and reports the largest difference against
DirectML, which stands in for the torch comparison when onnxruntime is absent from the torch venv.
Appends its lines to <out>/summary.txt and prints them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def timed(session: ort.InferenceSession, feed: dict, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        session.run(None, feed)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graphs", type=Path, nargs="+")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sam2_timing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    providers = ort.get_available_providers()
    lines = [f"onnxruntime {ort.__version__}, providers {providers}, {args.iters} iterations after {args.warmup} warm-up, median / p90"]
    for graph in args.graphs:
        if "DmlExecutionProvider" not in providers:
            lines.append(f"{graph.name}: DmlExecutionProvider not available")
            continue
        dml = ort.InferenceSession(str(graph), providers=["DmlExecutionProvider"])
        inp = dml.get_inputs()[0]
        shape = [1 if not isinstance(d, int) else d for d in inp.shape]
        feed = {inp.name: np.random.default_rng(0).standard_normal(shape).astype(np.float32)}
        med, p90 = timed(dml, feed, args.iters, args.warmup)
        line = f"{graph.name}: dml {med:7.1f} / {p90:7.1f} ms   input {shape}"
        if args.cpu or args.check:
            cpu = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
            if args.check:
                a, b = dml.run(None, feed), cpu.run(None, feed)
                worst = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
                line += f"   dml vs cpu max abs diff {worst:.3e}"
            if args.cpu:
                cmed, cp90 = timed(cpu, feed, max(args.iters // 10, 3), 2)
                line += f"   cpu {cmed:7.1f} / {cp90:7.1f} ms"
        lines.append(line)
        print(line)

    with open(args.out / "summary.txt", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
