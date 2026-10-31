"""Time exported SAM2 ONNX graphs, encoder and decoder, under ONNX Runtime's DirectML provider, the
runtime the MikanXR editor runs its models on.

    .venv_sam2_dml/Scripts/python tools/sam2_dml_timing.py <graph.onnx> [...] [--iters 50] [--warmup 10]
        [--cpu] [--check] [--fp16] [--iobinding] [--points 8] [--image <png>]
        [--opt-level all|extended|basic|disable] [--out out/sam2_timing]

Feeds every input of each graph from its declared shape and type. A free dim named num_points takes
--points and any other free dim takes 1. Integer inputs are filled with 1, which marks every point
positive. Inputs named point_coords get uniform values in [0, 1024). An input named image gets the
frame given by --image, read with cv2, converted to RGB, resized to the graph's spatial size, scaled
to [0, 1] and laid out NCHW. Every other input gets random normal values. Random noise is not
representative for fp16 range checks, which is why the real frame is the default.

Reports median and p90 ms per run under DmlExecutionProvider, with the outputs copied back to the
host as a frame would need. --cpu also times the CPU provider. --check also runs the CPU provider
once and reports the largest difference against DirectML, which stands in for the torch comparison
when onnxruntime is absent from the torch venv.

--fp16 first constant-folds each float32 graph offline with a CPU session at basic optimization, so
constant chains such as the positional embedding resize are not left to run on the CPU every call. It
then converts the folded graph with onnxruntime.transformers.float16 (keep_io_types, so the inputs and
outputs stay float32), saves it beside the original as <stem>_fp16.onnx, times it under DirectML and
reports per output the largest absolute difference against the float32 graph and that difference
divided by the float32 output's value range, on the same feed.

--iobinding applies to a graph that has an image_embed input, a decoder. It times the decoder two
ways. The plain run copies the three feature tensors from the host to the device on every call. The
bound run uploads them once as DirectML OrtValues, binds the point inputs from the host and the
outputs to the host, then runs with io_binding and copies the outputs back. It reports both medians
and p90s and the largest difference between their outputs. If the runtime refuses a DirectML
OrtValue the line says why.

When the arguments include both an encoder (an input named image) and a decoder (an input named
image_embed), a frame ms line per encoder gives the encoder median plus 1, 4 and 7 times the decoder
median. That is the per-frame segmenter cost at 1, 4 and 7 contour runs. It covers the plain and
the bound decoder when measured, and the fp16 encoder when converted.

--opt-level sets the graph optimization level of every session, all by default. If an fp16 graph
fails to initialize under DirectML at the requested level, it is retried once with optimizations
disabled and its line ends with (opt disabled), so a reader sees that its timing carries no graph
optimizations.

Appends its lines to <out>/summary.txt and prints them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

OPT_LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}
FEATURE_INPUTS = ("image_embed", "high_res_feats")
TWO_GB = 2 * 1024 * 1024 * 1024


def load_frame(path: Path, height: int, width: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return np.ascontiguousarray(chw[None])


def build_feed(session: ort.InferenceSession, points: int, image: Path | None) -> dict:
    rng = np.random.default_rng(0)
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else (points if d == "num_points" else 1) for d in inp.shape]
        dtype = ONNX_DTYPES[inp.type]
        if inp.name == "image" and image is not None and np.issubdtype(dtype, np.floating) and len(shape) == 4:
            feed[inp.name] = load_frame(image, shape[2], shape[3]).astype(dtype)
        elif np.issubdtype(dtype, np.integer) or dtype == np.bool_:
            feed[inp.name] = np.ones(shape, dtype=dtype)
        elif inp.name == "point_coords":
            feed[inp.name] = rng.uniform(0.0, 1024.0, shape).astype(dtype)
        else:
            feed[inp.name] = rng.standard_normal(shape).astype(dtype)
    return feed


def make_session(graph: Path, provider: str, opt_level: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = OPT_LEVELS[opt_level]
    return ort.InferenceSession(str(graph), sess_options=options, providers=[provider])


def timed(call, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def max_abs_diff(a: list, b: list) -> float:
    return max(float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(a, b))


def convert_fp16(graph: Path) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    # Constant-fold offline first. The fp16 converter leaves constant chains such as the positional
    # embedding Resize in place, and ONNX Runtime would then run them on the CPU every call.
    folded_path = graph.with_name(graph.stem + "_folded.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded_path)
    ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    try:
        model = onnx.load(str(folded_path))
    finally:
        folded_path.unlink(missing_ok=True)
    converted = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    target = graph.with_name(graph.stem + "_fp16.onnx")
    try:
        oversize = converted.ByteSize() > TWO_GB
    except Exception:
        oversize = True
    if oversize:
        onnx.save_model(converted, str(target), save_as_external_data=True, location=target.name + ".data")
    else:
        onnx.save_model(converted, str(target))
    return target


def fp16_report(fp32_session: ort.InferenceSession, fp16_session: ort.InferenceSession, feed: dict) -> list[str]:
    names = [o.name for o in fp32_session.get_outputs()]
    ref = fp32_session.run(None, feed)
    out = fp16_session.run(None, feed)
    parts = []
    for name, x, y in zip(names, ref, out):
        if not np.issubdtype(x.dtype, np.floating):
            continue
        x64, y64 = x.astype(np.float64), y.astype(np.float64)
        diff = float(np.max(np.abs(x64 - y64)))
        span = float(np.max(x64) - np.min(x64))
        rel = diff / span if span > 0 else float("nan")
        parts.append(f"{name} max abs {diff:.3e} rel {rel:.3e}")
    return parts


def iobinding_times(session: ort.InferenceSession, feed: dict, iters: int, warmup: int):
    """Returns the plain (median, p90) and either the bound result ((median, p90), diff) or a reason string."""
    plain = timed(lambda: session.run(None, feed), iters, warmup)
    plain_out = session.run(None, feed)
    try:
        binding = session.io_binding()
        for name, value in feed.items():
            if name.startswith(FEATURE_INPUTS):
                binding.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(value, "dml", 0))
            else:
                binding.bind_cpu_input(name, value)
        for out in session.get_outputs():
            binding.bind_output(out.name, "cpu")

        def bound_run():
            session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()

        bound = timed(bound_run, iters, warmup)
        diff = max_abs_diff(plain_out, bound_run())
    except Exception as exc:
        return plain, f"{type(exc).__name__}: {exc}"
    return plain, (bound, diff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graphs", type=Path, nargs="+")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--iobinding", action="store_true")
    parser.add_argument("--opt-level", choices=list(OPT_LEVELS), default="all")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--image", type=Path, default=ROOT / "out" / "live_occlusion" / "angle_01" / "video.png")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sam2_timing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    providers = ort.get_available_providers()
    lines = [f"onnxruntime {ort.__version__}, providers {providers}, opt level {args.opt_level}, {args.iters} iterations after {args.warmup} warm-up, median / p90"]
    encoders = []  # (label, median ms)
    decoders = []  # (label, median ms)

    def emit(line: str) -> None:
        lines.append(line)
        print(line)

    for graph in args.graphs:
        if "DmlExecutionProvider" not in providers:
            emit(f"{graph.name}: DmlExecutionProvider not available")
            continue
        dml = make_session(graph, "DmlExecutionProvider", args.opt_level)
        input_names = [i.name for i in dml.get_inputs()]
        is_encoder = "image" in input_names
        is_decoder = "image_embed" in input_names
        feed = build_feed(dml, args.points, args.image)
        shapes = {k: list(v.shape) for k, v in feed.items()}
        med, p90 = timed(lambda: dml.run(None, feed), args.iters, args.warmup)
        line = f"{graph.name}: dml {med:7.1f} / {p90:7.1f} ms   inputs {shapes}"
        if args.cpu or args.check:
            cpu = make_session(graph, "CPUExecutionProvider", args.opt_level)
            if args.check:
                line += f"   dml vs cpu max abs diff {max_abs_diff(dml.run(None, feed), cpu.run(None, feed)):.3e}"
            if args.cpu:
                cmed, cp90 = timed(lambda: cpu.run(None, feed), max(args.iters // 10, 3), 2)
                line += f"   cpu {cmed:7.1f} / {cp90:7.1f} ms"
        emit(line)
        if is_encoder:
            encoders.append((graph.name, med))
        if is_decoder:
            decoders.append((f"{graph.name} host-copy", med))

        if args.iobinding and is_decoder:
            plain, result = iobinding_times(dml, feed, args.iters, args.warmup)
            if isinstance(result, str):
                emit(f"{graph.name}: iobinding plain {plain[0]:7.1f} / {plain[1]:7.1f} ms   bound run failed: {result}")
            else:
                (bmed, bp90), diff = result
                emit(
                    f"{graph.name}: iobinding plain {plain[0]:7.1f} / {plain[1]:7.1f} ms"
                    f"   bound (dml features) {bmed:7.1f} / {bp90:7.1f} ms   plain vs bound max abs diff {diff:.3e}"
                )
                decoders.append((f"{graph.name} iobinding", bmed))

        if args.fp16 and not graph.stem.endswith("_fp16"):
            try:
                target = convert_fp16(graph)
                suffix = ""
                try:
                    half = make_session(target, "DmlExecutionProvider", args.opt_level)
                except Exception:
                    if args.opt_level == "disable":
                        raise
                    half = make_session(target, "DmlExecutionProvider", "disable")
                    suffix = " (opt disabled)"
                hmed, hp90 = timed(lambda: half.run(None, feed), args.iters, args.warmup)
                emit(f"{target.name}: dml {hmed:7.1f} / {hp90:7.1f} ms   fp16 vs fp32: {'; '.join(fp16_report(dml, half, feed))}{suffix}")
                if is_encoder:
                    encoders.append((target.name + suffix, hmed))
            except Exception as exc:
                emit(f"{graph.name}: fp16 failed: {type(exc).__name__}: {exc}")

    if encoders and decoders:
        for enc_name, enc_med in encoders:
            for dec_name, dec_med in decoders:
                cells = "   ".join(f"{n} runs {enc_med + n * dec_med:7.1f}" for n in (1, 4, 7))
                emit(f"frame ms: {enc_name} ({enc_med:.1f}) + decoder {dec_name} ({dec_med:.1f}): {cells}")

    with open(args.out / "summary.txt", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
