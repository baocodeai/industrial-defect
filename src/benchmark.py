import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from dataset import CLASS_NAMES
from train import build_mobilenetv2


def run_benchmark(model, onnx_path: str, num_threads: int, warmup: int, runs: int):
    torch.set_num_threads(num_threads)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    x_torch = torch.randn(1, 3, 224, 224)
    x_numpy = x_torch.numpy()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_torch)
        pt_lats = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x_torch)
            pt_lats.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(warmup):
        _ = session.run(None, {in_name: x_numpy})
    ort_lats = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, {in_name: x_numpy})
        ort_lats.append((time.perf_counter() - t0) * 1000.0)

    def stats(name, lats):
        arr = np.array(lats)
        return {
            "Engine": name,
            "Threads": num_threads,
            "Mean": float(arr.mean()),
            "P50": float(np.percentile(arr, 50)),
            "P95": float(np.percentile(arr, 95)),
            "P99": float(np.percentile(arr, 99)),
            "FPS": float(1000.0 / arr.mean()),
        }

    return stats("PyTorch Eager (CPU)", pt_lats), stats("ONNX Runtime Opt (CPU)", ort_lats)


def main():
    parser = argparse.ArgumentParser(description="CPU Production Benchmark: PyTorch vs ONNX Runtime")
    parser.add_argument("--weights", type=str, default="models/mobilenetv2_neu.pth")
    parser.add_argument("--onnx-model", type=str, default="models/model_opt.onnx")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    model = build_mobilenetv2(num_classes=len(CLASS_NAMES), pretrained=False).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))

    print(f"{'Engine':<24} | {'Thr':>3} | {'Mean(ms)':>9} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'P99(ms)':>8} | {'FPS':>7}")
    print("-" * 82)
    for t in args.threads:
        pt_r, ort_r = run_benchmark(model, args.onnx_model, t, args.warmup, args.runs)
        for r in [pt_r, ort_r]:
            print(f"{r['Engine']:<24} | {r['Threads']:>3} | {r['Mean']:>9.2f} | {r['P50']:>8.2f} | {r['P95']:>8.2f} | {r['P99']:>8.2f} | {r['FPS']:>7.1f}")
        print(f"=> Speedup ({t} thread(s)): {pt_r['Mean'] / ort_r['Mean']:.2f}x\n")


if __name__ == "__main__":
    main()
