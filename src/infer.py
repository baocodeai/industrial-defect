import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from dataset import CLASS_NAMES, get_transforms


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Single-image real-time ONNX inference for steel defect classification")
    parser.add_argument("--image", type=str, required=True, help="Path to input steel surface image")
    parser.add_argument("--onnx-model", type=str, default="models/model_opt.onnx")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(args.onnx_model, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    _, eval_tf = get_transforms(img_size=224)
    img = Image.open(args.image).convert("RGB")
    tensor = eval_tf(img).unsqueeze(0).numpy()

    # Warm-up 1 run
    _ = session.run(None, {in_name: tensor})

    t0 = time.perf_counter()
    logits = session.run(None, {in_name: tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax_np(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx]) * 100.0

    print(f"Image      : {args.image}")
    print(f"Prediction : {pred_cls} ({conf:.2f}%)")
    print(f"Latency    : {latency_ms:.2f} ms ({1000.0 / latency_ms:.1f} FPS)")


if __name__ == "__main__":
    main()
