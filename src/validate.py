import argparse

import numpy as np
import onnxruntime as ort
import torch

from dataset import CLASS_NAMES, get_dataloaders
from train import build_mobilenetv2


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Numerical Validation: PyTorch vs ONNX Runtime")
    parser.add_argument("--data-root", type=str, default="data/NEU-DET")
    parser.add_argument("--weights", type=str, default="models/mobilenetv2_neu.pth")
    parser.add_argument("--onnx-model", type=str, default="models/model_opt.onnx")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders(args.data_root, batch_size=32, num_workers=2, seed=args.seed)

    model = build_mobilenetv2(num_classes=len(CLASS_NAMES), pretrained=False).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.onnx_model, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    all_labels, pt_logits_list, ort_logits_list = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            pt_logits_list.append(model(imgs).numpy())
            ort_logits_list.append(session.run(None, {in_name: imgs.numpy()})[0])
            all_labels.append(lbls.numpy())

    pt_logits = np.concatenate(pt_logits_list, axis=0)
    ort_logits = np.concatenate(ort_logits_list, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    pt_probs = softmax_np(pt_logits)
    ort_probs = softmax_np(ort_logits)

    logit_diff = np.abs(pt_logits - ort_logits)
    prob_diff = np.abs(pt_probs - ort_probs)

    pt_preds = pt_logits.argmax(axis=1)
    ort_preds = ort_logits.argmax(axis=1)

    pt_acc = (pt_preds == y_true).mean() * 100
    ort_acc = (ort_preds == y_true).mean() * 100
    agreement = (pt_preds == ort_preds).mean() * 100

    print(f"=== NUMERICAL VALIDATION ({len(y_true)} TEST IMAGES) ===")
    print(f"Logits  -> Max diff: {logit_diff.max():.2e} | Mean diff: {logit_diff.mean():.2e}")
    print(f"Softmax -> Max diff: {prob_diff.max():.2e} | Mean diff: {prob_diff.mean():.2e}")
    print(f"PyTorch Accuracy      : {pt_acc:.2f}%")
    print(f"ONNX Runtime Accuracy : {ort_acc:.2f}%")
    print(f"Top-1 Agreement       : {agreement:.2f}%")

    assert prob_diff.max() < 1e-5, f"Softmax max diff >= 1e-5: {prob_diff.max():.2e}"
    assert agreement == 100.0, "Top-1 predictions do not match 100%!"
    print("Validation PASSED!")


if __name__ == "__main__":
    main()
