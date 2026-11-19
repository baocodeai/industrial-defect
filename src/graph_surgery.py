import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import onnx
from onnxsim import simplify
import torch
import torch.nn as nn

from dataset import CLASS_NAMES
from train import build_mobilenetv2


def verify_manual_bn_folding(model: nn.Module) -> float:
    """Kiểm chứng toán học gập cặp Conv2d + BatchNorm2d đầu tiên bằng NumPy."""
    conv_layer = model.features[0][0]
    bn_layer = model.features[0][1]

    W = conv_layer.weight.detach().numpy()
    b = conv_layer.bias.detach().numpy() if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)

    gamma = bn_layer.weight.detach().numpy()
    beta = bn_layer.bias.detach().numpy()
    mean = bn_layer.running_mean.detach().numpy()
    var = bn_layer.running_var.detach().numpy()
    eps = bn_layer.eps

    scale = gamma / np.sqrt(var + eps)
    W_fold = W * scale.reshape(-1, 1, 1, 1)
    b_fold = (b - mean) * scale + beta

    fused_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        groups=conv_layer.groups,
        bias=True,
    ).eval()

    with torch.no_grad():
        fused_conv.weight.copy_(torch.from_numpy(W_fold))
        fused_conv.bias.copy_(torch.from_numpy(b_fold))
        x = torch.randn(1, 3, 224, 224)
        y_sep = bn_layer(conv_layer(x))
        y_fused = fused_conv(x)

    max_diff = (y_sep - y_fused).abs().max().item()
    assert torch.allclose(y_sep, y_fused, rtol=1e-4, atol=1e-5), f"BN Folding diff too large: {max_diff:.2e}"
    return max_diff


def count_ops(onnx_path: Path):
    m = onnx.load(str(onnx_path))
    return len(m.graph.node), Counter(n.op_type for n in m.graph.node)


def main():
    parser = argparse.ArgumentParser(description="ONNX Export, BatchNorm Folding & Graph Simplification")
    parser.add_argument("--weights", type=str, default="models/mobilenetv2_neu.pth")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unfolded_path = out_dir / "model_unfolded.onnx"
    raw_path = out_dir / "model_raw.onnx"
    opt_path = out_dir / "model_opt.onnx"

    model = build_mobilenetv2(num_classes=len(CLASS_NAMES), pretrained=False).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))

    max_diff = verify_manual_bn_folding(model)
    print(f"Manual BN Folding (NumPy) passed | Max diff: {max_diff:.2e}")

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, dummy_input, str(unfolded_path),
        export_params=True, opset_version=args.opset, do_constant_folding=False,
        training=torch.onnx.TrainingMode.PRESERVE,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    torch.onnx.export(
        model, dummy_input, str(raw_path),
        export_params=True, opset_version=args.opset, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(raw_path))
    onnx.checker.check_model(onnx_model)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model failed validation!"
    onnx.save(model_simp, str(opt_path))

    n_unf, ops_unf = count_ops(unfolded_path)
    n_raw, ops_raw = count_ops(raw_path)
    n_opt, ops_opt = count_ops(opt_path)

    print(f"\n{'Operator':<22} | {'Unfolded':>10} | {'Raw (EVAL)':>10} | {'Optimized':>10}")
    print("-" * 62)
    for op in sorted(set(ops_unf) | set(ops_raw) | set(ops_opt)):
        print(f"{op:<22} | {ops_unf.get(op, 0):>10} | {ops_raw.get(op, 0):>10} | {ops_opt.get(op, 0):>10}")
    print("-" * 62)
    print(f"{'TOTAL NODES':<22} | {n_unf:>10} | {n_raw:>10} | {n_opt:>10}")
    print(f"{'FILE SIZE (MB)':<22} | {os.path.getsize(unfolded_path)/1024**2:>10.2f} | {os.path.getsize(raw_path)/1024**2:>10.2f} | {os.path.getsize(opt_path)/1024**2:>10.2f}")


if __name__ == "__main__":
    main()
