import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torchvision import models

from dataset import CLASS_NAMES, get_dataloaders


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mobilenetv2(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, test_acc: float, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="Ground Truth (Nhãn thật)",
        xlabel="Predicted (Dự đoán)",
        title=f"Confusion Matrix — MobileNetV2 (Test Acc: {test_acc*100:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on NEU Surface Defect")
    parser.add_argument("--data-root", type=str, default="data/NEU-DET")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", type=str, default="models/mobilenetv2_neu.pth")
    parser.add_argument("--output-metrics", type=str, default="models/metrics_baseline.json")
    parser.add_argument("--output-cm", type=str, default="assets/confusion_matrix_neu.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV2 on device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size, seed=args.seed
    )
    print(f"Split -> Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_mobilenetv2(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), args.output_model)
            marker = "  <-- saved best"

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc*100:6.2f}% | "
            f"Val Loss: {va_loss:.4f} | Acc: {va_acc*100:6.2f}%{marker}"
        )

    train_time = time.time() - t_start
    print(f"\nCompleted in {train_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    model.load_state_dict(torch.load(args.output_model, map_location=device, weights_only=True))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print(f"\nTEST SET RESULTS (270 images) -> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    save_confusion_matrix(y_true, y_pred, test_acc, args.output_cm)
    model_size_mb = os.path.getsize(args.output_model) / (1024 ** 2)

    metrics = {
        "model": "MobileNetV2",
        "device": str(device),
        "epochs": args.epochs,
        "train_time_sec": round(train_time, 2),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "model_size_mb": round(model_size_mb, 2),
        "per_class_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
    }
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved weights: {args.output_model} ({model_size_mb:.2f} MB)")
    print(f"Saved confusion matrix: {args.output_cm}")
    print(f"Saved metrics: {args.output_metrics}")


if __name__ == "__main__":
    main()
