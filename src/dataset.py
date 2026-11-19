from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NEUSurfaceDefectDataset(Dataset):
    """Dataset phân loại 6 loại khuyết tật bề mặt thép (NEU Surface Defect)."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = 224):
    """Chuẩn hóa và tăng cường dữ liệu an toàn cho bài toán kiểm tra ngoại quan (AOI)."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def collect_all_samples(data_root: str) -> List[Tuple[str, int]]:
    """Quét toàn bộ 1.800 ảnh trong data/NEU-DET/{train,validation}/images/<class_name>."""
    root = Path(data_root)
    samples: List[Tuple[str, int]] = []

    for split_dir in ["train", "validation"]:
        img_root = root / split_dir / "images"
        if not img_root.exists():
            continue
        for class_name in CLASS_NAMES:
            class_dir = img_root / class_name
            if not class_dir.exists():
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((str(img_file), CLASS_TO_IDX[class_name]))

    if len(samples) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong '{data_root}'. Hãy kiểm tra lại đường dẫn dữ liệu!"
        )
    return samples


def get_dataloaders(
    data_root: str = "data/NEU-DET",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    seed: int = 42,
):
    """Chia Stratified Split cố định (70% Train / 15% Val / 15% Test) với seed=42."""
    all_samples = collect_all_samples(data_root)
    labels = [label for _, label in all_samples]

    train_samples, temp_samples, _, temp_labels = train_test_split(
        all_samples, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_ds = NEUSurfaceDefectDataset(train_samples, transform=train_tf)
    val_ds = NEUSurfaceDefectDataset(val_samples, transform=eval_tf)
    test_ds = NEUSurfaceDefectDataset(test_samples, transform=eval_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = get_dataloaders("data/NEU-DET", batch_size=16, num_workers=0)
    print(f"Train: {len(tr.dataset)} | Val: {len(va.dataset)} | Test: {len(te.dataset)}")
    imgs, lbls = next(iter(tr))
    print(f"Sample batch -> images: {tuple(imgs.shape)}, labels: {lbls.tolist()}")
