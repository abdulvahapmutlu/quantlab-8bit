from __future__ import annotations
from typing import Any, Dict, Tuple, List
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision import datasets


def _eval_transforms(input_size: Tuple[int, int, int]):
    # Use ImageNet-like eval pipeline with resize+center crop if H/W > 32
    c, h, w = input_size
    if h <= 32 and w <= 32:
        return T.Compose([T.ToTensor()])
    short = max(h, w)
    return T.Compose([T.Resize(short + 32), T.CenterCrop((h, w)), T.ToTensor()])


def _cifar10(root: str, split: str, transform):
    train = (split == "train")
    return datasets.CIFAR10(root=root, train=train, download=False, transform=transform)


def _build_dataset(dataset_cfg: Dict[str, Any], split: str, transform):
    name = dataset_cfg["dataset"]["name"].lower()
    root = dataset_cfg["dataset"]["root"]
    if name == "cifar10":
        return _cifar10(root, "train" if split in ("train", "val") else "test", transform)
    if name in ("tinyimagenet", "tiny-imagenet", "tiny_image_net", "tinyimage"):
        # use ImageFolder train/val
        return _tiny_imagenet(root, "train" if split in ("train", "val") else "val", transform)
    raise NotImplementedError(f"Dataset '{name}' not implemented in this minimal build.")


def _apply_seeded_split(ds, dataset_cfg: Dict[str, Any], split: str):
    # Uses the precomputed split file from Step 5
    if split not in ("train", "val"):
        return ds
    split_file = dataset_cfg["dataset"].get("val_split_file", "")
    if "{dataset}" in split_file:
        split_file = split_file.format(dataset=dataset_cfg["dataset"]["name"].lower())
    p = Path(split_file)
    if not p.exists():
        # Fallback: 90/10 stratified could be added; for now just return full train
        return ds
    jj = json.loads(p.read_text(encoding="utf-8"))
    idx = jj["train_idx"] if split == "train" else jj["val_idx"]
    return Subset(ds, idx)


def build_eval_loader(dataset_cfg: Dict[str, Any], *, split: str, batch_size: int, num_workers: int):
    input_size = tuple(dataset_cfg["dataset"]["input_size"])
    tfm = _eval_transforms(input_size)
    ds = _build_dataset(dataset_cfg, split, tfm)
    ds = _apply_seeded_split(ds, dataset_cfg, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_calib_loader(dataset_cfg: Dict[str, Any], indices_json: str, *, batch_size: int, num_workers: int):
    input_size = tuple(dataset_cfg["dataset"]["input_size"])
    tfm = _eval_transforms(input_size)
    # Calibration always from train
    base = _build_dataset(dataset_cfg, "train", tfm)
    jj = json.loads(Path(indices_json).read_text(encoding="utf-8"))
    subset = Subset(base, jj["indices"])
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def build_train_loader(dataset_cfg: Dict[str, Any], *, batch_size: int, num_workers: int):
    """Deterministic train loader using the same seeded split file (Step 5)."""
    input_size = tuple(dataset_cfg["dataset"]["input_size"])
    tfm = _eval_transforms(input_size)  # keep it modest; QAT usually uses light aug for stability
    ds = _build_dataset(dataset_cfg, "train", tfm)
    ds = _apply_seeded_split(ds, dataset_cfg, "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def _tiny_imagenet(root: str, split: str, transform):
    """
    Expects Tiny-ImageNet in the canonical layout:
      root/train/<class>/images/*.JPEG
      root/val/<class>/images/*.JPEG   (you can use ImageFolder for val too)
    If your val is flat with val_annotations.txt, run your materialize script to folderize first.
    """
    if split == "train":
        path = Path(root) / "train"
    elif split in ("val", "valid", "validation"):
        path = Path(root) / "val"
    else:
        path = Path(root) / "val"
    return datasets.ImageFolder(str(path), transform=transform)

