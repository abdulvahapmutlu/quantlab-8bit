from __future__ import annotations
from typing import Any, Dict
import json
import torch
import numpy as np
import random

def set_seeds(seeds_json_path: str) -> None:
    try:
        with open(seeds_json_path, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        s = int(seeds.get("global", 42))
    except Exception:
        s = 42
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(model_cfg: Dict[str, Any], num_classes: int):
    """
    Supported:
      - torchvision: resnet18, mobilenet_v2
      - timm: efficientnet_lite0, vit_tiny (vit_tiny_patch16_224)
    """
    name = model_cfg["model"]["name"].lower()
    source = model_cfg["model"].get("source", "torchvision").lower()
    pretrained = bool(model_cfg["model"].get("pretrained", True))

    if source == "torchvision":
        import torchvision.models as tvm
        if name in ("resnet18", "resnet-18"):
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            return m
        if name in ("mobilenet_v2", "mobilenetv2", "mobilenet-v2"):
            m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, num_classes)
            return m
        raise ValueError(f"Unsupported torchvision model '{name}'.")

    if source == "timm":
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm not installed. Run: pip install timm") from e

        # map friendly names
        if name in ("efficientnet_lite0", "efficientnet-lite0"):
            model_name = "efficientnet_lite0"
        elif name in ("vit_tiny", "vit-tiny", "vit_tiny_patch16_224"):
            model_name = "vit_tiny_patch16_224"
        else:
            model_name = name  # let timm resolve if exact

        m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return m

    raise ValueError(f"Unsupported source '{source}' in model config.")

def fuse_bn_if_requested(model, model_cfg: Dict[str, Any]):
    # placeholder (torch.fx fuse hook if you add it later)
    return model

def load_checkpoint(model, ckpt_path: str):
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    except FileNotFoundError:
        pass
    model.eval()
    return model
