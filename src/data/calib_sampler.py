from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import json
import torch
from torch.utils.data import Subset


def subset_from_indices(dataset, indices_json_path: str | Path):
    jj = json.loads(Path(indices_json_path).read_text(encoding="utf-8"))
    return Subset(dataset, jj["indices"])
