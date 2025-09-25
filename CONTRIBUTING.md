# Contributing to QuantLab-8bit

Thanks for your interest! This project benchmarks PTQ/QAT on compact vision models with a strong focus on reproducibility. Contributions that improve **models**, **datasets**, **quantization recipes**, **parity checks**, **bench harnesses**, and **documentation** are welcome.

---

## Code of Conduct
Be respectful. Assume good intent. Focus on technical merit. (If you need a formal CoC, open an issue and we’ll add one.)

---

## Getting Started

### 1) Clone & environment

```
git clone https://github.com/abdulvahapmutlu/quantlab-8bit.git
cd quantlab-8bit
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

# Recommended: match the pinned lock
pip install -r repro/environment/requirements.lock
```

### 2) Pre-commit hooks (lint & basic hygiene)

```
pip install pre-commit
pre-commit install
```

### 3) Datasets

Materialize CIFAR-10 and Tiny-ImageNet locally (or edit paths in configs/datasets/*.yaml):

```
python scripts/data/materialize_cifar10.py
python scripts/data/materialize_tinyimagenet.py
```

# Development Guide

## Repo Layout (High-Level)

- **src/** – library & CLI modules (`bench`, `eval`, `quant`, `viz`, `report`, `utils`)
- **configs/** – YAML/JSON configs (datasets, models, train, quant, bench, eval, reporting)
- **artifacts/** – generated outputs (ignored by git, except indices & splits)
- **repro/** – seeds, requirements lock, reproducibility metadata
- **scripts/** – one-off helpers for data, splits, hooks

---

## Style & Conventions

- Python ≥ 3.10, type hints where practical  
- Keep functions small and testable; avoid hidden global state  
- Config-driven behavior; avoid hard-coded paths  
- Logging > `print` inside library code  
- Prefer pure functions in `utils/` and `viz/`  

---

## Commit Messages

Use clear, imperative subject lines:

- `feat(quant): add per-channel MSE estimator`
- `fix(eval): correct cosine threshold logic`
- `docs(readme): explain leaderboard badges`

---

## Adding a New Model

1. Create a model config under `configs/models/<name>.yaml`.  
2. Ensure `src/utils/torch_utils.py::build_model` supports it (or add a factory).  
3. Export FP32 → run parity (torch vs ONNX) → add PTQ/QAT paths if applicable.  
4. Update `configs/experiment_matrix.yaml` if you want it included in leaderboards.  
5. For **timm** backbones, make sure the input resolution aligns (e.g., `224×224`).  

---

## Adding a New Dataset

1. Add a dataset config to `configs/datasets/`.  
2. Extend `src/data/loaders.py` with a loader builder (train/val/test splits + transforms).  
3. Provide a small calibration index JSON under `artifacts/reports/calibration_indices/`.  
4. Update `scripts/splits/build_val_split.py` if a custom split policy is needed.  

---

## Quantization Recipes

- Put PTQ recipes in `configs/quant/ptq_static.yaml` (each with an `id:`).  
- Run `src/quant/ptq_static_export.py` and record metrics in `artifacts/reports/ptq_static/`.  
- Keep parity thresholds sensible (`configs/eval/parity.yaml`).  

---

## Testing & CI

- **Local smoke**: run one end-to-end (export → PTQ → bench → leaderboard).  
- **CI**: `.github/workflows/ci.yml` runs lint + a tiny smoke (time-boxed).  
- **Nightly**: `.github/workflows/nightly.yml` can regenerate leaderboards with caches.  

---

## Documentation

- Update `README.md` for new features or breaking changes.

## Opening a Pull Request

- Make sure pre-commit passes:  
  ```
  pre-commit run -a
  ```

* Add/adjust configs and minimal docs.
* Include before/after numbers (accuracy/latency) for quant changes.
* Reference any issues and describe rationale + tradeoffs.

Thanks for contributing! 🎉
