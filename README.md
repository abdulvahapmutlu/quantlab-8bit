# QuantLab-8bit ⚡  
**Reproducible PTQ vs QAT benchmark suite for compact vision models.**

---

## ✨ Overview

QuantLab-8bit is a **research-grade framework** for exploring **quantization** on compact vision backbones.  
It provides:

- 📊 **Leaderboards**: Accuracy (Top-1/Top-5), size, latency (p50/p95/p99).  
- 🔍 **Visuals**: weight/activation histograms, error heatmaps, saliency drift, outlier analysis.  
- ⚙️ **Reproducibility**: powered by [ReproKit-ML](https://pypi.org/project/reprokit-ml/).  
- 🖥️ **Cross-runtime**: PyTorch eager (training), ONNX Runtime CPU (latency target).  

---

## 🔧 Models & Datasets

**Models**  
- MobileNetV2  
- ResNet18  
- EfficientNet-Lite0  
- ViT-Tiny (stretch, transformer quirks)

**Datasets**  
- CIFAR-10 (primary)  
- Tiny-ImageNet (auxiliary)  
- Calibration sets: 1–5% train split (for PTQ static only)

---

## 📦 Installation

```
# clone repo
git clone https://github.com/abdulvahapmutlu/quantlab-8bit.git
cd quantlab-8bit

# create virtual environment (Python ≥3.10)
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# source .venv/bin/activate  # (Linux/Mac)

# install dependencies
pip install -r repro/environment/requirements.lock
```

Optional: install [ReproKit-ML](https://pypi.org/project/reprokit-ml/) for full reproducibility hooks.

---

## 🚀 Quickstart

### 1. Prepare datasets

```
python scripts/data/materialize_cifar10.py
python scripts/data/materialize_tinyimagenet.py
```

### 2. Export FP32 baseline

```
python -m src.quant.export_fp32_onnx \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-config configs/models/mobilenetv2.yaml \
  --out-dir artifacts/onnx/fp32/mobilenet_v2_cifar10
```

### 3. Run PTQ (static)

```
python -m src.quant.ptq_static_export \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-config configs/models/mobilenetv2.yaml \
  --quant-config configs/quant/ptq_static.yaml \
  --calib-indices artifacts/reports/calibration_indices/cifar10.json \
  --out-dir artifacts/onnx/ptq/mobilenet_v2_cifar10/pcw_symW_asymA_minmax
```

### 4. Run QAT

```
python -m src.quant.qat_train_export \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-config configs/models/mobilenetv2.yaml \
  --train-config configs/train/qat_finetune.yaml \
  --out-dir artifacts/onnx/qat/mobilenet_v2_cifar10
```

### 5. Benchmark latency

```
python -m src.bench.ort_cpu_bench \
  --dataset-config configs/datasets/cifar10.yaml \
  --model-config configs/models/mobilenetv2.yaml \
  --bench-config configs/bench/onnxruntime_cpu.yaml \
  --method ptq_static:pcw_symW_asymA_minmax \
  --onnx-path artifacts/onnx/ptq/mobilenet_v2_cifar10/pcw_symW_asymA_minmax/model.onnx \
  --out-json artifacts/reports/bench/mobilenet_v2_cifar10_ptq_static.json \
  --out-csv artifacts/reports/bench/ort_cpu_results.csv
```

### 6. Build leaderboard

```
python -m src.report.leaderboard_builder \
  --matrix-config configs/experiment_matrix.yaml \
  --leaderboard-config configs/reporting/leaderboard.yaml \
  --bench-csv artifacts/reports/bench/ort_cpu_results.csv
```

📑 Final tables will appear in:

* `artifacts/reports/leaderboard.md`
* `artifacts/reports/leaderboard_html/index.html`

---

## 🔍 Visual Analytics

* **Histograms**: `artifacts/reports/viz/weights/`, `activations/`
* **Error heatmaps**: `artifacts/reports/viz/error_heatmaps/`
* **Saliency drift (Grad-CAM)**: `artifacts/reports/viz/saliency/`
* **Outliers**: `artifacts/reports/viz/outliers/`

---

## 📐 Reproducibility

* Fixed seed: see [`repro/seeds.json`](repro/seeds.json)
* Env freeze: [`repro/environment/requirements.lock`](repro/environment/requirements.lock)
* Data hashes + manifests: auto-generated under `repro/`
* CI guards: pre-commit hooks check staleness of manifests and seeds

Re-running the pipeline yields identical **hashes**, stable p95 latency (±10%), and matching leaderboards.

---

## ⚠️ Caveats

* **PTQ may fail** on activations with heavy outliers (esp. ViT).
* **QAT costs** more training time but recovers accuracy.
* **Dynamic quant** is included for completeness but underperforms on CNNs.

---

## 🌐 Interactive Demo

Run locally:

```
streamlit run src/demo/app.py
```

Where `demo_app.py` lets you:

* select model + method
* view accuracy/latency plots
* drag-and-drop an image to compare FP32 vs INT8 predictions

---

## 👥 Contributions

* PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 License

[Apache 2.0](LICENSE) © 2025
