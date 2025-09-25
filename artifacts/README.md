# artifacts/

This directory holds **generated outputs**. Everything here is **ignored by git** except for:
- `reports/calibration_indices/*.json` – small, deterministic lists of calibration sample indices
- `reports/splits/*.json` – deterministic train/val split indices

Typical structure after running the pipeline:

```

artifacts/
onnx/             # exported models (fp32, ptq, qat) — not committed
checkpoints/      # training/QAT checkpoints — not committed
reports/
bench/          # ORT CPU bench JSON/CSV — not committed
parity/         # parity reports — not committed
ptq_static/     # PTQ static calibration+metrics — not committed
ptq_dynamic/    # PTQ dynamic metrics — not committed
qat/            # QAT export/metrics — not committed
viz/            # figures (histograms, heatmaps, saliency) — not committed
leaderboard.md  # regenerated
leaderboard_html/ # regenerated site

```

## Regeneration

All contents (except `calibration_indices/` and `splits/`) are **reproducible** via the CLI commands in `README.md`.  
If you want to share models or leaderboards publicly, prefer **GitHub Releases** rather than committing binaries to the repo.
