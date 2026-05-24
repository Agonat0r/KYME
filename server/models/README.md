Local biosignal model packs can live here or under `sessions/ai_models`.

Each pack is a folder with:

```text
your_pack/
  manifest.json
  weights.pt
```

Minimal `manifest.json`:

```json
{
  "name": "Compact Signal QA",
  "task": "signal_qa",
  "format": "torchscript",
  "weights": "weights.pt",
  "layout": "bct",
  "normalize": "zscore",
  "channels": 8,
  "window_samples": 256,
  "sample_rate_hz": 250,
  "labels": ["clean", "watch", "reject"]
}
```

Supported tasks:

- `signal_qa`
- `artifact_classifier`
- `fatigue_readiness`
- `condition_screen` (research-only screening heads; not diagnosis)

Supported formats:

- `torchscript`

The copilot will load these packs on startup and blend their outputs into the local workflow summary without making clinical claims.

Minimal research screening manifest:

```json
{
  "name": "ECG Arrhythmia Screen",
  "task": "condition_screen",
  "format": "torchscript",
  "weights": "weights.pt",
  "layout": "bct",
  "normalize": "zscore",
  "channels": 12,
  "window_samples": 2048,
  "sample_rate_hz": 500,
  "labels": ["normal_sinus", "afib_candidate", "other_arrhythmia_candidate"]
}
```

`condition_screen` packs should be treated as research-only heads. Their outputs are intended for workflow triage, cohort review, and offline research, not patient-facing diagnosis.
