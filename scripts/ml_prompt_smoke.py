#!/usr/bin/env python3
"""Train a tiny synthetic prompt-model smoke classifier for CI/MLOps checks."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def feature_vector(window):
    x = np.asarray(window, dtype=np.float64)
    dx = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros_like(x)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    mav = np.mean(np.abs(x), axis=1)
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    p2p = np.ptp(x, axis=1)
    wl = np.sum(np.abs(dx), axis=1)
    zc = np.sum((x[:, 1:] * x[:, :-1]) < 0, axis=1) if x.shape[1] > 1 else np.zeros(x.shape[0])
    return np.concatenate([
        rms,
        mav,
        mean,
        std,
        p2p,
        wl,
        zc,
        np.asarray([
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.max(rms)),
            float(np.mean(p2p)),
            float(np.std(p2p)),
        ]),
    ]).astype(np.float32)


def make_dataset(labels, samples_per_label, channels, window_samples, seed):
    rng = np.random.default_rng(seed)
    windows = []
    y = []
    t = np.linspace(0, 1, window_samples, endpoint=False)
    for label_idx, label in enumerate(labels):
      base_gain = 0.7 + label_idx * 0.65
      emphasis = rng.uniform(0.75, 1.35, size=channels)
      emphasis[label_idx % channels] += 0.85
      for _ in range(samples_per_label):
          carrier = np.sin(2 * np.pi * (24 + label_idx * 12) * t)
          burst = np.sin(2 * np.pi * (66 + label_idx * 9) * t + rng.uniform(-0.3, 0.3))
          noise = rng.normal(0, 0.18 + label_idx * 0.03, size=(channels, window_samples))
          envelope = rng.uniform(0.8, 1.2) * (1.0 + 0.25 * np.sin(2 * np.pi * t * (label_idx + 1)))
          window = ((carrier * 0.32 + burst * 0.18) * envelope)[None, :] * emphasis[:, None]
          window = (window * base_gain + noise).astype(np.float32)
          windows.append(window)
          y.append(label)
    X = np.vstack([feature_vector(window)[None, :] for window in windows])
    y = np.asarray(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def maybe_log_mlflow(metrics, artifacts):
    try:
        import mlflow
    except Exception:
        return
    mlflow.set_experiment("kyma-prompt-model-smoke")
    with mlflow.start_run(run_name="prompt_model_smoke"):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        for artifact in artifacts:
            if Path(artifact).exists():
                mlflow.log_artifact(str(artifact))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-label", type=int, default=48)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--window-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-accuracy", type=float, default=0.85)
    parser.add_argument("--out-dir", default="reports/ml")
    parser.add_argument("--artifact-dir", default="artifacts/ml")
    args = parser.parse_args()

    labels = ["fresh", "fatiguing"]
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X, raw_y = make_dataset(labels, args.samples_per_label, args.channels, args.window_samples, args.seed)
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=y,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
    ])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_accuracy = float(accuracy_score(y_train, train_pred))
    val_accuracy = float(accuracy_score(y_val, val_pred))

    model_path = artifact_dir / "prompt_smoke_model.pkl"
    card_path = artifact_dir / "prompt_smoke_model_card.json"
    metrics_path = out_dir / "prompt_smoke_metrics.json"
    report_path = out_dir / "prompt_smoke_report.md"
    payload = {
        "model": model,
        "label_encoder": encoder,
        "labels": encoder.classes_.tolist(),
        "feature_dim": int(X.shape[1]),
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)

    metrics = {
        "labels": encoder.classes_.tolist(),
        "samples": int(len(X)),
        "feature_dim": int(X.shape[1]),
        "train_accuracy": round(train_accuracy, 4),
        "val_accuracy": round(val_accuracy, 4),
        "minimum_accuracy": float(args.minimum_accuracy),
        "confusion_matrix": confusion_matrix(y_val, val_pred, labels=list(range(len(labels)))).tolist(),
        "model_path": model_path.as_posix(),
    }
    card_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(
        "\n".join([
            "# KYMA Prompt Model Smoke",
            "",
            f"- Samples: {metrics['samples']}",
            f"- Feature dim: {metrics['feature_dim']}",
            f"- Train accuracy: {metrics['train_accuracy']:.4f}",
            f"- Validation accuracy: {metrics['val_accuracy']:.4f}",
            f"- Minimum required: {metrics['minimum_accuracy']:.4f}",
            f"- Model artifact: `{metrics['model_path']}`",
        ]),
        encoding="utf-8",
    )
    maybe_log_mlflow(metrics, [model_path, card_path, report_path])
    if val_accuracy < args.minimum_accuracy:
        raise SystemExit(f"Validation accuracy {val_accuracy:.4f} is below {args.minimum_accuracy:.4f}")


if __name__ == "__main__":
    main()
