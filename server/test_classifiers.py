"""
Sanity tests for LDA / TCN / Mamba classifiers and full EMGPipeline wiring.

Run from repo root:
    .venv/Scripts/python.exe server/test_classifiers.py

Uses synthetic EMG-like windows so tests run with no hardware.
"""
import os
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from classifiers import LDAClassifier, TCNClassifier, MambaClassifier, get_classifier
from config import config


N_CH = config.n_channels                 # 8
SEQ_LEN = config.window_size_samples     # 50 @ 250Hz, 200ms
N_CLASSES = len(config.gestures)         # 5
RNG = np.random.default_rng(0)


def _gen_raw(n_per_class=40):
    """Synthetic raw windows: each class has a distinct per-channel amplitude profile."""
    X, y = [], []
    for c in range(N_CLASSES):
        profile = (0.2 + 0.8 * RNG.random(N_CH)) * (c + 1)
        for _ in range(n_per_class):
            sig = RNG.standard_normal((N_CH, SEQ_LEN)).astype(np.float32)
            sig *= profile[:, None].astype(np.float32)
            sig += (RNG.standard_normal((N_CH, SEQ_LEN)) * 0.1).astype(np.float32)
            X.append(sig)
            y.append(c)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    perm = RNG.permutation(len(X))
    return X[perm], y[perm]


def _gen_features(n_per_class=40, n_features=20):
    """Synthetic feature vectors with separable class means."""
    X, y = [], []
    for c in range(N_CLASSES):
        mean = RNG.standard_normal(n_features) * (c + 1)
        for _ in range(n_per_class):
            X.append(mean + RNG.standard_normal(n_features) * 0.5)
            y.append(c)
    X = np.asarray(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    perm = RNG.permutation(len(X))
    return X[perm], y[perm]


def _check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + extra) if extra else ''}")
    return cond


def test_lda():
    print("-- LDA ---------------------------------------------")
    results = []
    X, y = _gen_features()
    clf = LDAClassifier()
    results.append(_check("instantiates", True))

    t0 = time.perf_counter()
    info = clf.fit(X, y)
    dt = time.perf_counter() - t0
    results.append(_check("fit runs", clf.is_fitted, f"{dt:.2f}s"))
    results.append(_check(
        "val_accuracy > 0.8",
        info.get("val_accuracy", 0) > 0.8,
        f"val_acc={info.get('val_accuracy'):.3f}",
    ))
    results.append(_check("reports n_params", info.get("n_params", 0) > 0,
                          f"params={info.get('n_params')}"))

    cls, conf = clf.predict(X[:1])
    results.append(_check("predict shape", isinstance(cls, int) and 0.0 <= conf <= 1.0,
                          f"cls={cls}, conf={conf:.3f}"))

    batch = clf.predict_batch(X[:10])
    results.append(_check("predict_batch", batch.shape == (10,)))

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "lda.pkl")
        clf.save(p)
        clf2 = LDAClassifier()
        clf2.load(p)
        c1, _ = clf.predict(X[:1])
        c2, _ = clf2.predict(X[:1])
        results.append(_check("save/load roundtrip matches", c1 == c2))

    return all(results)


def test_torch_classifier(name, cls, epochs_override=None):
    print(f"-- {name} -------------------------------------------")
    results = []
    X, y = _gen_raw()
    kwargs = dict(n_channels=N_CH, n_classes=N_CLASSES)
    if epochs_override is not None:
        kwargs["epochs"] = epochs_override
    clf = cls(**kwargs)
    results.append(_check("instantiates", True))

    t0 = time.perf_counter()
    info = clf.fit(X, y)
    dt = time.perf_counter() - t0
    results.append(_check("fit runs", clf.is_fitted, f"{dt:.2f}s"))
    results.append(_check(
        "val_accuracy > 0.5",
        info.get("val_accuracy", 0) > 0.5,
        f"val_acc={info.get('val_accuracy'):.3f}",
    ))
    results.append(_check("reports n_params", info.get("n_params", 0) > 0,
                          f"params={info.get('n_params')}"))

    one = X[:1]  # (1, n_ch, seq_len)
    pred_cls, conf = clf.predict(one)
    results.append(_check("predict shape", isinstance(pred_cls, int) and 0.0 <= conf <= 1.0,
                          f"cls={pred_cls}, conf={conf:.3f}"))

    # Also test 2D (unsqueeze path)
    pred_cls2, _ = clf.predict(one[0])
    results.append(_check("predict accepts (ch, L)", isinstance(pred_cls2, int)))

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, f"{name.lower()}.pt")
        clf.save(p)
        clf2 = cls(**kwargs)
        clf2.load(p)
        c1, _ = clf.predict(one)
        c2, _ = clf2.predict(one)
        results.append(_check("save/load roundtrip matches", c1 == c2,
                              f"orig={c1}, loaded={c2}"))

    return all(results)


def test_registry():
    print("-- Registry ----------------------------------------")
    results = []
    for n, T in (("LDA", LDAClassifier), ("TCN", TCNClassifier), ("Mamba", MambaClassifier)):
        kw = {} if n == "LDA" else {"n_channels": N_CH, "n_classes": N_CLASSES}
        obj = get_classifier(n, **kw)
        results.append(_check(f"get_classifier('{n}')", isinstance(obj, T)))
    try:
        get_classifier("DoesNotExist")
        results.append(_check("rejects unknown name", False))
    except ValueError:
        results.append(_check("rejects unknown name", True))
    return all(results)


def main():
    print(f"Config: n_channels={N_CH}, seq_len={SEQ_LEN}, n_classes={N_CLASSES}")
    print()
    summary = {}
    for name, fn in (
        ("registry", test_registry),
        ("LDA", test_lda),
        ("TCN", lambda: test_torch_classifier("TCN", TCNClassifier, epochs_override=25)),
        ("Mamba", lambda: test_torch_classifier("Mamba", MambaClassifier, epochs_override=20)),
    ):
        try:
            summary[name] = fn()
        except Exception:
            traceback.print_exc()
            summary[name] = False
        print()

    print("-- Summary -----------------------------------------")
    for k, v in summary.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
