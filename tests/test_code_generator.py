import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from code_generator import CodeGenerator


def _compile_generated_python(content: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(content, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)


def test_fallback_codegen_uses_kyma_emg_window_and_features():
    project = CodeGenerator().generate("build an EMG classifier webapp", {
        "signal_modality": "EMG",
        "gestures": ["rest", "open", "close", "pinch", "point"],
        "classifier": "LDA",
        "accuracy": "N/A",
        "n_channels": 8,
        "sample_rate": 250,
        "window_size_ms": 200,
        "window_size_samples": 50,
        "features": ["MAV", "RMS", "WL", "ZC", "SSC"],
        "feature_dim": 40,
        "model_path": r"models\emg_lda_model.pkl",
    })

    files = {file.path: file.content for file in project.files}
    inference = files["src/inference.py"]
    train = files["src/train.py"]

    assert "firmware/controller.ino" not in files
    assert "import serial" not in inference
    assert "SERIAL_PORT" not in inference
    assert "pyserial" not in files["requirements.txt"]
    assert "WINDOW_SIZE = 50" in inference
    assert 'FEATURE_ORDER = ["MAV", "RMS", "WL", "ZC", "SSC"]' in inference
    assert 'FEATURE_MODE = "ordered_emg"' in inference
    assert "EXPECTED_FEATURES = 40" in inference
    assert 'MODEL_PATH = "models/emg_lda_model.pkl"' in inference
    assert "decode_prediction" in inference

    _compile_generated_python(inference, "inference.py")
    _compile_generated_python(train, "train.py")


def test_fallback_codegen_includes_firmware_only_for_hardware_prompts():
    project = CodeGenerator().generate("build Arduino servo firmware for EMG hand control", {
        "signal_modality": "EMG",
        "gestures": ["rest", "open"],
        "classifier": "LDA",
        "n_channels": 8,
        "sample_rate": 250,
        "window_size_ms": 200,
        "window_size_samples": 50,
        "features": ["MAV", "RMS", "WL", "ZC", "SSC"],
        "feature_dim": 40,
        "model_path": "models/emg_lda_model.pkl",
    })

    paths = {file.path for file in project.files}
    files = {file.path: file.content for file in project.files}

    assert "firmware/controller.ino" in paths
    assert "import serial" in files["src/inference.py"]
    assert "pyserial>=3.5" in files["requirements.txt"]


def test_fallback_codegen_supports_prompt_model_feature_contract():
    project = CodeGenerator().generate("build an EMG classifier webapp", {
        "signal_modality": "EMG",
        "gestures": ["rest", "open", "close", "pinch", "point"],
        "classifier": "logistic_regression",
        "n_channels": 8,
        "sample_rate": 250,
        "window_size_ms": 200,
        "window_size_samples": 50,
        "features": ["MAV", "RMS", "WL", "ZC", "SSC"],
        "feature_dim": 61,
        "model_path": "sessions/pipeline_models/emg_prompt_model.pkl",
    })

    inference = next(file.content for file in project.files if file.path == "src/inference.py")

    assert 'FEATURE_MODE = "pipeline_window"' in inference
    assert "EXPECTED_FEATURES = 61" in inference
    assert "rms, mav, mean, std, p2p, wl, zc" in inference
    _compile_generated_python(inference, "prompt_inference.py")
