"""
Code Generator — uses LLM to generate full-stack project code.

After biosignal training completes, this module generates:
- Python training/inference scripts
- Arduino/embedded firmware
- Web dashboard apps

Supports: OpenAI, Anthropic, DeepSeek, any OpenAI-compatible endpoint.
"""
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFile:
    path: str
    content: str
    language: str = "python"
    description: str = ""


@dataclass
class GeneratedProject:
    files: List[GeneratedFile] = field(default_factory=list)
    summary: str = ""
    timestamp: float = 0.0

    def to_dict(self):
        return {
            "files": [{"path": f.path, "content": f.content,
                        "language": f.language, "description": f.description}
                       for f in self.files],
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


# Language detection for syntax highlighting
_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".html": "html", ".css": "css", ".json": "json",
    ".ino": "cpp", ".cpp": "cpp", ".h": "cpp", ".c": "c",
    ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
    ".sh": "bash", ".bat": "batch", ".txt": "plaintext",
    ".jsx": "javascript", ".tsx": "typescript",
}


def _detect_lang(path: str) -> str:
    for ext, lang in _LANG_MAP.items():
        if path.endswith(ext):
            return lang
    return "plaintext"


class CodeGenerator:
    """Generate full-stack project code via LLM."""

    def __init__(self):
        self.api_key: Optional[str] = None
        self.provider: str = "openai"
        self.base_url: Optional[str] = None
        self.model: Optional[str] = None
        self.last_project: Optional[GeneratedProject] = None
        self._streaming_lines: List[str] = []
        self._streaming_file: Optional[str] = None
        self._status_cb = None

    def set_api_key(self, key: str, provider: str = "openai",
                    base_url: str = None, model: str = None):
        self.api_key = key
        self.provider = provider
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

    def set_status_callback(self, cb):
        self._status_cb = cb

    def generate(self, prompt: str, context: dict = None) -> GeneratedProject:
        """Generate a full project from prompt + training context."""
        ctx = context or {}
        if not self.api_key:
            return self._generate_fallback(prompt, ctx)
        try:
            return self._generate_llm(prompt, ctx)
        except Exception as e:
            detail = self._format_provider_error(e)
            logger.warning("LLM code gen failed, using fallback: %s", detail)
            self._emit_status(f"Cloud AI failed: {detail}", "error")
            return self._generate_fallback(prompt, ctx)

    def refine(self, project: dict, prompt: str, context: dict = None) -> GeneratedProject:
        """Revise an existing generated project from a change request."""
        if not self.api_key:
            return self._refine_fallback(project, prompt)
        ctx = context or {}
        try:
            return self._refine_llm(project, prompt, ctx)
        except Exception as e:
            detail = self._format_provider_error(e)
            logger.warning("LLM code refine failed, using fallback note: %s", detail)
            self._emit_status(f"Cloud AI failed: {detail}", "error")
            return self._refine_fallback(project, prompt)

    def _build_system_prompt(self, ctx: dict) -> str:
        return f"""You are KYMA Code Generator - an expert full-stack developer for biosignal projects.

Given a project description and training context, generate a COMPLETE, WORKING project.

CONTEXT:
- Signal modality: {ctx.get('signal_modality', 'EMG')}
- Classes/gestures: {json.dumps(ctx.get('gestures', ['rest', 'active']))}
- Model type: {ctx.get('classifier', 'LDA')}
- Model accuracy: {ctx.get('accuracy', 'N/A')}
- Channels: {ctx.get('n_channels', 8)}
- Sample rate: {ctx.get('sample_rate', 250)} Hz
- Model path: {ctx.get('model_path', 'models/model.pkl')}

OUTPUT FORMAT - respond with ONLY a JSON array of file objects:
[
  {{"path": "src/main.py", "content": "..full code..", "description": "Main entry point"}},
  {{"path": "src/inference.py", "content": "...", "description": "Real-time inference"}},
  {{"path": "firmware/controller.ino", "content": "...", "description": "Arduino firmware"}},
  {{"path": "webapp/index.html", "content": "...", "description": "Web dashboard"}}
]

RULES:
1. Generate COMPLETE, RUNNABLE code - no placeholders or TODOs
2. Include ALL necessary imports and dependencies
3. Python scripts should use the trained model for inference
4. Firmware should accept serial commands from the Python script
5. Web apps should be self-contained single-file HTML+JS+CSS
6. Generate firmware only when the user explicitly asks for Arduino, firmware,
   embedded code, servos, motors, robot control, or hardware actuation
7. Include a requirements.txt with all Python dependencies
8. Include a README.md with setup instructions
9. Code quality should match production standards"""

    def _generate_llm(self, prompt: str, ctx: dict) -> GeneratedProject:
        system = self._build_system_prompt(ctx)
        user_msg = f"Build this project:\n\n{prompt}"

        if self.provider == "anthropic":
            return self._call_anthropic(system, user_msg)

        # OpenAI-compatible (OpenAI, DeepSeek, etc.)
        urls = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "deepseek": self.base_url or "https://api.deepseek.com/v1/chat/completions",
        }
        models = {
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat",
        }
        url = self._chat_completions_url(urls.get(self.provider, self.base_url or
                        f"https://api.{self.provider}.com/v1/chat/completions")
        )
        model = self.model or models.get(self.provider, self.provider)

        self._emit_status("Generating code...", "working")
        payload = {"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], "temperature": 0.2, "max_tokens": 16000}
        if self.provider == "nvidia":
            payload["chat_template_kwargs"] = self._nvidia_chat_template_kwargs()
            payload["max_tokens"] = 4096
            payload["stream"] = True
            content = self._stream_openai_compatible(url, payload, self._request_timeout(120.0))
        else:
            data = self._post_openai_compatible(url, payload, self._request_timeout(120.0))
            content = data["choices"][0]["message"]["content"]
        return self._parse_response(content)

    def _call_anthropic(self, system: str, prompt: str) -> GeneratedProject:
        import httpx
        self._emit_status("Generating code...", "working")
        resp = httpx.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key,
                      "Content-Type": "application/json",
                      "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 16000,
                  "system": system,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120.0)
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return self._parse_response(content)

    def _refine_llm(self, project: dict, prompt: str, ctx: dict) -> GeneratedProject:
        files = project.get("files") if isinstance(project, dict) else []
        system = self._build_system_prompt(ctx) + """

You are revising an EXISTING generated project. Return the complete revised
JSON array of file objects. Keep unchanged files unless the request requires
changes. Do not omit files from the revised project."""
        user_msg = "Change request:\n" + prompt + "\n\nCurrent project files:\n" + json.dumps(files, ensure_ascii=False)

        if self.provider == "anthropic":
            return self._call_anthropic(system, user_msg)

        urls = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "deepseek": self.base_url or "https://api.deepseek.com/v1/chat/completions",
        }
        models = {
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat",
        }
        url = self._chat_completions_url(urls.get(self.provider, self.base_url or
                        f"https://api.{self.provider}.com/v1/chat/completions")
        )
        model = self.model or models.get(self.provider, self.provider)

        self._emit_status("Applying code changes...", "working")
        payload = {"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], "temperature": 0.15, "max_tokens": 16000}
        if self.provider == "nvidia":
            payload["chat_template_kwargs"] = self._nvidia_chat_template_kwargs()
            payload["max_tokens"] = 4096
            payload["stream"] = True
            content = self._stream_openai_compatible(url, payload, self._request_timeout(120.0))
        else:
            data = self._post_openai_compatible(url, payload, self._request_timeout(120.0))
            content = data["choices"][0]["message"]["content"]
        return self._parse_response(content)

    def _chat_completions_url(self, url: str) -> str:
        clean = str(url or "").rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        return f"{clean}/chat/completions"

    def _request_timeout(self, default: float) -> float:
        return 900.0 if self.provider == "nvidia" else default

    def _nvidia_chat_template_kwargs(self) -> dict:
        model = str(self.model or "").lower()
        if "glm" in model or "qwen" in model:
            return {"enable_thinking": False}
        return {"thinking": False}

    def _post_openai_compatible(self, url: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{exc.code} from {self.provider}: {body[:500]}") from exc

    def _stream_openai_compatible(self, url: str, payload: dict, timeout: float) -> str:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        chunks: List[str] = []
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    try:
                        event = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            chunks.append(content)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{exc.code} from {self.provider}: {body[:500]}") from exc
        return "".join(chunks)

    def _format_provider_error(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) is not None:
            text = getattr(response, "text", "") or ""
            return f"{response.status_code} from {self.provider}: {text[:500]}"
        return str(exc)

    def _refine_fallback(self, project: dict, prompt: str) -> GeneratedProject:
        """Preserve files and record the requested change when no LLM key is configured."""
        revised = GeneratedProject(timestamp=time.time())
        files = project.get("files") if isinstance(project, dict) else []
        for fd in files or []:
            path = fd.get("path", "unknown.txt")
            revised.files.append(GeneratedFile(
                path=path,
                content=fd.get("content", ""),
                language=fd.get("language") or _detect_lang(path),
                description=fd.get("description", ""),
            ))
        revised.files.append(GeneratedFile(
            path="docs/requested_changes.md",
            language="markdown",
            description="Requested AI change note",
            content=(
                "# Requested Changes\n\n"
                "KYMA could not call an external code model because no code-generation API key "
                "is configured for this request. The project files were preserved.\n\n"
                "## Request\n\n"
                f"{prompt.strip() or 'No request provided.'}\n"
            ),
        ))
        revised.summary = f"Saved change request ({len(revised.files)} files)"
        self.last_project = revised
        self._emit_status("Change request saved; configure an AI key to rewrite code", "complete")
        return revised

    def _parse_response(self, content: str) -> GeneratedProject:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            files_data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                files_data = json.loads(match.group())
            else:
                # Wrap entire response as single file
                files_data = [{"path": "output.txt", "content": content,
                               "description": "Generated output"}]

        project = GeneratedProject(timestamp=time.time())
        for fd in files_data:
            path = fd.get("path", "unknown.txt")
            project.files.append(GeneratedFile(
                path=path,
                content=fd.get("content", ""),
                language=_detect_lang(path),
                description=fd.get("description", ""),
            ))
        project.summary = f"Generated {len(project.files)} files"
        self.last_project = project
        self._emit_status(f"Done - {len(project.files)} files generated", "complete")
        return project

    def _generate_fallback(self, prompt: str, ctx: dict) -> GeneratedProject:
        """Rule-based code generation when no API key is available."""
        modality = ctx.get("signal_modality", "EMG")
        gestures = ctx.get("gestures", ["rest", "active"])
        classifier = ctx.get("classifier", "LDA")
        n_ch = ctx.get("n_channels", 8)
        sr = ctx.get("sample_rate", 250)
        window_size = int(ctx.get("window_size_samples") or max(1, int(sr * 0.2)))
        window_ms = int(ctx.get("window_size_ms") or round(window_size * 1000 / max(int(sr), 1)))
        feature_names = [str(f).upper() for f in (ctx.get("features") or ["MAV", "RMS", "WL", "ZC", "SSC"])]
        if not feature_names:
            feature_names = ["MAV", "RMS", "WL", "ZC", "SSC"]
        feature_dim = int(ctx.get("feature_dim") or (int(n_ch) * len(feature_names)))
        feature_mode = "pipeline_window" if feature_dim == int(n_ch) * 7 + 5 else "ordered_emg"
        model_path = str(ctx.get("model_path") or "models/model.pkl").replace("\\", "/")
        accuracy = ctx.get("accuracy", "N/A")
        gesture_list = ", ".join(f'"{g}"' for g in gestures)
        feature_list = ", ".join(f'"{name}"' for name in feature_names)
        model_path_literal = json.dumps(model_path)
        wants_firmware = self._prompt_wants_firmware(prompt)
        firmware_readme_row = "| `firmware/controller.ino` | Arduino gesture controller |\n" if wants_firmware else ""
        serial_import = "import serial\n" if wants_firmware else ""
        serial_config = 'SERIAL_PORT = "COM4"  # Arduino port\nSERIAL_BAUD = 115200\n' if wants_firmware else ""
        serial_setup = '''    # Connect to Arduino
    arduino = None
    try:
        arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print(f"Arduino connected on {SERIAL_PORT}")
    except Exception as e:
        print(f"Arduino not available: {e}")

''' if wants_firmware else "    arduino = None\n\n"
        serial_write = '''                if arduino:
                    arduino.write(f"{label}\\n".encode())
''' if wants_firmware else ""
        serial_cleanup = '''        if arduino:
            arduino.close()
''' if wants_firmware else "        pass\n"

        project = GeneratedProject(timestamp=time.time())

        # 1. Main inference script
        project.files.append(GeneratedFile(
            path="src/inference.py",
            language="python",
            description="Real-time biosignal inference engine",
            content=f'''"""
KYMA Generated - Real-time {modality} Inference
Project: {prompt}
Model: {classifier} | Accuracy: {accuracy}
"""
import pickle
import numpy as np
{serial_import.rstrip()}
import time
from numbers import Integral
from collections import deque

# Configuration
MODEL_PATH = {model_path_literal}
CLASSES = [{gesture_list}]
FEATURE_ORDER = [{feature_list}]
FEATURE_MODE = "{feature_mode}"
EXPECTED_FEATURES = {feature_dim}
N_CHANNELS = {n_ch}
SAMPLE_RATE = {sr}
WINDOW_SIZE = {window_size}  # {window_ms} ms windows, matching KYMA training
{serial_config.rstrip()}

def zero_crossings(sig):
    return np.sum((sig[1:] * sig[:-1]) < 0) if sig.size > 1 else 0

def slope_sign_changes(sig):
    dx = np.diff(sig)
    return np.sum((dx[1:] * dx[:-1]) < 0) if dx.size > 1 else 0

def extract_features(window):
    """Extract features in the same order used by KYMA EMG training."""
    x = window.astype(np.float32, copy=False)
    if FEATURE_MODE == "pipeline_window":
        if x.shape[0] < N_CHANNELS:
            pad = np.zeros((N_CHANNELS - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        x = x[:N_CHANNELS]
        dx = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros_like(x)
        rms = np.sqrt(np.mean(np.square(x), axis=1))
        mav = np.mean(np.abs(x), axis=1)
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        p2p = np.ptp(x, axis=1)
        wl = np.sum(np.abs(dx), axis=1)
        zc = np.sum((x[:, 1:] * x[:, :-1]) < 0, axis=1) if x.shape[1] > 1 else np.zeros(x.shape[0])
        out = np.concatenate([
            rms, mav, mean, std, p2p, wl, zc,
            np.asarray([
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.max(rms)),
                float(np.mean(p2p)),
                float(np.std(p2p)),
            ], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        if out.size != EXPECTED_FEATURES:
            raise ValueError(f"Feature mismatch: got {{out.size}}, expected {{EXPECTED_FEATURES}}")
        return out

    features = []
    for ch in range(N_CHANNELS):
        if ch >= window.shape[0]:
            sig = np.zeros(window.shape[1], dtype=np.float32)
        else:
            sig = window[ch].astype(np.float32, copy=False)
        dx = np.diff(sig) if sig.size > 1 else np.zeros(1, dtype=np.float32)
        values = {{
            "MAV": np.mean(np.abs(sig)),
            "RMS": np.sqrt(np.mean(sig ** 2)),
            "WL": np.sum(np.abs(dx)),
            "ZC": zero_crossings(sig),
            "SSC": slope_sign_changes(sig),
        }}
        for name in FEATURE_ORDER:
            features.append(float(values.get(name, 0.0)))
    out = np.array(features, dtype=np.float32)
    if out.size != EXPECTED_FEATURES:
        raise ValueError(f"Feature mismatch: got {{out.size}}, expected {{EXPECTED_FEATURES}}")
    return out

def decode_prediction(raw_prediction):
    if isinstance(raw_prediction, Integral) or isinstance(raw_prediction, np.integer):
        idx = int(raw_prediction)
        return CLASSES[idx] if 0 <= idx < len(CLASSES) else str(raw_prediction)
    return str(raw_prediction)

def predict_label(model, features):
    raw_prediction = model.predict(features)[0]
    label = decode_prediction(raw_prediction)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(np.max(model.predict_proba(features)[0]))
        except Exception:
            confidence = None
    return label, confidence

def main():
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded: {{len(CLASSES)}} classes")

{serial_setup.rstrip()}
    # Simulated stream (replace with real hardware)
    print("Starting inference loop...")
    buffer = deque(maxlen=WINDOW_SIZE)

    try:
        while True:
            # Replace with real data acquisition
            sample = np.random.randn(N_CHANNELS) * 0.001
            buffer.append(sample)

            if len(buffer) == WINDOW_SIZE:
                window = np.array(buffer).T  # (channels, samples)
                features = extract_features(window).reshape(1, -1)
                label, confidence = predict_label(model, features)
                suffix = f" ({{confidence:.0%}})" if confidence is not None else ""
                print(f"Predicted: {{label}}{{suffix}}")

{serial_write.rstrip()}

            time.sleep(1.0 / SAMPLE_RATE)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
{serial_cleanup.rstrip()}

if __name__ == "__main__":
    main()
'''
        ))

        # 2. Training script
        project.files.append(GeneratedFile(
            path="src/train.py",
            language="python",
            description="Model training pipeline",
            content=f'''"""
KYMA Generated - {modality} Training Pipeline
Project: {prompt}
"""
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import os

CLASSES = [{gesture_list}]
FEATURE_ORDER = [{feature_list}]
FEATURE_MODE = "{feature_mode}"
EXPECTED_FEATURES = {feature_dim}
MODEL_PATH = {model_path_literal}

def zero_crossings(sig):
    return np.sum((sig[1:] * sig[:-1]) < 0) if sig.size > 1 else 0

def slope_sign_changes(sig):
    dx = np.diff(sig)
    return np.sum((dx[1:] * dx[:-1]) < 0) if dx.size > 1 else 0

def extract_features(window):
    x = window.astype(np.float32, copy=False)
    if FEATURE_MODE == "pipeline_window":
        if x.shape[0] < {n_ch}:
            pad = np.zeros(({n_ch} - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        x = x[:{n_ch}]
        dx = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros_like(x)
        rms = np.sqrt(np.mean(np.square(x), axis=1))
        mav = np.mean(np.abs(x), axis=1)
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        p2p = np.ptp(x, axis=1)
        wl = np.sum(np.abs(dx), axis=1)
        zc = np.sum((x[:, 1:] * x[:, :-1]) < 0, axis=1) if x.shape[1] > 1 else np.zeros(x.shape[0])
        out = np.concatenate([
            rms, mav, mean, std, p2p, wl, zc,
            np.asarray([
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.max(rms)),
                float(np.mean(p2p)),
                float(np.std(p2p)),
            ], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        if out.size != EXPECTED_FEATURES:
            raise ValueError(f"Feature mismatch: got {{out.size}}, expected {{EXPECTED_FEATURES}}")
        return out

    features = []
    for ch in range({n_ch}):
        if ch >= window.shape[0]:
            sig = np.zeros(window.shape[1], dtype=np.float32)
        else:
            sig = window[ch].astype(np.float32, copy=False)
        dx = np.diff(sig) if sig.size > 1 else np.zeros(1, dtype=np.float32)
        values = {{
            "MAV": np.mean(np.abs(sig)),
            "RMS": np.sqrt(np.mean(sig ** 2)),
            "WL": np.sum(np.abs(dx)),
            "ZC": zero_crossings(sig),
            "SSC": slope_sign_changes(sig),
        }}
        for name in FEATURE_ORDER:
            features.append(float(values.get(name, 0.0)))
    out = np.array(features, dtype=np.float32)
    if out.size != EXPECTED_FEATURES:
        raise ValueError(f"Feature mismatch: got {{out.size}}, expected {{EXPECTED_FEATURES}}")
    return out

def generate_synthetic_data(n_per_class=200, n_channels={n_ch}, window_size={window_size}):
    X, y = [], []
    for i, gesture in enumerate(CLASSES):
        for _ in range(n_per_class):
            window = np.random.randn(n_channels, window_size) * (0.001 + i * 0.0005)
            X.append(extract_features(window))
            y.append(i)
    return np.array(X), np.array(y)

def main():
    print("Generating training data...")
    X, y = generate_synthetic_data()
    print(f"Dataset: {{X.shape[0]}} samples, {{X.shape[1]}} features")

    model = LinearDiscriminantAnalysis()
    scores = cross_val_score(model, X, y, cv=5)
    print(f"CV Accuracy: {{scores.mean():.1%}} +/- {{scores.std():.1%}}")

    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {{MODEL_PATH}}")

if __name__ == "__main__":
    main()
'''
        ))

        # 3. Firmware, only when the requested product includes embedded actuation.
        if wants_firmware:
            servo_lines = "\n".join(
                f'  else if (cmd == "{g}") {{ servo.write({30 + i * 30}); }}'
                if i > 0 else f'  if (cmd == "{g}") {{ servo.write(0); }}'
                for i, g in enumerate(gestures)
            )
            project.files.append(GeneratedFile(
                path="firmware/controller.ino",
                language="cpp",
                description="Arduino gesture controller firmware",
                content=f'''// KYMA Generated - {modality} Gesture Controller
// Project: {prompt}
#include <Servo.h>

Servo servo;
String cmd = "";

void setup() {{
  Serial.begin(115200);
  servo.attach(9);
  servo.write(0);
  Serial.println("KYMA Controller Ready");
}}

void loop() {{
  if (Serial.available()) {{
    cmd = Serial.readStringUntil('\\n');
    cmd.trim();
    if (cmd.length() > 0) {{
      handleCommand(cmd);
    }}
  }}
}}

void handleCommand(String cmd) {{
  Serial.print("CMD: ");
  Serial.println(cmd);
{servo_lines}
  else {{
    Serial.println("Unknown command");
  }}
}}
'''
            ))

        # 4. Web dashboard
        gesture_btns = "\n".join(
            f'          <button class="gesture-btn" onclick="sendGesture(\'{g}\')">{g.replace("_"," ").title()}</button>'
            for g in gestures
        )
        project.files.append(GeneratedFile(
            path="webapp/index.html",
            language="html",
            description="Real-time monitoring web dashboard",
            content=f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KYMA - {prompt}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: #0a0e14;
    color: #e8ecf5;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  .header {{
    padding: 20px 32px;
    background: linear-gradient(135deg, rgba(88,111,218,0.15), rgba(168,85,247,0.08));
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .header h1 {{
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #586fda, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .header .status {{
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(62,207,142,0.12);
    color: #3ecf8e;
    border: 1px solid rgba(62,207,142,0.2);
  }}
  .main {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 24px 32px;
  }}
  .card {{
    background: rgba(16,20,28,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(20px);
  }}
  .card h2 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7a99;
    margin-bottom: 16px;
  }}
  .prediction {{
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    padding: 24px;
    background: linear-gradient(135deg, rgba(88,111,218,0.1), rgba(168,85,247,0.05));
    border-radius: 12px;
    margin-bottom: 12px;
  }}
  .gesture-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px;
  }}
  .gesture-btn {{
    padding: 10px 16px;
    border-radius: 10px;
    border: 1px solid rgba(88,111,218,0.2);
    background: rgba(88,111,218,0.08);
    color: #e8ecf5;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .gesture-btn:hover {{
    background: rgba(88,111,218,0.2);
    border-color: rgba(88,111,218,0.4);
    transform: translateY(-1px);
  }}
  .meter {{ height: 6px; background: rgba(255,255,255,0.06); border-radius: 6px; overflow: hidden; margin: 4px 0; }}
  .meter-fill {{ height: 100%; border-radius: 6px; transition: width 0.3s; }}
  .channel-list {{ display: flex; flex-direction: column; gap: 6px; }}
  .channel {{ display: flex; align-items: center; gap: 8px; font-size: 11px; }}
  .channel label {{ width: 32px; color: #6b7a99; }}
  .log {{ font-family: monospace; font-size: 11px; color: #6b7a99; max-height: 200px; overflow-y: auto; }}
  .log div {{ padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }}
</style>
</head>
<body>
  <div class="header">
    <h1>KYMA - {prompt[:40]}</h1>
    <div class="status" id="status">Connecting...</div>
  </div>
  <div class="main">
    <div class="card">
      <h2>Live Prediction</h2>
      <div class="prediction" id="prediction">--</div>
      <div style="text-align:center;font-size:12px;color:#6b7a99" id="confidence">Confidence: --</div>
    </div>
    <div class="card">
      <h2>Manual Control</h2>
      <div class="gesture-grid">
{gesture_btns}
      </div>
    </div>
    <div class="card">
      <h2>Signal Channels</h2>
      <div class="channel-list" id="channels"></div>
    </div>
    <div class="card">
      <h2>Event Log</h2>
      <div class="log" id="log"></div>
    </div>
  </div>
  <script>
    const N = {n_ch};
    const chEl = document.getElementById('channels');
    for (let i = 0; i < N; i++) {{
      chEl.innerHTML += '<div class="channel"><label>CH' + (i+1) + '</label>' +
        '<div class="meter" style="flex:1"><div class="meter-fill" id="ch' + i + '" ' +
        'style="width:0%;background:linear-gradient(90deg,#586fda,#a855f7)"></div></div></div>';
    }}
    function log(msg) {{
      const el = document.getElementById('log');
      const d = document.createElement('div');
      d.textContent = new Date().toLocaleTimeString() + ' ' + msg;
      el.prepend(d);
      if (el.children.length > 50) el.lastChild.remove();
    }}
    function sendGesture(g) {{
      log('Manual: ' + g);
      document.getElementById('prediction').textContent = g.replace(/_/g,' ');
    }}
    // Simulated updates
    setInterval(() => {{
      for (let i = 0; i < N; i++) {{
        document.getElementById('ch'+i).style.width = (Math.random()*80+10)+'%';
      }}
    }}, 200);
    document.getElementById('status').textContent = 'Active';
    log('Dashboard initialized');
  </script>
</body>
</html>
'''
        ))

        # 5. Requirements
        project.files.append(GeneratedFile(
            path="requirements.txt",
            language="plaintext",
            description="Python dependencies",
            content="numpy>=1.24\nscikit-learn>=1.3\n" + ("pyserial>=3.5\n" if wants_firmware else "")
        ))

        # 6. README
        project.files.append(GeneratedFile(
            path="README.md",
            language="markdown",
            description="Project documentation",
            content=f"""# {prompt}

Generated by **KYMA** - AI-Guided Biosignal Infrastructure

## Setup

```bash
pip install -r requirements.txt
python src/train.py
python src/inference.py
```

## Project Structure

| File | Description |
|------|-------------|
| `src/train.py` | Training pipeline ({classifier}) |
| `src/inference.py` | Real-time inference engine |
| `webapp/index.html` | Monitoring web dashboard |
{firmware_readme_row}

## Configuration

- **Signal**: {modality} ({n_ch} channels @ {sr} Hz)
- **Window**: {window_ms} ms ({window_size} samples)
- **Features**: {', '.join(feature_names)} ({feature_dim} total)
- **Classes**: {', '.join(gestures)}
- **Model**: {classifier} ({accuracy} accuracy)
"""
        ))

        project.summary = f"Generated {len(project.files)} files (Python + webapp" + (" + firmware)" if wants_firmware else ")")
        self.last_project = project
        self._emit_status(f"Done - {len(project.files)} files", "complete")
        return project

    def _prompt_wants_firmware(self, prompt: str) -> bool:
        text = (prompt or "").lower()
        keywords = (
            "arduino", "firmware", "servo", "motor", "robot", "robotic",
            "embedded", "microcontroller", "esp32", "teensy", "raspberry pi pico",
            "actuator", "pwm", "serial command", "hardware control",
        )
        return any(keyword in text for keyword in keywords)

    def _emit_status(self, msg: str, phase: str = "working"):
        if self._status_cb:
            self._status_cb({"type": "codegen_status",
                             "data": {"message": msg, "phase": phase}})
