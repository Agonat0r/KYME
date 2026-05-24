"""
Intent Engine — converts natural language prompts into pipeline configurations.

A user writes: "Build an EMG fatigue detector"
KYMA responds by constructing a dynamic pipeline plan.

Supports multiple LLM providers: OpenAI, Anthropic, DeepSeek.
With a robust rule-based fallback for offline/no-key usage.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ElectrodePlacement:
    channel: int
    muscle: str
    position_description: str
    confidence: float  # 0-1

@dataclass
class PipelinePlan:
    task_type: str = "classification"  # classification | regression | control
    task_description: str = ""
    signal_modality: str = "EMG"
    temporal_mode: str = "realtime"  # realtime | offline
    noise_tolerance: float = 0.5  # 0 = no noise, 1 = very tolerant
    suggested_electrodes: List[ElectrodePlacement] = field(default_factory=list)
    suggested_gestures: List[str] = field(default_factory=list)
    dataset_targets: Dict[str, int] = field(default_factory=lambda: {"min_windows": 50, "recommended_windows": 200})
    model_recommendations: List[str] = field(default_factory=lambda: ["LDA"])
    recording_protocol: List[Dict] = field(default_factory=list)
    estimated_duration_min: float = 10.0
    confidence: float = 0.5
    explanation: str = ""
    # ── Interactive training instructions (per-step user guidance) ──────
    training_instructions: List[Dict] = field(default_factory=list)
    project_outputs: List[str] = field(default_factory=lambda: ["python", "firmware", "webapp"])

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "task_description": self.task_description,
            "signal_modality": self.signal_modality,
            "temporal_mode": self.temporal_mode,
            "noise_tolerance": self.noise_tolerance,
            "suggested_electrodes": [{"channel": e.channel, "muscle": e.muscle,
                "position": e.position_description, "confidence": e.confidence}
                for e in self.suggested_electrodes],
            "suggested_gestures": self.suggested_gestures,
            "dataset_targets": self.dataset_targets,
            "model_recommendations": self.model_recommendations,
            "recording_protocol": self.recording_protocol,
            "estimated_duration_min": self.estimated_duration_min,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "training_instructions": self.training_instructions,
            "project_outputs": self.project_outputs,
        }


# ── Keyword patterns for rule-based parsing ──────────────────────────────

_TASK_PATTERNS = {
    "classification": r"classif|detect|recogni[zs]|identify|distinguish|gesture|movement|action",
    "regression": r"regress|predict|estimat|measure|quantif|intensity|force|fatigue",
    "control": r"control|drive|actuate|move|prosthe|robot|servo|interface",
}

_MODALITY_PATTERNS = {
    "EMG": r"emg|electromyogra|muscle|gesture|grip|flex|contraction|prosthe",
    "EEG": r"eeg|electroencephalogra|brain|neural|bci|thought|mental|focus|meditation|sleep",
    "ECG": r"ecg|ekg|electrocardio|heart|cardiac|pulse|arrhythmia",
    "EOG": r"eog|electrooculogra|eye|blink|gaze|saccade|ocular",
}

_GESTURE_PRESETS = {
    "hand": ["rest", "open", "close", "pinch", "point"],
    "arm": ["rest", "flex", "extend", "rotate_in", "rotate_out"],
    "bilateral_forearm": ["rest", "left_flex", "right_flex"],
    "fatigue": ["rest", "fresh", "moderate_fatigue", "high_fatigue"],
    "finger": ["rest", "thumb", "index", "middle", "ring", "pinky"],
    "grip": ["rest", "power_grip", "precision_grip", "lateral_grip", "open"],
    "eeg_focus": ["rest", "focused", "relaxed", "drowsy"],
    "eeg_motor": ["rest", "left_hand_imagine", "right_hand_imagine", "feet_imagine"],
    "eog_blink": ["rest", "single_blink", "double_blink", "left_look", "right_look"],
}

_ELECTRODE_PRESETS = {
    "bilateral_forearm": [
        ElectrodePlacement(0, "Left forearm flexor pair", "Left forearm muscle belly; one differential EMG pair on the left flexor mass", 0.95),
        ElectrodePlacement(1, "Right forearm flexor pair", "Right forearm muscle belly; one differential EMG pair on the right flexor mass", 0.95),
        ElectrodePlacement(7, "Ground/reference", "Ground clip on ear lobe; keep lead slack low and avoid tugging", 0.9),
    ],
    "hand": [
        ElectrodePlacement(0, "Flexor Digitorum", "Inner forearm, 1/3 from elbow", 0.9),
        ElectrodePlacement(1, "Extensor Digitorum", "Outer forearm, 1/3 from elbow", 0.9),
        ElectrodePlacement(2, "Flexor Carpi Radialis", "Inner forearm, radial side", 0.8),
        ElectrodePlacement(3, "Extensor Carpi Ulnaris", "Outer forearm, ulnar side", 0.8),
        ElectrodePlacement(4, "Biceps Brachii", "Front upper arm, belly", 0.7),
        ElectrodePlacement(5, "Triceps Brachii", "Back upper arm, belly", 0.7),
    ],
    "fatigue": [
        ElectrodePlacement(0, "Target Muscle (primary)", "Center of muscle belly", 0.95),
        ElectrodePlacement(1, "Target Muscle (secondary)", "2cm distal from primary", 0.85),
        ElectrodePlacement(2, "Antagonist Muscle", "Opposing muscle group", 0.7),
        ElectrodePlacement(3, "Reference Muscle", "Uninvolved muscle for baseline", 0.6),
    ],
    "eeg": [
        ElectrodePlacement(0, "Fp1", "Left forehead, 10-20 system", 0.9),
        ElectrodePlacement(1, "Fp2", "Right forehead, 10-20 system", 0.9),
        ElectrodePlacement(2, "C3", "Left motor cortex", 0.85),
        ElectrodePlacement(3, "C4", "Right motor cortex", 0.85),
        ElectrodePlacement(4, "O1", "Left occipital", 0.8),
        ElectrodePlacement(5, "O2", "Right occipital", 0.8),
        ElectrodePlacement(6, "T3", "Left temporal", 0.7),
        ElectrodePlacement(7, "T4", "Right temporal", 0.7),
    ],
    "eog": [
        ElectrodePlacement(0, "Left Outer Canthus", "Left eye, outer corner", 0.95),
        ElectrodePlacement(1, "Right Outer Canthus", "Right eye, outer corner", 0.95),
        ElectrodePlacement(2, "Above Left Eye", "Supraorbital, left", 0.85),
        ElectrodePlacement(3, "Below Left Eye", "Infraorbital, left", 0.85),
    ],
}

# ── Modality-specific training instructions ──────────────────────────────

_TRAINING_INSTRUCTIONS_BY_MODALITY = {
    "EMG": {
        "setup": "Attach EMG electrodes to the target muscle groups. Clean skin with alcohol wipe. Apply conductive gel.",
        "rest": "Keep your arm completely relaxed. Avoid any movement or tension. Breathe normally.",
        "record_prefix": "Contract the target muscle for",
        "tips": [
            "Keep consistent contraction intensity across repetitions",
            "Avoid co-contraction of neighboring muscles",
            "Return to full rest between each recording",
        ],
    },
    "EEG": {
        "setup": "Place the EEG headband/cap on your head. Ensure good contact at all electrode sites. Check impedance if possible.",
        "rest": "Close your eyes, relax, and try not to think about anything specific. Breathe slowly.",
        "record_prefix": "Focus on the mental task:",
        "tips": [
            "Minimize eye blinks and jaw clenching during recording",
            "Stay as still as possible — movement creates artifacts",
            "Maintain consistent focus intensity across repetitions",
        ],
    },
    "EOG": {
        "setup": "Place EOG electrodes around the eyes. Two horizontal (outer canthi), two vertical (above/below one eye).",
        "rest": "Look straight ahead at the center fixation point. Avoid blinking.",
        "record_prefix": "Perform the eye movement:",
        "tips": [
            "Make crisp, deliberate eye movements",
            "Return gaze to center between each action",
            "Keep head still — only move your eyes",
        ],
    },
    "ECG": {
        "setup": "Place ECG electrodes: right arm, left arm, and left leg (standard Lead II). Clean skin first.",
        "rest": "Sit still, breathe normally. This captures your baseline heart rhythm.",
        "record_prefix": "Continue the cardiac measurement:",
        "tips": [
            "Avoid movement during recording",
            "Breathe at a normal pace",
            "Stay relaxed — stress affects heart rate variability",
        ],
    },
}


class IntentEngine:
    """Parse natural language → PipelinePlan with multi-provider LLM support."""

    SUPPORTED_PROVIDERS = {"openai", "anthropic", "deepseek"}

    def __init__(self, api_key: str = None, provider: str = "openai",
                 base_url: str = None, model: str = None):
        self.api_key = api_key
        self.provider = provider
        self.base_url = base_url  # custom endpoint for DeepSeek/local models
        self.model = model

    def parse(self, prompt: str) -> PipelinePlan:
        """Parse a natural language prompt into a pipeline plan."""
        if self.api_key:
            try:
                return self._parse_llm(prompt)
            except Exception as e:
                logger.warning("LLM parse failed, falling back to rules: %s", self._format_provider_error(e))
        return self._parse_rules(prompt)

    def set_api_key(self, key: str, provider: str = "openai",
                    base_url: str = None, model: str = None):
        self.api_key = key
        self.provider = provider
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

    def enhance_prompt(self, raw_prompt: str, hardware_context: dict = None) -> str:
        """Enhance a raw user prompt with hardware/system context for the LLM."""
        ctx = hardware_context or {}
        n_channels = ctx.get("n_channels", 8)
        sample_rate = ctx.get("sample_rate", 250)
        profile = ctx.get("signal_profile", "EMG")
        board = ctx.get("board_name", "Synthetic")
        has_model = ctx.get("has_trained_model", False)

        enhanced = f"""KYMA Biosignal Project Request
---
User prompt: "{raw_prompt}"

Hardware context:
- Board: {board}
- Channels: {n_channels}
- Sample rate: {sample_rate} Hz
- Active signal profile: {profile}
- Has trained model: {has_model}

Requirements:
- Generate a complete pipeline plan including electrode placement, gestures/classes, recording protocol, and model selection
- The recording protocol should have step-by-step instructions tailored to the signal modality ({profile})
- Include modality-specific tips for the user during recording
- Consider the channel count when suggesting electrode placements
- The project should output: Python training/inference scripts, firmware for embedded deployment, and a web dashboard"""

        return enhanced

    # ── Rule-based parser ─────────────────────────────────────────────────

    def _parse_rules(self, prompt: str) -> PipelinePlan:
        text = prompt.lower().strip()
        plan = PipelinePlan(task_description=prompt)

        # Detect task type
        for task_type, pattern in _TASK_PATTERNS.items():
            if re.search(pattern, text):
                plan.task_type = task_type
                break

        # Detect modality
        for modality, pattern in _MODALITY_PATTERNS.items():
            if re.search(pattern, text):
                plan.signal_modality = modality
                break

        # Detect temporal mode
        if re.search(r"offline|batch|post.?process|analys[ei]s", text):
            plan.temporal_mode = "offline"
        else:
            plan.temporal_mode = "realtime"

        # Select gesture preset based on modality + keywords
        preset_key = self._detect_preset(text, plan.signal_modality)

        plan.suggested_gestures = _GESTURE_PRESETS.get(preset_key, _GESTURE_PRESETS["hand"])

        # Electrode presets based on modality
        electrode_key = preset_key
        if plan.signal_modality == "EEG":
            electrode_key = "eeg"
        elif plan.signal_modality == "EOG":
            electrode_key = "eog"
        plan.suggested_electrodes = _ELECTRODE_PRESETS.get(electrode_key, _ELECTRODE_PRESETS["hand"])

        # Model recommendations
        n_gestures = len(plan.suggested_gestures)
        if n_gestures <= 3:
            plan.model_recommendations = ["LDA"]
            plan.dataset_targets = {"min_windows": 30, "recommended_windows": 100}
            plan.estimated_duration_min = 5.0
        elif n_gestures <= 5:
            plan.model_recommendations = ["LDA", "TCN"]
            plan.dataset_targets = {"min_windows": 50, "recommended_windows": 200}
            plan.estimated_duration_min = 10.0
        else:
            plan.model_recommendations = ["TCN", "Mamba"]
            plan.dataset_targets = {"min_windows": 100, "recommended_windows": 400}
            plan.estimated_duration_min = 20.0

        # Recording protocol with interactive gates
        plan.recording_protocol = self._generate_protocol(plan)

        # Training instructions (modality-specific guidance per step)
        plan.training_instructions = self._generate_training_instructions(plan)

        # Noise tolerance
        if re.search(r"clean|precise|accurate|medical", text):
            plan.noise_tolerance = 0.2
        elif re.search(r"noisy|robust|outdoor|moving", text):
            plan.noise_tolerance = 0.8
        else:
            plan.noise_tolerance = 0.5

        plan.confidence = 0.7
        plan.explanation = (f"Detected {plan.task_type} task for {plan.signal_modality} "
                           f"with {n_gestures} classes. "
                           f"Recommended: {', '.join(plan.model_recommendations)}.")
        return plan

    def _detect_preset(self, text: str, modality: str) -> str:
        """Detect the gesture preset key from text and modality."""
        if modality == "EMG" and re.search(r"\b(left|right|bilateral|both)\b", text) and re.search(r"\b(flex|forearm|servo|robot|arm)\b", text):
            return "bilateral_forearm"
        if modality == "EEG":
            if re.search(r"motor|imagin|left|right|feet|hand", text):
                return "eeg_motor"
            return "eeg_focus"
        if modality == "EOG":
            return "eog_blink"
        # EMG/ECG presets
        if re.search(r"fatigue|endurance|tired", text):
            return "fatigue"
        elif re.search(r"finger|digit|individual", text):
            return "finger"
        elif re.search(r"grip|grasp|hold", text):
            return "grip"
        elif re.search(r"arm|elbow|shoulder|bicep", text):
            return "arm"
        return "hand"

    def _generate_protocol(self, plan: PipelinePlan) -> List[Dict]:
        if plan.suggested_gestures == _GESTURE_PRESETS.get("bilateral_forearm"):
            return [
                {
                    "step": 1,
                    "instruction": "Relax both forearms completely",
                    "gesture": "rest",
                    "duration_s": 5,
                    "reps": 4,
                    "type": "record",
                    "requires_confirmation": True,
                    "user_prompt": "Keep both arms relaxed. This teaches KYMA the baseline for both channels.",
                },
                {
                    "step": 2,
                    "instruction": "Flex only the left forearm",
                    "gesture": "left_flex",
                    "duration_s": 5,
                    "reps": 5,
                    "type": "record",
                    "requires_confirmation": True,
                    "user_prompt": "Flex your left forearm only. Keep the right forearm relaxed.",
                },
                {"step": 3, "instruction": "Relax both forearms briefly", "duration_s": 3, "type": "rest", "requires_confirmation": False},
                {
                    "step": 4,
                    "instruction": "Flex only the right forearm",
                    "gesture": "right_flex",
                    "duration_s": 5,
                    "reps": 5,
                    "type": "record",
                    "requires_confirmation": True,
                    "user_prompt": "Flex your right forearm only. Keep the left forearm relaxed.",
                },
            ]
        steps = []
        modality = plan.signal_modality
        instructions = _TRAINING_INSTRUCTIONS_BY_MODALITY.get(modality,
            _TRAINING_INSTRUCTIONS_BY_MODALITY["EMG"])

        for i, gesture in enumerate(plan.suggested_gestures):
            if gesture == "rest":
                steps.append({
                    "step": len(steps)+1,
                    "instruction": instructions["rest"],
                    "gesture": gesture, "duration_s": 5, "reps": 3, "type": "record",
                    "requires_confirmation": True,
                    "user_prompt": f"We're going to record your resting baseline. {instructions['rest']}",
                })
            else:
                steps.append({
                    "step": len(steps)+1,
                    "instruction": f"{instructions['record_prefix']} '{gesture}' — hold steady for 5 seconds",
                    "gesture": gesture, "duration_s": 5, "reps": 3, "type": "record",
                    "requires_confirmation": True,
                    "user_prompt": f"Next: '{gesture}'. Get into position. Click 'Start' when ready.",
                })
            if i < len(plan.suggested_gestures) - 1:
                steps.append({
                    "step": len(steps)+1,
                    "instruction": "Relax briefly",
                    "duration_s": 3, "type": "rest",
                    "requires_confirmation": False,
                })
        return steps

    def _generate_training_instructions(self, plan: PipelinePlan) -> List[Dict]:
        """Generate modality-specific user guidance for each training step."""
        if plan.suggested_gestures == _GESTURE_PRESETS.get("bilateral_forearm"):
            return [
                {
                    "phase": "setup",
                    "title": "Bilateral forearm EMG setup",
                    "body": "Connect one EMG pair to the left forearm flexor mass, one EMG pair to the right forearm flexor mass, and the ground/reference clip to the ear lobe.",
                    "tips": [
                        "Record left_flex with the right arm relaxed",
                        "Record right_flex with the left arm relaxed",
                        "Keep both arms relaxed for rest samples",
                    ],
                },
                {"phase": "record", "gesture": "rest", "title": "Record: Rest", "body": "Relax both forearms.", "tips": ["Do not touch the servos during baseline"]},
                {"phase": "record", "gesture": "left_flex", "title": "Record: Left Flex", "body": "Flex only the left forearm.", "tips": ["Right forearm stays relaxed"]},
                {"phase": "record", "gesture": "right_flex", "title": "Record: Right Flex", "body": "Flex only the right forearm.", "tips": ["Left forearm stays relaxed"]},
            ]
        modality = plan.signal_modality
        info = _TRAINING_INSTRUCTIONS_BY_MODALITY.get(modality,
            _TRAINING_INSTRUCTIONS_BY_MODALITY["EMG"])
        instructions = [{
            "phase": "setup",
            "title": f"{modality} Setup",
            "body": info["setup"],
            "tips": info["tips"],
        }]
        for gesture in plan.suggested_gestures:
            if gesture == "rest":
                instructions.append({
                    "phase": "record",
                    "gesture": gesture,
                    "title": "Record Baseline (Rest)",
                    "body": info["rest"],
                    "tips": ["Stay still", "Breathe normally", "This sets the noise floor"],
                })
            else:
                instructions.append({
                    "phase": "record",
                    "gesture": gesture,
                    "title": f"Record: {gesture.replace('_', ' ').title()}",
                    "body": f"{info['record_prefix']} '{gesture}' — hold steady for 5 seconds.",
                    "tips": info["tips"],
                })
        return instructions

    # ── LLM-based parser ──────────────────────────────────────────────────

    def _parse_llm(self, prompt: str) -> PipelinePlan:
        system_msg = """You are KYMA, an AI biosignal pipeline planner. Given a natural language request, 
output a JSON pipeline plan with these fields:
- task_type: "classification" | "regression" | "control"
- task_description: string
- signal_modality: "EMG" | "EEG" | "ECG" | "EOG"
- temporal_mode: "realtime" | "offline"
- noise_tolerance: float 0-1
- suggested_gestures: list of gesture/class names
- model_recommendations: list of "LDA" | "TCN" | "Mamba"
- dataset_targets: {min_windows: int, recommended_windows: int}
- estimated_duration_min: float
- explanation: string explaining your reasoning
- training_instructions: list of {phase, gesture, title, body, tips} for each step
Respond with ONLY valid JSON."""

        if self.provider == "openai":
            return self._call_openai_compatible(
                system_msg, prompt,
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4o-mini",
                auth_header={"Authorization": f"Bearer {self.api_key}"})
        elif self.provider == "anthropic":
            return self._call_anthropic(system_msg, prompt)
        elif self.provider == "deepseek":
            base = self.base_url or "https://api.deepseek.com/v1/chat/completions"
            return self._call_openai_compatible(
                system_msg, prompt,
                url=self._chat_completions_url(base),
                model=self.model or "deepseek-chat",
                auth_header={"Authorization": f"Bearer {self.api_key}"})
        else:
            # Try as OpenAI-compatible with custom base URL
            base = self.base_url or f"https://api.{self.provider}.com/v1/chat/completions"
            return self._call_openai_compatible(
                system_msg, prompt,
                url=self._chat_completions_url(base),
                model=self.model or self.provider,
                auth_header={"Authorization": f"Bearer {self.api_key}"})

    def _chat_completions_url(self, url: str) -> str:
        clean = str(url or "").rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        return f"{clean}/chat/completions"

    def _call_openai_compatible(self, system_msg, prompt, url, model, auth_header):
        """Generic OpenAI-compatible API call (works for OpenAI, DeepSeek, etc.)."""
        payload = {"model": model, "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ], "temperature": 0.3, "max_tokens": 2000}
        if self.provider == "nvidia":
            payload["chat_template_kwargs"] = self._nvidia_chat_template_kwargs()
            payload["stream"] = True
            content = self._stream_openai_compatible(url, payload, auth_header, self._request_timeout(60.0))
        else:
            data = self._post_openai_compatible(url, payload, auth_header, self._request_timeout(60.0))
            content = data["choices"][0]["message"]["content"]
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        return self._dict_to_plan(data, prompt)

    def _call_anthropic(self, system_msg, prompt):
        import httpx
        resp = httpx.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "Content-Type": "application/json",
                      "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000, "system": system_msg,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0)
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        return self._dict_to_plan(data, prompt)

    def _format_provider_error(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) is not None:
            text = getattr(response, "text", "") or ""
            return f"{response.status_code} from {self.provider}: {text[:500]}"
        return str(exc)

    def _request_timeout(self, default: float) -> float:
        return 300.0 if self.provider == "nvidia" else default

    def _nvidia_chat_template_kwargs(self) -> dict:
        model = str(self.model or "").lower()
        if "glm" in model or "qwen" in model:
            return {"enable_thinking": False}
        return {"thinking": False}

    def _post_openai_compatible(self, url: str, payload: dict, auth_header: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**auth_header, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{exc.code} from {self.provider}: {body[:500]}") from exc

    def _stream_openai_compatible(self, url: str, payload: dict, auth_header: dict, timeout: float) -> str:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**auth_header, "Content-Type": "application/json", "Accept": "text/event-stream"},
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

    def _dict_to_plan(self, data: dict, prompt: str) -> PipelinePlan:
        plan = PipelinePlan(
            task_type=data.get("task_type", "classification"),
            task_description=prompt,
            signal_modality=data.get("signal_modality", "EMG"),
            temporal_mode=data.get("temporal_mode", "realtime"),
            noise_tolerance=data.get("noise_tolerance", 0.5),
            suggested_gestures=data.get("suggested_gestures", ["rest", "active"]),
            model_recommendations=data.get("model_recommendations", ["LDA"]),
            dataset_targets=data.get("dataset_targets", {"min_windows": 50, "recommended_windows": 200}),
            estimated_duration_min=data.get("estimated_duration_min", 10.0),
            confidence=0.9,
            explanation=data.get("explanation", "LLM-generated plan"),
        )
        # Use LLM-provided training instructions if available
        plan.training_instructions = data.get("training_instructions",
            self._generate_training_instructions(plan))
        # Electrode presets (TODO: let LLM suggest these too)
        modality_key = plan.signal_modality.lower()
        plan.suggested_electrodes = _ELECTRODE_PRESETS.get(
            modality_key, _ELECTRODE_PRESETS.get("hand", []))
        plan.recording_protocol = self._generate_protocol(plan)
        return plan
