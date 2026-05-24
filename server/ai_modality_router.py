"""Compact modality-to-model routing policy for KYMA's AI copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


MODEL_CATALOG = [
    {
        "id": "tiny_biomoe",
        "name": "TinyBioMoE",
        "modalities": ["eeg", "emg", "ecg"],
        "purpose": "cross-modal biosignal embedding backbone",
        "outputs": ["192-d embedding"],
        "download_path": "sessions/ai_models/downloads/Tiny-BioMoE.pth",
        "expected_size_mb": 89,
        "requires_head": True,
        "notes": "Use this as a shared representation model; it does not emit diagnosis or class labels by itself.",
    },
    {
        "id": "st_eegformer_small",
        "name": "ST-EEGFormer Small",
        "modalities": ["eeg"],
        "purpose": "EEG foundation encoder",
        "outputs": ["512-d embedding"],
        "download_path": "sessions/ai_models/downloads/ST-EEGFormer_small_encoder.safetensors",
        "expected_size_mb": 102,
        "requires_head": True,
        "notes": "EEG-only encoder for embedding extraction and downstream heads.",
    },
    {
        "id": "neurorvq_emg_foundation",
        "name": "NeuroRVQ EMG Foundation",
        "modalities": ["emg"],
        "purpose": "EMG encoder",
        "outputs": ["pooled embedding", "branch features"],
        "download_path": "sessions/ai_models/downloads/NeuroRVQ_EMG_foundation_model_v1.safetensors",
        "expected_size_mb": 445,
        "requires_head": True,
        "notes": "EMG foundation encoder; use pooled features for downstream classification or retrieval.",
    },
]


RESEARCH_HEAD_REGISTRY = [
    {
        "id": "luna_abnormality_head",
        "name": "LUNA EEG Abnormality Head",
        "modalities": ["eeg"],
        "purpose": "research screening for EEG abnormality and slowing",
        "outputs": ["label probabilities", "screening score"],
        "install_path": "sessions/ai_models/condition_heads/luna_abnormality",
        "notes": "Pair with a LUNA-style EEG encoder; use for research triage only, never diagnosis.",
    },
    {
        "id": "artifact_seizure_head",
        "name": "EEG Artifact + Seizure Head",
        "modalities": ["eeg"],
        "purpose": "research screening for seizure-like events and artifact separation",
        "outputs": ["event probability", "artifact vs seizure label"],
        "install_path": "sessions/ai_models/condition_heads/artifact_seizure",
        "notes": "Use only on EEG workflows with validated seizure review protocols.",
    },
    {
        "id": "ecg_fm_arrhythmia_head",
        "name": "ECG-FM Arrhythmia Head",
        "modalities": ["ecg"],
        "purpose": "research screening for atrial fibrillation and related rhythm labels",
        "outputs": ["label probabilities", "risk score"],
        "install_path": "sessions/ai_models/condition_heads/ecg_fm_arrhythmia",
        "notes": "Requires ECG-specific embeddings or fine-tuning; intended for research review only.",
    },
    {
        "id": "sleepfm_disease_head",
        "name": "SleepFM Disease Head",
        "modalities": ["eeg", "ecg", "emg", "eog", "resp"],
        "purpose": "multimodal overnight PSG disease-risk screening",
        "outputs": ["condition risk scores", "sleep-stage-aware embeddings"],
        "install_path": "sessions/ai_models/condition_heads/sleepfm_disease",
        "notes": "Only valid in multimodal overnight PSG-style workflows, not single-channel spot checks.",
    },
    {
        "id": "emg_neuromuscular_head",
        "name": "EMG Neuromuscular Screen Head",
        "modalities": ["emg"],
        "purpose": "research screening for neuromuscular pattern differences and fatigue-like behavior",
        "outputs": ["class probabilities", "screening score"],
        "install_path": "sessions/ai_models/condition_heads/emg_neuromuscular",
        "notes": "EMG clinical-condition heads remain sparse and should be treated as exploratory only.",
    },
]


MODALITY_GUIDANCE = {
    "emg": {
        "strategy": "Prefer EMG-specific encoders first, then cross-modal embeddings, then local task heads.",
        "why": "EMG needs artifact checking, activation quality, readiness, and optional gesture downstream heads.",
    },
    "eeg": {
        "strategy": "Prefer EEG-specific encoders first, then cross-modal embeddings, then local task heads.",
        "why": "EEG benefits from foundation embeddings for downstream rhythm, workload, and artifact workflows.",
    },
    "ecg": {
        "strategy": "Use cross-modal embeddings plus local QA or artifact models first.",
        "why": "The installed model set currently has a shared biosignal encoder but no ECG-only foundation encoder in KYMA.",
    },
    "eog": {
        "strategy": "Use heuristics and local task heads only until an EOG-specific checkpoint is installed.",
        "why": "No EOG foundation checkpoint is currently registered in KYMA.",
    },
    "eda": {
        "strategy": "Use heuristics and lightweight local heads only.",
        "why": "EDA is slow-moving and the current external model set is not targeted at it.",
    },
    "ppg": {
        "strategy": "Use heuristics and lightweight local heads only.",
        "why": "PPG currently has no registered external encoder in KYMA.",
    },
    "resp": {
        "strategy": "Use heuristics and lightweight local heads only.",
        "why": "Respiration currently has no registered external encoder in KYMA.",
    },
    "temp": {
        "strategy": "Use heuristics and lightweight local heads only.",
        "why": "Temperature is trend-heavy and not covered by the current external encoder set.",
    },
}


LOCAL_TASKS = [
    {
        "task": "signal_qa",
        "when": "always",
        "output": "score_0_100 plus label",
    },
    {
        "task": "artifact_classifier",
        "when": "artifact review or auto-scan",
        "output": "artifact label plus confidence",
    },
    {
        "task": "fatigue_readiness",
        "when": "motor effort or endurance context",
        "output": "readiness score plus label",
    },
    {
        "task": "condition_screen",
        "when": "only when a research-only condition head is explicitly installed",
        "output": "screening labels or ranked risk scores",
    },
]


class AIModalityRouter:
    """Produces a small routing brief for each AI pass."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def plan(self, snapshot: Dict[str, Any], local_status: Dict[str, Any]) -> Dict[str, Any]:
        profile = dict(snapshot.get("profile") or {})
        modality = str(profile.get("key") or "signal").strip().lower()
        foundation_runtime = {
            str(item.get("id") or ""): dict(item)
            for item in list(local_status.get("foundation_models") or [])
        }
        guidance = MODALITY_GUIDANCE.get(
            modality,
            {
                "strategy": "Use heuristics first and only enable models with explicit modality support.",
                "why": "Unknown or unsupported modality.",
            },
        )
        primary_models = [
            self._model_status(item, foundation_runtime.get(str(item.get("id") or "")))
            for item in MODEL_CATALOG
            if modality in item["modalities"]
        ]
        installed_models = [item for item in primary_models if item["status"] in {"ready", "adapter_ready"}]
        partial_models = [item for item in primary_models if item["status"] == "partial"]

        return {
            "modality": modality,
            "family": str(profile.get("family") or ""),
            "strategy": guidance["strategy"],
            "why": guidance["why"],
            "primary_models": primary_models,
            "research_heads": [self._research_head_status(item) for item in RESEARCH_HEAD_REGISTRY if modality in item["modalities"]],
            "installed_model_ids": [item["id"] for item in installed_models],
            "partial_model_ids": [item["id"] for item in partial_models],
            "local_tasks": LOCAL_TASKS,
            "local_runtime_tasks": list(local_status.get("tasks") or []),
            "output_contract": [
                "Foundation encoders output embeddings or features, not direct diagnosis.",
                "Local task heads output QA scores, artifact labels, or readiness scores.",
                "Condition heads, when installed, are research-only screening models and must not be presented as diagnoses.",
                "Heuristics remain active for all modalities as a fallback and sanity layer.",
            ],
        }

    def prompt(self, plan: Dict[str, Any]) -> str:
        model_bits: List[str] = []
        for item in list(plan.get("primary_models") or [])[:4]:
            outputs = ", ".join(item.get("outputs") or [])
            model_bits.append(f"{item['name']} [{item['status']}] -> {outputs}")
        research_bits: List[str] = []
        for item in list(plan.get("research_heads") or [])[:3]:
            outputs = ", ".join(item.get("outputs") or [])
            research_bits.append(f"{item['name']} [{item['status']}] -> {outputs}")
        local_bits = ", ".join(str(task.get("task") or "") for task in list(plan.get("local_tasks") or []))
        return (
            f"Routing policy: modality={plan.get('modality')} ({plan.get('family')}). "
            f"Strategy: {plan.get('strategy')} "
            f"Reason: {plan.get('why')} "
            f"Primary models: {'; '.join(model_bits) if model_bits else 'none'} "
            f"Research condition heads: {'; '.join(research_bits) if research_bits else 'none'} "
            f"Local task heads: {local_bits or 'none'}. "
            "Never treat foundation embeddings as direct diagnoses. Use local task outputs for QA, artifacts, readiness, and research-only screening. "
            "If a model is partial or missing, ignore it and fall back to heuristics plus any ready local heads."
        ).strip()

    def _model_status(self, item: Dict[str, Any], runtime: Dict[str, Any] | None = None) -> Dict[str, Any]:
        expected_size_mb = float(item.get("expected_size_mb") or 0.0)
        path = self.repo_root / str(item.get("download_path") or "")
        if not path.exists():
            status = "missing"
            size_mb = 0.0
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            if expected_size_mb and size_mb < expected_size_mb * 0.95:
                status = "partial"
            else:
                status = "ready"
        runtime_loaded = bool((runtime or {}).get("loaded"))
        runtime_validated = bool((runtime or {}).get("validated"))
        inference_ready = bool((runtime or {}).get("inference_ready"))
        if runtime_loaded and runtime_validated:
            status = "ready" if inference_ready else "adapter_ready"
        return {
            "id": str(item.get("id") or ""),
            "name": str(item.get("name") or ""),
            "status": status,
            "purpose": str(item.get("purpose") or ""),
            "outputs": list(item.get("outputs") or []),
            "requires_head": bool(item.get("requires_head")),
            "download_path": str(item.get("download_path") or ""),
            "expected_size_mb": expected_size_mb,
            "local_size_mb": round(size_mb, 1),
            "adapter": str((runtime or {}).get("adapter") or ""),
            "validated": runtime_validated,
            "inference_ready": inference_ready,
            "summary": dict((runtime or {}).get("summary") or {}),
            "notes": str(item.get("notes") or ""),
        }

    def _research_head_status(self, item: Dict[str, Any]) -> Dict[str, Any]:
        install_path = str(item.get("install_path") or "").strip()
        path = (self.repo_root / install_path) if install_path else None
        if path and path.exists():
            status = "ready"
        else:
            status = "missing"
        return {
            "id": str(item.get("id") or ""),
            "name": str(item.get("name") or ""),
            "status": status,
            "purpose": str(item.get("purpose") or ""),
            "outputs": list(item.get("outputs") or []),
            "install_path": install_path,
            "research_only": True,
            "notes": str(item.get("notes") or ""),
        }
