"""Offline dataset and experiment helpers for saved KYMA sessions."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from biosignal_profiles import get_profile
from classifiers import LDAClassifier, get_classifier
from config import config
from ecg_pipeline import ECGPipeline
from eda_pipeline import EDAPipeline
from eeg_pipeline import EEGPipeline
from emg_pipeline import EMGPipeline
from eog_pipeline import EOGPipeline
from ppg_pipeline import PPGPipeline
from resp_pipeline import RespirationPipeline
from session_recorder import SessionRecorder
from temp_pipeline import TemperaturePipeline

logger = logging.getLogger(__name__)


PIPELINE_REGISTRY = {
    "emg": EMGPipeline,
    "eeg": EEGPipeline,
    "ecg": ECGPipeline,
    "eog": EOGPipeline,
    "eda": EDAPipeline,
    "ppg": PPGPipeline,
    "resp": RespirationPipeline,
    "temp": TemperaturePipeline,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:48] or fallback


class DatasetManager:
    """Create compact dataset manifests from saved sessions."""

    def __init__(self, recorder: SessionRecorder):
        self.recorder = recorder
        self.datasets_dir = Path(config.data_dir) / "datasets"
        self.experiments_dir = Path(config.data_dir) / "experiments"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in sorted(self.datasets_dir.glob("*.json"), reverse=True):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                out.append(self._dataset_summary(data))
            except Exception as exc:
                logger.warning("Failed to read dataset manifest %s: %s", path, exc)
        return out

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        path = self.datasets_dir / f"{dataset_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def create_dataset(
        self,
        session_ids: List[str],
        name: str = "",
        profile_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not session_ids:
            raise ValueError("Select at least one session")

        target_profile = (profile_key or config.signal_profile_name or "emg").lower()
        profile = get_profile(target_profile)
        pipeline = self._make_pipeline(profile.key)
        segment_samples = int(getattr(pipeline, "_segment_samples", config.window_size_samples))
        stride_samples = int(getattr(pipeline, "_record_stride_samples", config.window_increment_samples))

        session_entries: List[Dict[str, Any]] = []
        label_counts = {label: 0 for label in profile.class_labels}
        total_windows = 0
        labeled_sessions = 0
        warnings: List[str] = []

        for session_id in session_ids:
            summary = self.recorder.get_session_summary(session_id)
            meta = self.recorder.get_session_meta(session_id)
            if not summary or not meta:
                raise FileNotFoundError(f"Session not found: {session_id}")

            session_profile = str(summary.get("signal_profile") or "").lower()
            if session_profile != profile.key:
                raise ValueError(
                    f"Session {session_id} uses profile '{session_profile or 'unknown'}', "
                    f"but the active dataset profile is '{profile.key}'"
                )

            intervals = self._label_intervals(meta, profile.class_labels)
            if intervals:
                labeled_sessions += 1

            session_entry = {
                "session_id": session_id,
                "label": meta.get("label") or "",
                "created_at_utc": summary.get("created_at_utc"),
                "duration_s": float(summary.get("duration_s") or 0.0),
                "n_samples": int(summary.get("n_samples") or 0),
                "stream_source": summary.get("stream_source") or "unknown",
                "subject_id": summary.get("subject_id") or "",
                "condition": summary.get("condition") or "",
                "session_group_id": summary.get("session_group_id") or "",
                "protocol_key": summary.get("protocol_key") or "",
                "protocol_title": summary.get("protocol_title") or "",
                "trial_index": summary.get("trial_index"),
                "repetition_index": summary.get("repetition_index"),
                "signal_profile": session_profile,
                "labeled": bool(intervals),
                "label_intervals": [],
            }

            if not intervals:
                warnings.append(f"{session_id}: no compatible labels were found in label field or events")

            for interval in intervals:
                est = self._estimate_window_count(
                    start_s=float(interval["start_s"]),
                    end_s=float(interval["end_s"]),
                    sample_rate=int(summary.get("sample_rate") or config.sample_rate),
                    segment_samples=segment_samples,
                    stride_samples=stride_samples,
                )
                label_counts.setdefault(interval["label"], 0)
                label_counts[interval["label"]] += est
                total_windows += est
                session_entry["label_intervals"].append(
                    {
                        **interval,
                        "estimated_windows": est,
                    }
                )

            session_entries.append(session_entry)

        nonzero_labels = {k: int(v) for k, v in label_counts.items() if int(v) > 0}
        split_group_stats = self._split_group_stats(session_entries, tuple(sorted(nonzero_labels.keys())))
        dataset_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe_name(name, 'dataset')}"
        manifest = {
            "dataset_id": dataset_id,
            "name": name.strip() or dataset_id,
            "created_at_utc": _utc_now_iso(),
            "signal_profile": profile.key,
            "signal_profile_name": profile.display_name,
            "segment_duration_s": round(segment_samples / config.sample_rate, 3),
            "segment_stride_s": round(stride_samples / config.sample_rate, 3),
            "window_size_samples": segment_samples,
            "window_stride_samples": stride_samples,
            "n_sessions": len(session_entries),
            "labeled_sessions": labeled_sessions,
            "unlabeled_sessions": max(len(session_entries) - labeled_sessions, 0),
            "estimated_windows": int(total_windows),
            "labels_present": sorted(nonzero_labels.keys()),
            "label_counts": nonzero_labels,
            "ready": len(nonzero_labels) >= 2 and total_windows >= 8,
            "subjects_present": split_group_stats["subjects_present"],
            "conditions_present": split_group_stats["conditions_present"],
            "session_groups_present": split_group_stats["session_groups_present"],
            "protocols_present": split_group_stats["protocols_present"],
            "full_session_groups": split_group_stats["full_session_groups"],
            "ready_for_loso": split_group_stats["ready_for_loso"],
            "full_subjects": split_group_stats["full_subjects"],
            "ready_for_subject_holdout": split_group_stats["ready_for_subject_holdout"],
            "warnings": warnings,
            "session_entries": session_entries,
        }

        with open(self.datasets_dir / f"{dataset_id}.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest

    def _dataset_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_id": data.get("dataset_id"),
            "name": data.get("name") or data.get("dataset_id"),
            "created_at_utc": data.get("created_at_utc"),
            "signal_profile": data.get("signal_profile"),
            "signal_profile_name": data.get("signal_profile_name"),
            "n_sessions": int(data.get("n_sessions") or 0),
            "labeled_sessions": int(data.get("labeled_sessions") or 0),
            "estimated_windows": int(data.get("estimated_windows") or 0),
            "labels_present": data.get("labels_present") or [],
            "ready": bool(data.get("ready")),
            "ready_for_loso": bool(data.get("ready_for_loso")),
            "ready_for_subject_holdout": bool(data.get("ready_for_subject_holdout")),
            "subjects_present": data.get("subjects_present") or [],
            "session_groups_present": data.get("session_groups_present") or [],
            "full_session_groups": data.get("full_session_groups") or [],
            "full_subjects": data.get("full_subjects") or [],
        }

    def _make_pipeline(self, profile_key: str):
        cls = PIPELINE_REGISTRY.get(profile_key)
        if cls is None:
            raise ValueError(f"No trainable pipeline registered for profile '{profile_key}'")
        return cls()

    def _events_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        session_path = self.recorder.get_session_path(session_id)
        if not session_path:
            return []
        events_path = Path(session_path) / "events.json"
        if not events_path.exists():
            return []
        try:
            with open(events_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Failed to read events for %s: %s", session_id, exc)
            return []

    def _label_intervals(self, meta: Dict[str, Any], class_labels: Tuple[str, ...]) -> List[Dict[str, Any]]:
        duration_s = float(meta.get("duration_s") or 0.0)
        session_id = meta.get("session_id") or ""
        events = self._events_for_session(session_id)
        valid = {str(label) for label in class_labels}
        intervals: List[Dict[str, Any]] = []

        current_label: Optional[str] = None
        current_start: Optional[float] = None
        for event in events:
            event_type = str(event.get("type") or "")
            ts = float(event.get("timestamp_s") or 0.0)
            data = event.get("data") or {}
            label = str(data.get("label") or data.get("gesture") or "").strip()

            if event_type == "train_start":
                if current_label is not None and current_start is not None and ts > current_start:
                    intervals.append(
                        {
                            "label": current_label,
                            "start_s": round(current_start, 4),
                            "end_s": round(ts, 4),
                            "source": "events",
                        }
                    )
                if label in valid:
                    current_label = label
                    current_start = ts
                else:
                    current_label = None
                    current_start = None
            elif event_type == "train_stop" and current_label is not None and current_start is not None:
                end_s = max(ts, current_start)
                if end_s > current_start:
                    intervals.append(
                        {
                            "label": current_label,
                            "start_s": round(current_start, 4),
                            "end_s": round(end_s, 4),
                            "source": "events",
                        }
                    )
                current_label = None
                current_start = None

        if current_label is not None and current_start is not None and duration_s > current_start:
            intervals.append(
                {
                    "label": current_label,
                    "start_s": round(current_start, 4),
                    "end_s": round(duration_s, 4),
                    "source": "events",
                }
            )

        if intervals:
            return [row for row in intervals if row["label"] in valid and row["end_s"] > row["start_s"]]

        session_label = str(meta.get("label") or "").strip()
        if session_label in valid and duration_s > 0:
            return [
                {
                    "label": session_label,
                    "start_s": 0.0,
                    "end_s": round(duration_s, 4),
                    "source": "session_label",
                }
            ]
        return []

    def _estimate_window_count(
        self,
        start_s: float,
        end_s: float,
        sample_rate: int,
        segment_samples: int,
        stride_samples: int,
    ) -> int:
        start_idx = max(0, int(round(start_s * sample_rate)))
        end_idx = max(start_idx, int(round(end_s * sample_rate)))
        length = max(end_idx - start_idx, 0)
        if length < segment_samples:
            return 0
        return 1 + max(0, (length - segment_samples) // max(stride_samples, 1))

    def _split_group_stats(
        self,
        session_entries: List[Dict[str, Any]],
        labels_present: Tuple[str, ...],
    ) -> Dict[str, Any]:
        subjects = sorted(
            {
                str(entry.get("subject_id") or "").strip()
                for entry in session_entries
                if str(entry.get("subject_id") or "").strip()
            }
        )
        conditions = sorted(
            {
                str(entry.get("condition") or "").strip()
                for entry in session_entries
                if str(entry.get("condition") or "").strip()
            }
        )
        protocol_titles = sorted(
            {
                str(entry.get("protocol_title") or entry.get("protocol_key") or "").strip()
                for entry in session_entries
                if str(entry.get("protocol_title") or entry.get("protocol_key") or "").strip()
            }
        )

        group_map: Dict[str, set] = {}
        group_names: List[str] = []
        subject_map: Dict[str, set] = {}
        label_set = set(labels_present)
        for entry in session_entries:
            group_id = str(entry.get("session_group_id") or entry.get("session_id") or "").strip()
            subject_id = str(entry.get("subject_id") or "").strip()
            for interval in entry.get("label_intervals") or []:
                label = str(interval.get("label") or "").strip()
                if not label:
                    continue
                if group_id:
                    if group_id not in group_map:
                        group_names.append(group_id)
                        group_map[group_id] = set()
                    group_map[group_id].add(label)
                if subject_id:
                    subject_map.setdefault(subject_id, set()).add(label)

        full_groups = [
            group_id
            for group_id in group_names
            if label_set and label_set.issubset(group_map.get(group_id, set()))
        ]
        full_subjects = [
            subject_id
            for subject_id in subjects
            if label_set and label_set.issubset(subject_map.get(subject_id, set()))
        ]
        return {
            "subjects_present": subjects,
            "conditions_present": conditions,
            "session_groups_present": group_names,
            "protocols_present": protocol_titles,
            "full_session_groups": full_groups,
            "ready_for_loso": len(full_groups) >= 2,
            "full_subjects": full_subjects,
            "ready_for_subject_holdout": len(full_subjects) >= 2,
        }


class ExperimentManager:
    """Run offline experiments from saved dataset manifests."""

    def __init__(self, recorder: SessionRecorder, datasets: DatasetManager):
        self.recorder = recorder
        self.datasets = datasets
        self.experiments_dir = datasets.experiments_dir

    def list_experiments(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in sorted(self.experiments_dir.glob("*.json"), reverse=True):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                out.append(self._experiment_summary(data))
            except Exception as exc:
                logger.warning("Failed to read experiment report %s: %s", path, exc)
        return out

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def run_experiment(
        self,
        dataset_id: str,
        classifier: str = "LDA",
        notes: str = "",
        split_strategy: str = "temporal_holdout",
        holdout_fraction: float = 0.25,
        holdout_gap_s: float = 0.2,
    ) -> Dict[str, Any]:
        dataset = self.datasets.get_dataset(dataset_id)
        if not dataset:
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")

        profile_key = str(dataset.get("signal_profile") or "").lower()
        if profile_key != config.signal_profile_name:
            raise ValueError(
                f"Active profile is '{config.signal_profile_name}', but dataset '{dataset_id}' uses '{profile_key}'. "
                "Switch the UI signal profile before running the experiment."
            )

        split_strategy = (split_strategy or "temporal_holdout").strip().lower()
        if split_strategy not in {"random_window", "temporal_holdout", "leave_one_session_out", "leave_one_subject_out"}:
            raise ValueError(
                "Split strategy must be 'random_window', 'temporal_holdout', 'leave_one_session_out', or 'leave_one_subject_out'"
            )

        holdout_fraction = float(holdout_fraction)
        holdout_gap_s = max(float(holdout_gap_s), 0.0)

        profile = get_profile(profile_key)
        class_labels = tuple(str(label) for label in profile.class_labels)
        pipeline = self.datasets._make_pipeline(profile_key)
        segment_samples = int(getattr(pipeline, "_segment_samples", config.window_size_samples))
        stride_samples = int(getattr(pipeline, "_record_stride_samples", config.window_increment_samples))

        started = time.perf_counter()
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{classifier.lower()}"
        report: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset.get("name") or dataset_id,
            "created_at_utc": _utc_now_iso(),
            "signal_profile": profile_key,
            "signal_profile_name": dataset.get("signal_profile_name"),
            "classifier": classifier,
            "notes": notes.strip(),
            "segment_duration_s": round(segment_samples / config.sample_rate, 3),
            "segment_stride_s": round(stride_samples / config.sample_rate, 3),
            "split_strategy": split_strategy,
            "holdout_fraction": round(holdout_fraction, 3),
            "holdout_gap_s": round(holdout_gap_s, 3),
        }

        try:
            records, build_stats = self._load_dataset_records(
                dataset=dataset,
                class_labels=class_labels,
                segment_samples=segment_samples,
                stride_samples=stride_samples,
            )
            report["window_build"] = build_stats
            report["n_windows"] = len(records)
            report["class_counts"] = self._count_indexed_labels(
                [int(record["label_idx"]) for record in records],
                class_labels,
            )

            if len(records) < 8:
                raise ValueError(f"Need at least 8 labeled segments, found {len(records)}")
            if len({int(record["label_idx"]) for record in records}) < 2:
                raise ValueError("Need at least 2 distinct labels to run an experiment")

            train_records, val_records, split_info = self._split_records(
                records=records,
                class_labels=class_labels,
                split_strategy=split_strategy,
                holdout_fraction=holdout_fraction,
                holdout_gap_s=holdout_gap_s,
                stride_samples=stride_samples,
            )
            report["split"] = split_info

            if not train_records:
                raise ValueError("The training split is empty. Record longer labeled sessions or reduce the holdout settings.")
            if not val_records:
                raise ValueError("The validation split is empty. Record longer labeled sessions or reduce the holdout settings.")
            if len({int(record["label_idx"]) for record in train_records}) < 2:
                raise ValueError("The training split must contain at least 2 labels.")

            result = self._fit_and_evaluate(
                pipeline=pipeline,
                classifier=classifier,
                class_labels=class_labels,
                train_records=train_records,
                val_records=val_records,
            )

            report.update(
                {
                    "status": "completed" if result.get("success") else "failed",
                    "result": result,
                    "duration_s": round(time.perf_counter() - started, 3),
                }
            )
        except Exception as exc:
            report.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "duration_s": round(time.perf_counter() - started, 3),
                }
            )

        with open(self.experiments_dir / f"{experiment_id}.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return report

    def _load_dataset_records(
        self,
        dataset: Dict[str, Any],
        class_labels: Tuple[str, ...],
        segment_samples: int,
        stride_samples: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        stats = {
            "sessions_used": 0,
            "intervals_used": 0,
            "intervals_skipped": 0,
        }

        for entry in dataset.get("session_entries") or []:
            intervals = entry.get("label_intervals") or []
            if not intervals:
                continue
            session_id = str(entry.get("session_id") or "")
            raw_path = self.recorder.get_signal_path(session_id)
            if not raw_path:
                stats["intervals_skipped"] += len(intervals)
                continue

            try:
                raw = np.loadtxt(raw_path, delimiter=",", skiprows=1)
            except Exception as exc:
                logger.warning("Failed to load session signal for experiment: %s", exc)
                stats["intervals_skipped"] += len(intervals)
                continue

            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 2:
                stats["intervals_skipped"] += len(intervals)
                continue

            signal = raw[:, 1:].T.astype(np.float32)
            sample_rate = int(config.sample_rate)
            session_used = False

            for interval_index, interval in enumerate(intervals):
                label_name = str(interval.get("label") or "")
                if label_name not in class_labels:
                    stats["intervals_skipped"] += 1
                    continue

                start_idx = max(0, int(round(float(interval.get("start_s") or 0.0) * sample_rate)))
                end_idx = min(signal.shape[1], int(round(float(interval.get("end_s") or 0.0) * sample_rate)))
                if end_idx - start_idx < segment_samples:
                    stats["intervals_skipped"] += 1
                    continue

                interval_id = f"{session_id}:{interval_index}:{label_name}"
                label_idx = class_labels.index(label_name)
                interval_records: List[Dict[str, Any]] = []
                for offset in range(0, end_idx - start_idx - segment_samples + 1, max(stride_samples, 1)):
                    begin = start_idx + offset
                    end = begin + segment_samples
                    interval_records.append(
                        {
                            "window": signal[:, begin:end].copy(),
                            "label_idx": label_idx,
                            "label_name": label_name,
                            "session_id": session_id,
                            "subject_id": str(entry.get("subject_id") or ""),
                            "condition": str(entry.get("condition") or ""),
                            "session_group_id": str(entry.get("session_group_id") or session_id),
                            "protocol_key": str(entry.get("protocol_key") or ""),
                            "protocol_title": str(entry.get("protocol_title") or ""),
                            "created_at_utc": str(entry.get("created_at_utc") or ""),
                            "interval_id": interval_id,
                            "interval_source": interval.get("source") or "unknown",
                            "window_start_idx": int(begin),
                            "window_end_idx": int(end),
                            "window_start_s": round(begin / sample_rate, 4),
                            "window_end_s": round(end / sample_rate, 4),
                        }
                    )

                if not interval_records:
                    stats["intervals_skipped"] += 1
                    continue

                records.extend(interval_records)
                stats["intervals_used"] += 1
                session_used = True

            if session_used:
                stats["sessions_used"] += 1

        return records, stats

    def _split_records(
        self,
        records: List[Dict[str, Any]],
        class_labels: Tuple[str, ...],
        split_strategy: str,
        holdout_fraction: float,
        holdout_gap_s: float,
        stride_samples: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if split_strategy == "random_window":
            train_records, val_records, warnings = self._random_window_split(records, holdout_fraction, class_labels)
            gap_windows = 0
            extra_info: Dict[str, Any] = {}
        elif split_strategy == "temporal_holdout":
            gap_windows = int(math.ceil((holdout_gap_s * config.sample_rate) / max(stride_samples, 1))) if holdout_gap_s > 0 else 0
            train_records, val_records, warnings = self._temporal_holdout_split(records, holdout_fraction, gap_windows)
            extra_info = {}
        elif split_strategy == "leave_one_session_out":
            gap_windows = 0
            train_records, val_records, warnings, extra_info = self._leave_one_session_out_split(records, class_labels)
        else:
            gap_windows = 0
            train_records, val_records, warnings, extra_info = self._leave_one_subject_out_split(records, class_labels)

        split_info = {
            "strategy": split_strategy,
            "holdout_fraction": round(
                holdout_fraction if split_strategy not in {"leave_one_session_out", "leave_one_subject_out"} else 0.0,
                3,
            ),
            "holdout_gap_s": round(holdout_gap_s if split_strategy == "temporal_holdout" else 0.0, 3),
            "gap_windows": int(gap_windows),
            "train_windows": len(train_records),
            "val_windows": len(val_records),
            "train_sessions": len({str(record["session_id"]) for record in train_records}),
            "val_sessions": len({str(record["session_id"]) for record in val_records}),
            "train_session_groups": sorted({str(record.get("session_group_id") or record["session_id"]) for record in train_records}),
            "val_session_groups": sorted({str(record.get("session_group_id") or record["session_id"]) for record in val_records}),
            "train_subjects": sorted({str(record.get("subject_id") or "").strip() for record in train_records if str(record.get("subject_id") or "").strip()}),
            "val_subjects": sorted({str(record.get("subject_id") or "").strip() for record in val_records if str(record.get("subject_id") or "").strip()}),
            "train_class_counts": self._count_indexed_labels(
                [int(record["label_idx"]) for record in train_records],
                class_labels,
            ),
            "val_class_counts": self._count_indexed_labels(
                [int(record["label_idx"]) for record in val_records],
                class_labels,
            ),
            "warnings": warnings,
        }
        split_info.update(extra_info)
        return train_records, val_records, split_info

    def _random_window_split(
        self,
        records: List[Dict[str, Any]],
        holdout_fraction: float,
        class_labels: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        class_labels = class_labels or tuple(config.class_labels)
        rng = np.random.default_rng(42)
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(int(record["label_idx"]), []).append(record)

        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for label_idx, label_records in grouped.items():
            label_records = list(label_records)
            if len(label_records) < 2:
                train_records.extend(label_records)
                warnings.append(
                    f"Label '{class_labels[label_idx]}' has only {len(label_records)} window(s); it stays in train only."
                )
                continue

            order = rng.permutation(len(label_records))
            n_val = max(1, int(round(len(label_records) * holdout_fraction)))
            n_val = min(n_val, len(label_records) - 1)

            shuffled = [label_records[int(idx)] for idx in order]
            val_records.extend(shuffled[:n_val])
            train_records.extend(shuffled[n_val:])

        return train_records, val_records, warnings

    def _temporal_holdout_split(
        self,
        records: List[Dict[str, Any]],
        holdout_fraction: float,
        gap_windows: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record["interval_id"]), []).append(record)

        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for interval_id, interval_records in grouped.items():
            ordered = sorted(interval_records, key=lambda item: int(item["window_start_idx"]))
            n = len(ordered)
            if n < 2:
                train_records.extend(ordered)
                warnings.append(f"{interval_id}: only {n} window(s), so no validation holdout was possible.")
                continue

            n_val = max(1, int(round(n * holdout_fraction)))
            n_val = min(n_val, n - 1)
            local_gap = gap_windows
            train_end = n - n_val - local_gap
            if train_end < 1:
                local_gap = max(0, n - n_val - 1)
                train_end = n - n_val - local_gap
            if train_end < 1:
                train_end = n - 1
                n_val = 1
                local_gap = 0

            if train_end < 1:
                train_records.extend(ordered)
                warnings.append(f"{interval_id}: interval too short for temporal holdout; all windows kept in train.")
                continue

            if local_gap < gap_windows:
                warnings.append(
                    f"{interval_id}: reduced temporal gap from {gap_windows} to {local_gap} window(s) to keep train and validation non-empty."
                )

            train_records.extend(ordered[:train_end])
            val_records.extend(ordered[-n_val:])

        return train_records, val_records, warnings

    def _leave_one_session_out_split(
        self,
        records: List[Dict[str, Any]],
        class_labels: Tuple[str, ...],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            group_id = str(record.get("session_group_id") or record.get("session_id") or "").strip()
            grouped.setdefault(group_id, []).append(record)

        labels_present = {class_labels[int(record["label_idx"])] for record in records}
        full_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
        for group_id, group_records in grouped.items():
            group_labels = {class_labels[int(record["label_idx"])] for record in group_records}
            if labels_present.issubset(group_labels):
                full_groups.append((group_id, group_records))

        if len(full_groups) < 2:
            raise ValueError(
                "Leave-one-session-out requires at least 2 protocol runs whose session_group_id covers every label in the dataset."
            )

        full_groups.sort(
            key=lambda item: (
                max(str(record.get("created_at_utc") or "") for record in item[1]),
                item[0],
            )
        )
        holdout_group_id, val_records = full_groups[-1]
        train_records = [record for record in records if str(record.get("session_group_id") or record.get("session_id") or "") != holdout_group_id]
        warnings: List[str] = []

        holdout_subjects = sorted(
            {
                str(record.get("subject_id") or "").strip()
                for record in val_records
                if str(record.get("subject_id") or "").strip()
            }
        )
        holdout_conditions = sorted(
            {
                str(record.get("condition") or "").strip()
                for record in val_records
                if str(record.get("condition") or "").strip()
            }
        )
        return train_records, val_records, warnings, {
            "holdout_group_id": holdout_group_id,
            "holdout_subjects": holdout_subjects,
            "holdout_conditions": holdout_conditions,
            "full_group_candidates": [group_id for group_id, _ in full_groups],
        }

    def _leave_one_subject_out_split(
        self,
        records: List[Dict[str, Any]],
        class_labels: Tuple[str, ...],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            subject_id = str(record.get("subject_id") or "").strip()
            if not subject_id:
                raise ValueError(
                    "Leave-one-subject-out requires subject_id metadata on every session. "
                    "Record future sessions with the Subject field filled in."
                )
            grouped.setdefault(subject_id, []).append(record)

        labels_present = {class_labels[int(record["label_idx"])] for record in records}
        full_subjects: List[Tuple[str, List[Dict[str, Any]]]] = []
        for subject_id, subject_records in grouped.items():
            subject_labels = {class_labels[int(record["label_idx"])] for record in subject_records}
            if labels_present.issubset(subject_labels):
                full_subjects.append((subject_id, subject_records))

        if len(full_subjects) < 2:
            raise ValueError(
                "Leave-one-subject-out requires at least 2 subjects whose recordings cover every label in the dataset."
            )

        full_subjects.sort(
            key=lambda item: (
                max(str(record.get("created_at_utc") or "") for record in item[1]),
                item[0],
            )
        )
        holdout_subject_id, val_records = full_subjects[-1]
        train_records = [
            record
            for record in records
            if str(record.get("subject_id") or "").strip() != holdout_subject_id
        ]
        warnings: List[str] = []
        holdout_groups = sorted(
            {
                str(record.get("session_group_id") or record.get("session_id") or "").strip()
                for record in val_records
                if str(record.get("session_group_id") or record.get("session_id") or "").strip()
            }
        )
        holdout_conditions = sorted(
            {
                str(record.get("condition") or "").strip()
                for record in val_records
                if str(record.get("condition") or "").strip()
            }
        )
        return train_records, val_records, warnings, {
            "holdout_subject_id": holdout_subject_id,
            "holdout_session_groups": holdout_groups,
            "holdout_conditions": holdout_conditions,
            "full_subject_candidates": [subject_id for subject_id, _ in full_subjects],
        }

    def _fit_and_evaluate(
        self,
        pipeline: Any,
        classifier: str,
        class_labels: Tuple[str, ...],
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        train_windows = [record["window"] for record in train_records]
        train_labels = [int(record["label_idx"]) for record in train_records]
        val_windows = [record["window"] for record in val_records]
        val_labels = np.asarray([int(record["label_idx"]) for record in val_records], dtype=np.int32)

        if classifier == "LDA":
            X_train = pipeline.extract_features_batch(train_windows)
            X_val = pipeline.extract_features_batch(val_windows)
            clf = LDAClassifier()
            fit_result = clf.fit_split(
                X_train,
                np.asarray(train_labels, dtype=np.int32),
                X_val,
                val_labels,
            )
            y_pred = clf.predict_batch(X_val) if len(X_val) else np.asarray([], dtype=np.int32)
            augmented_train_windows = len(train_windows)
        elif classifier in ("TCN", "Mamba"):
            aug_windows, aug_labels = (
                pipeline._augment_windows(train_windows, train_labels, factor=3)
                if hasattr(pipeline, "_augment_windows")
                else (train_windows, train_labels)
            )
            X_train = np.stack(aug_windows, axis=0).astype(np.float32, copy=False)
            X_val = np.stack(val_windows, axis=0).astype(np.float32, copy=False)
            clf = get_classifier(
                classifier,
                n_channels=config.n_channels,
                n_classes=len(class_labels),
            )
            fit_result = clf.fit_split(
                X_train,
                np.asarray(aug_labels, dtype=np.int32),
                X_val,
                val_labels,
            )
            y_pred = clf.predict_batch(X_val) if len(X_val) else np.asarray([], dtype=np.int32)
            augmented_train_windows = len(aug_windows)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        metrics = self._classification_metrics(val_labels, np.asarray(y_pred, dtype=np.int32), class_labels)
        return {
            **fit_result,
            **metrics,
            "success": True,
            "classifier": classifier,
            "n_train_windows": len(train_windows),
            "n_train_windows_augmented": int(augmented_train_windows),
            "n_val_windows": len(val_windows),
            "train_class_counts": self._count_indexed_labels(train_labels, class_labels),
            "val_class_counts": self._count_indexed_labels(val_labels.tolist(), class_labels),
        }

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_labels: Tuple[str, ...],
    ) -> Dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        if y_true.size == 0:
            raise ValueError("Validation split is empty after preprocessing")

        label_indices = list(range(len(class_labels)))
        return {
            "val_accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_labels": list(class_labels),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_indices).astype(int).tolist(),
        }

    def _count_indexed_labels(
        self,
        label_indices: List[int],
        class_labels: Tuple[str, ...],
    ) -> Dict[str, int]:
        if not label_indices:
            return {}
        arr = np.asarray(label_indices, dtype=np.int32)
        return {
            class_labels[int(idx)]: int((arr == idx).sum())
            for idx in np.unique(arr)
        }

    def _experiment_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.get("result") or {}
        split = data.get("split") or {}
        return {
            "experiment_id": data.get("experiment_id"),
            "dataset_id": data.get("dataset_id"),
            "dataset_name": data.get("dataset_name"),
            "created_at_utc": data.get("created_at_utc"),
            "signal_profile": data.get("signal_profile"),
            "signal_profile_name": data.get("signal_profile_name"),
            "classifier": data.get("classifier"),
            "status": data.get("status"),
            "n_windows": int(data.get("n_windows") or 0),
            "split_strategy": split.get("strategy") or data.get("split_strategy"),
            "holdout_group_id": split.get("holdout_group_id"),
            "holdout_subject_id": split.get("holdout_subject_id"),
            "val_accuracy": result.get("val_accuracy"),
            "f1_macro": result.get("f1_macro"),
            "duration_s": data.get("duration_s"),
        }
