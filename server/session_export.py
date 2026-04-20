"""Session export helpers for BIDS-oriented research bundles."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from biosignal_profiles import get_profile
from config import config
from session_recorder import SessionRecorder

logger = logging.getLogger(__name__)

try:
    import mne
    from bids import BIDSLayout
    from mne_bids import BIDSPath, write_raw_bids

    _MNE_BIDS_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    mne = None  # type: ignore[assignment]
    BIDSLayout = None  # type: ignore[assignment]
    BIDSPath = None  # type: ignore[assignment]
    write_raw_bids = None  # type: ignore[assignment]
    _MNE_BIDS_ERROR = exc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bids_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value or "").strip())
    return token or fallback


def _channel_name(value: str, fallback: str) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", text)
    return text or fallback


class SessionExporter:
    """Export recorded sessions into BIDS-friendly bundles."""

    def __init__(self, recorder: SessionRecorder):
        self.recorder = recorder
        self.export_root = Path(config.data_dir) / "exports"

    def export_bids(
        self,
        session_id: str,
        subject_id: str = "01",
        task_name: str = "kyma",
    ) -> Dict[str, Any]:
        meta = self.recorder.get_session_meta(session_id)
        if not meta:
            raise FileNotFoundError(f"Session not found: {session_id}")

        summary = self.recorder.get_session_summary(session_id)
        if not summary:
            raise FileNotFoundError(f"Session summary not available: {session_id}")

        signal_path_str = self.recorder.get_signal_path(session_id)
        if not signal_path_str:
            raise ValueError("Session does not have a signal recording to export")

        signal_path = Path(signal_path_str)
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file missing: {signal_path}")

        session_path_str = self.recorder.get_session_path(session_id)
        if not session_path_str:
            raise FileNotFoundError(f"Session directory missing: {session_id}")
        session_path = Path(session_path_str)

        cfg = meta.get("config", {})
        profile_key = str(summary.get("signal_profile") or cfg.get("signal_profile") or "biosignal")
        try:
            profile = get_profile(profile_key)
        except Exception:
            profile = get_profile(config.signal_profile_name)
            profile_key = profile.key

        sample_rate = int(summary.get("sample_rate") or cfg.get("sample_rate") or config.sample_rate)
        units = str(summary.get("units") or cfg.get("units") or profile.units)
        raw_channel_labels = list(cfg.get("channel_labels") or config.channel_labels)

        subject_token = _bids_token(subject_id, "01")
        session_token = _bids_token(session_id, "session1")
        task_token = _bids_token(task_name, "kyma")
        recording_token = _bids_token(profile_key, "biosignal")

        dataset_root = self.export_root / "bids"
        self._write_dataset_root(dataset_root)
        self._write_participants(dataset_root, subject_token)
        self._write_sessions_tsv(dataset_root, subject_token, session_token, meta)

        timestamps, samples = self._load_signal_matrix(signal_path)
        if samples.size == 0:
            raise ValueError("Session signal file is empty")

        if len(raw_channel_labels) < samples.shape[1]:
            raw_channel_labels = raw_channel_labels + [
                f"CH{i + 1}" for i in range(len(raw_channel_labels), samples.shape[1])
            ]
        channel_labels = [
            _channel_name(raw_channel_labels[i], f"CH{i + 1}") for i in range(samples.shape[1])
        ]
        events = self._load_events(session_path)

        if profile_key == "eeg" and mne is not None and BIDSPath is not None and write_raw_bids is not None:
            export_info = self._export_eeg_raw(
                dataset_root=dataset_root,
                subject_token=subject_token,
                session_token=session_token,
                task_token=task_token,
                sample_rate=sample_rate,
                channel_labels=channel_labels,
                samples=samples,
                events=events,
                session_id=session_id,
                profile_key=profile_key,
            )
        else:
            export_info = self._export_physio_bundle(
                dataset_root=dataset_root,
                subject_token=subject_token,
                session_token=session_token,
                task_token=task_token,
                recording_token=recording_token,
                sample_rate=sample_rate,
                channel_labels=channel_labels,
                timestamps=timestamps,
                samples=samples,
                profile=profile,
                profile_key=profile_key,
                units=units,
                meta=meta,
                summary=summary,
                events=events,
                session_id=session_id,
            )

        export_info.update(self._validate_dataset(dataset_root))
        export_info["exported_at_utc"] = _utc_now_iso()
        self.recorder.register_export(session_id, "bids", export_info)
        logger.info("BIDS export ready for %s -> %s", session_id, export_info.get("session_dir"))
        return export_info

    def _load_signal_matrix(self, signal_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.loadtxt(signal_path, delimiter=",", skiprows=1)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] < 2:
            raise ValueError("Session signal file does not contain channel samples")
        timestamps = rows[:, 0].astype(np.float64)
        samples = rows[:, 1:].astype(np.float64)
        return timestamps, samples

    def _load_events(self, session_path: Path) -> List[Dict[str, Any]]:
        events_path = session_path / "events.json"
        if not events_path.exists():
            return []
        with open(events_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            return []
        return [event for event in payload if isinstance(event, dict)]

    def _export_eeg_raw(
        self,
        *,
        dataset_root: Path,
        subject_token: str,
        session_token: str,
        task_token: str,
        sample_rate: int,
        channel_labels: List[str],
        samples: np.ndarray,
        events: List[Dict[str, Any]],
        session_id: str,
        profile_key: str,
    ) -> Dict[str, Any]:
        info = mne.create_info(channel_labels, sfreq=sample_rate, ch_types=["eeg"] * len(channel_labels), verbose="ERROR")
        raw = mne.io.RawArray(samples.T, info, verbose="ERROR")
        raw.set_meas_date(None)
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
        except Exception:
            pass

        if events:
            annotations = mne.Annotations(
                onset=[float(event.get("timestamp_s") or 0.0) for event in events],
                duration=[0.0 for _ in events],
                description=[str(event.get("type") or "event") for event in events],
            )
            raw.set_annotations(annotations)

        bids_path = BIDSPath(
            root=dataset_root,
            subject=subject_token,
            session=session_token,
            task=task_token,
            datatype="eeg",
        )
        write_raw_bids(
            raw,
            bids_path,
            allow_preload=True,
            format="BrainVision",
            overwrite=True,
            verbose="ERROR",
        )

        session_dir = dataset_root / f"sub-{subject_token}" / f"ses-{session_token}" / "eeg"
        return {
            "format": "mne_bids_raw",
            "dataset_root": str(dataset_root),
            "session_dir": str(session_dir),
            "subject": f"sub-{subject_token}",
            "session": f"ses-{session_token}",
            "task": task_token,
            "profile": profile_key,
            "session_id": session_id,
            "n_samples": int(samples.shape[0]),
            "n_channels": int(samples.shape[1]),
            "signal_file": str(next(session_dir.glob("*_eeg.vhdr"), session_dir)),
        }

    def _export_physio_bundle(
        self,
        *,
        dataset_root: Path,
        subject_token: str,
        session_token: str,
        task_token: str,
        recording_token: str,
        sample_rate: int,
        channel_labels: List[str],
        timestamps: np.ndarray,
        samples: np.ndarray,
        profile,
        profile_key: str,
        units: str,
        meta: Dict[str, Any],
        summary: Dict[str, Any],
        events: List[Dict[str, Any]],
        session_id: str,
    ) -> Dict[str, Any]:
        datatype_dir = dataset_root / f"sub-{subject_token}" / f"ses-{session_token}" / "beh"
        datatype_dir.mkdir(parents=True, exist_ok=True)

        stem = f"sub-{subject_token}_ses-{session_token}_task-{task_token}_recording-{recording_token}"
        signal_out = datatype_dir / f"{stem}_physio.tsv.gz"
        sidecar_out = datatype_dir / f"{stem}_physio.json"
        events_out = datatype_dir / f"{stem}_events.tsv"

        with gzip.open(signal_out, "wt", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["time"] + channel_labels)
            for idx in range(samples.shape[0]):
                writer.writerow(
                    [f"{float(timestamps[idx]):.5f}"] + [f"{float(value):.6f}" for value in samples[idx]]
                )

        sidecar = {
            "TaskName": "kyma",
            "SamplingFrequency": sample_rate,
            "StartTime": 0.0,
            "Columns": ["time", *channel_labels],
            "Manufacturer": "KYMA",
            "SignalProfile": profile_key,
            "SignalProfileName": profile.display_name,
            "SignalFamily": profile.family,
            "Units": units,
            "StreamSource": summary.get("stream_source") or "unknown",
            "SourceDetails": meta.get("source_details") or {},
            "RecordingDuration": round(float(summary.get("duration_s") or 0.0), 3),
            "SampleCount": int(samples.shape[0]),
            "ChannelCount": int(samples.shape[1]),
            "Filters": [stage.to_dict() for stage in profile.filters],
        }
        with open(sidecar_out, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

        with open(events_out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["onset", "duration", "trial_type", "value"],
                delimiter="\t",
            )
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {
                        "onset": round(float(event.get("timestamp_s") or 0.0), 4),
                        "duration": 0.0,
                        "trial_type": str(event.get("type") or "event"),
                        "value": json.dumps(event.get("data") or {}, separators=(",", ":")),
                    }
                )

        return {
            "format": "bids_physio",
            "dataset_root": str(dataset_root),
            "session_dir": str(datatype_dir),
            "subject": f"sub-{subject_token}",
            "session": f"ses-{session_token}",
            "task": task_token,
            "profile": profile_key,
            "session_id": session_id,
            "n_samples": int(samples.shape[0]),
            "n_channels": int(samples.shape[1]),
            "signal_file": str(signal_out),
            "events_file": str(events_out),
        }

    def _validate_dataset(self, dataset_root: Path) -> Dict[str, Any]:
        if BIDSLayout is None:
            return {
                "validated": False,
                "validation_backend": "unavailable",
                "validation_error": f"{type(_MNE_BIDS_ERROR).__name__}: {_MNE_BIDS_ERROR}",
            }

        try:
            layout = BIDSLayout(str(dataset_root), validate=True, reset_database=True)
            indexed = layout.get(return_type="filename")
            return {
                "validated": True,
                "validation_backend": "pybids",
                "indexed_files": len(indexed),
            }
        except Exception as exc:
            logger.warning("BIDS validation failed: %s", exc)
            return {
                "validated": False,
                "validation_backend": "pybids",
                "validation_error": f"{type(exc).__name__}: {exc}",
            }

    def _write_dataset_root(self, dataset_root: Path) -> None:
        dataset_root.mkdir(parents=True, exist_ok=True)

        desc_path = dataset_root / "dataset_description.json"
        desc = {
            "Name": "KYMA Biosignal Session Exports",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "GeneratedBy": [
                {
                    "Name": "KYMA",
                    "Version": "1.0.0",
                    "Description": "BIDS-oriented export bundles generated from recorded KYMA sessions.",
                }
            ],
        }
        with open(desc_path, "w", encoding="utf-8") as fh:
            json.dump(desc, fh, indent=2)

        readme_path = dataset_root / "README"
        readme = (
            "KYMA biosignal exports\n"
            "=====================\n\n"
            "This dataset contains BIDS-oriented exports generated from recorded KYMA sessions.\n"
            "EEG sessions are exported through mne-bids. Other biosignal sessions are exported\n"
            "as BIDS-physio bundles so they remain compatible with broader research tooling.\n"
        )
        with open(readme_path, "w", encoding="utf-8") as fh:
            fh.write(readme)

    def _write_participants(self, dataset_root: Path, subject_token: str) -> None:
        path = dataset_root / "participants.tsv"
        rows: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with open(path, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    participant_id = str(row.get("participant_id") or "").strip()
                    if participant_id:
                        rows[participant_id] = row

        participant_id = f"sub-{subject_token}"
        rows[participant_id] = {
            "participant_id": participant_id,
            "species": "homo sapiens",
            "group": "pilot",
        }

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["participant_id", "species", "group"],
                delimiter="\t",
            )
            writer.writeheader()
            for pid in sorted(rows):
                writer.writerow(rows[pid])

    def _write_sessions_tsv(
        self,
        dataset_root: Path,
        subject_token: str,
        session_token: str,
        meta: Dict[str, Any],
    ) -> None:
        subject_dir = dataset_root / f"sub-{subject_token}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        path = subject_dir / f"sub-{subject_token}_sessions.tsv"
        rows: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with open(path, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sid = str(row.get("session_id") or "").strip()
                    if sid:
                        rows[sid] = row

        rows[f"ses-{session_token}"] = {
            "session_id": f"ses-{session_token}",
            "acq_time": str(meta.get("created_at_utc") or meta.get("started_at_utc") or ""),
            "source": str(meta.get("stream_source") or "unknown"),
        }

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["session_id", "acq_time", "source"],
                delimiter="\t",
            )
            writer.writeheader()
            for sid in sorted(rows):
                writer.writerow(rows[sid])
