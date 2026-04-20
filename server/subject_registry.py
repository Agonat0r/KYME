"""Persistent subject registry for research-grade session metadata."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_subject_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (value or "").strip())
    return clean.strip("._-")


class SubjectRegistry:
    def __init__(self) -> None:
        self._path = Path(config.data_dir) / "subjects.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write({"schema_version": 1, "subjects": []})

    def list_subjects(self) -> List[Dict[str, Any]]:
        data = self._read()
        subjects = data.get("subjects") or []
        subjects = [self._normalize_record(record) for record in subjects if isinstance(record, dict)]
        subjects.sort(key=lambda item: (item.get("subject_id") or "").lower())
        return subjects

    def get_subject(self, subject_id: str) -> Optional[Dict[str, Any]]:
        wanted = _normalize_subject_id(subject_id)
        if not wanted:
            return None
        for record in self.list_subjects():
            if record.get("subject_id") == wanted:
                return record
        return None

    def upsert_subject(self, payload: Dict[str, Any], mark_session: bool = False) -> Dict[str, Any]:
        subject_id = _normalize_subject_id(str(payload.get("subject_id") or ""))
        if not subject_id:
            raise ValueError("subject_id is required")

        with self._lock:
            data = self._read()
            subjects = data.setdefault("subjects", [])
            now = _utc_now_iso()
            incoming = {
                "subject_id": subject_id,
                "display_name": str(payload.get("display_name") or "").strip(),
                "cohort": str(payload.get("cohort") or "").strip(),
                "handedness": str(payload.get("handedness") or "").strip(),
                "notes": str(payload.get("notes") or "").strip(),
            }

            for idx, record in enumerate(subjects):
                existing_id = _normalize_subject_id(str((record or {}).get("subject_id") or ""))
                if existing_id != subject_id:
                    continue
                merged = self._normalize_record(record)
                for key, value in incoming.items():
                    if key == "subject_id":
                        continue
                    if value or key not in merged:
                        merged[key] = value
                merged["subject_id"] = subject_id
                merged["updated_at_utc"] = now
                if mark_session:
                    merged["last_session_at_utc"] = now
                    merged["session_count"] = int(merged.get("session_count") or 0) + 1
                subjects[idx] = merged
                self._write(data)
                return merged

            record = {
                **incoming,
                "subject_id": subject_id,
                "created_at_utc": now,
                "updated_at_utc": now,
                "last_session_at_utc": now if mark_session else "",
                "session_count": 1 if mark_session else 0,
            }
            subjects.append(record)
            self._write(data)
            return record

    def touch_subject(self, subject_id: str) -> Optional[Dict[str, Any]]:
        clean = _normalize_subject_id(subject_id)
        if not clean:
            return None
        return self.upsert_subject({"subject_id": clean}, mark_session=True)

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"schema_version": 1, "subjects": []}

    def _write(self, data: Dict[str, Any]) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subject_id": _normalize_subject_id(str(record.get("subject_id") or "")),
            "display_name": str(record.get("display_name") or "").strip(),
            "cohort": str(record.get("cohort") or "").strip(),
            "handedness": str(record.get("handedness") or "").strip(),
            "notes": str(record.get("notes") or "").strip(),
            "created_at_utc": str(record.get("created_at_utc") or ""),
            "updated_at_utc": str(record.get("updated_at_utc") or ""),
            "last_session_at_utc": str(record.get("last_session_at_utc") or ""),
            "session_count": int(record.get("session_count") or 0),
        }
