"""Firmware workspace and Arduino CLI helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


class FirmwareWorkspace:
    ALLOWED_EXTENSIONS = {
        ".ino",
        ".pde",
        ".h",
        ".hpp",
        ".c",
        ".cpp",
        ".txt",
        ".md",
        ".json",
        ".py",
    }

    def __init__(self, root_dir: Path, *, seed_paths: List[Path] | None = None):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir = self.root_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.seed_paths = [Path(path).resolve() for path in (seed_paths or []) if path]
        self._seed_workspace()

    def status(self) -> Dict[str, Any]:
        cli_path = shutil.which("arduino-cli") or ""
        return {
            "root_dir": self.root_dir.as_posix(),
            "generated_dir": self.generated_dir.as_posix(),
            "arduino_cli": {
                "available": bool(cli_path),
                "path": cli_path,
                "note": (
                    "Compile/upload works for any Arduino-compatible sketch when arduino-cli and the board core are installed."
                    if cli_path
                    else "arduino-cli was not found on PATH. You can still save and edit firmware files."
                ),
            },
        }

    def list_files(self) -> Dict[str, Any]:
        files: List[Dict[str, Any]] = []
        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                continue
            rel = path.relative_to(self.root_dir).as_posix()
            files.append(
                {
                    "path": rel,
                    "name": path.name,
                    "dir": path.parent.relative_to(self.root_dir).as_posix() if path.parent != self.root_dir else "",
                    "ext": path.suffix.lower(),
                    "size": path.stat().st_size,
                    "kind": self._kind_for_path(path),
                    "generated": self.generated_dir in path.parents,
                }
            )
        return {"files": files}

    def read_file(self, rel_path: str) -> Dict[str, Any]:
        path = self._resolve_path(rel_path, must_exist=True)
        return {
            "path": path.relative_to(self.root_dir).as_posix(),
            "name": path.name,
            "kind": self._kind_for_path(path),
            "content": path.read_text(encoding="utf-8"),
        }

    def save_file(self, rel_path: str, content: str) -> Dict[str, Any]:
        path = self._resolve_path(rel_path, must_exist=False)
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported firmware file type: {path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content).replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        return {
            "ok": True,
            "saved": {
                "path": path.relative_to(self.root_dir).as_posix(),
                "filename": path.name,
                "size": path.stat().st_size,
            },
        }

    def save_generated(self, *, name: str, content: str, filename: str = "", target: str = "arduino") -> Dict[str, Any]:
        raw_name = filename or name or "generated_sketch.ino"
        safe_filename = self._safe_filename(raw_name)
        suffix = Path(safe_filename).suffix.lower() or ".ino"
        stem = self._safe_stem(Path(safe_filename).stem or "generated_sketch")
        safe_filename = f"{stem}{suffix}"
        folder = self.generated_dir / stem
        rel_path = (folder / safe_filename).relative_to(self.root_dir).as_posix()
        saved = self.save_file(rel_path, content)
        saved["saved"]["target"] = str(target or "arduino")
        return saved

    def compile_sketch(self, *, rel_path: str, fqbn: str) -> Dict[str, Any]:
        clean_fqbn = str(fqbn or "").strip()
        if not clean_fqbn:
            raise ValueError("FQBN is required to compile a sketch")
        sketch_dir = self._sketch_dir_for(rel_path)
        proc = self._run_arduino_cli(["compile", "--fqbn", clean_fqbn, str(sketch_dir)])
        return {
            "ok": proc.returncode == 0,
            "command": proc.args,
            "path": sketch_dir.relative_to(self.root_dir).as_posix(),
            "fqbn": clean_fqbn,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }

    def upload_sketch(self, *, rel_path: str, fqbn: str, port: str) -> Dict[str, Any]:
        clean_fqbn = str(fqbn or "").strip()
        clean_port = str(port or "").strip()
        if not clean_fqbn:
            raise ValueError("FQBN is required to upload a sketch")
        if not clean_port:
            raise ValueError("Port is required to upload a sketch")
        sketch_dir = self._sketch_dir_for(rel_path)
        proc = self._run_arduino_cli(["upload", "-p", clean_port, "--fqbn", clean_fqbn, str(sketch_dir)])
        return {
            "ok": proc.returncode == 0,
            "command": proc.args,
            "path": sketch_dir.relative_to(self.root_dir).as_posix(),
            "fqbn": clean_fqbn,
            "port": clean_port,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }

    def _run_arduino_cli(self, args: List[str]) -> subprocess.CompletedProcess[str]:
        cli_path = shutil.which("arduino-cli")
        if not cli_path:
            raise RuntimeError("arduino-cli was not found on PATH")
        return subprocess.run(
            [cli_path, *args],
            cwd=self.root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )

    def _sketch_dir_for(self, rel_path: str) -> Path:
        path = self._resolve_path(rel_path, must_exist=True)
        if path.suffix.lower() not in {".ino", ".pde"}:
            raise ValueError("Arduino CLI actions require an .ino or .pde sketch file")
        return path.parent

    def _resolve_path(self, rel_path: str, *, must_exist: bool) -> Path:
        raw = str(rel_path or "").strip().replace("\\", "/")
        if not raw:
            raise ValueError("Firmware path is required")
        path = (self.root_dir / raw).resolve()
        if self.root_dir != path and self.root_dir not in path.parents:
            raise ValueError("Firmware path is outside the workspace")
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Firmware file not found: {raw}")
        return path

    def _kind_for_path(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".ino", ".pde"}:
            return "sketch"
        if ext in {".h", ".hpp", ".c", ".cpp"}:
            return "source"
        return "text"

    def _safe_filename(self, value: str) -> str:
        name = str(value or "").strip().replace("\\", "/").split("/")[-1]
        allowed = [ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name]
        cleaned = "".join(allowed).strip("._")
        return cleaned[:96] or "generated_sketch.ino"

    def _safe_stem(self, value: str) -> str:
        allowed = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip()]
        cleaned = "".join(allowed).strip("._-")
        return cleaned[:64] or "generated_sketch"

    def _seed_workspace(self) -> None:
        for source_root in self.seed_paths:
            if not source_root.exists() or not source_root.is_dir():
                continue
            for path in source_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                    continue
                rel = path.relative_to(source_root)
                dest = self.root_dir / rel
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
