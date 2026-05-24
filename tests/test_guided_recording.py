import asyncio
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import guided_recording as guided_module
from guided_recording import GuidedRecordingFlow


class DummyPipeline:
    def __init__(self):
        self.current_label = None
        self.counts = {}

    def start_recording(self, label):
        self.current_label = label

    def stop_recording(self):
        self.current_label = None

    def add_window(self):
        if self.current_label:
            self.counts[self.current_label] = self.counts.get(self.current_label, 0) + 1

    def get_training_summary(self):
        return {"per_gesture": dict(self.counts), "total_windows": sum(self.counts.values())}


class DummyStream:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_window(self):
        self.pipeline.add_window()
        return np.ones((8, 50), dtype=np.float32)


def test_guided_confirm_advances_waiting_and_quality(monkeypatch):
    original_sleep = asyncio.sleep

    async def fast_sleep(_seconds):
        await original_sleep(0)

    monkeypatch.setattr(guided_module.asyncio, "sleep", fast_sleep)

    async def scenario():
        flow = GuidedRecordingFlow()
        pipeline = DummyPipeline()
        stream = DummyStream(pipeline)
        flow.create_session([
            {
                "instruction": "Relax for test",
                "gesture": "rest",
                "duration_s": 0.001,
                "type": "record",
                "reps": 1,
                "requires_confirmation": True,
            }
        ], target_windows=1)

        task = asyncio.create_task(flow.run(stream, pipeline))

        for _ in range(50):
            if flow.get_state()["phase"] == "waiting":
                break
            await original_sleep(0)
        assert flow.get_state()["phase"] == "waiting"

        flow.confirm_step()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if flow.get_state()["phase"] == "quality":
                break
            await original_sleep(0)
        assert flow.get_state()["phase"] == "quality"
        assert flow.get_state()["can_redo"] is True

        flow.confirm_step()
        result = await task

        assert result["success"] is True
        assert flow.get_state()["phase"] == "complete"
        assert flow.get_state()["active"] is False

    asyncio.run(scenario())
