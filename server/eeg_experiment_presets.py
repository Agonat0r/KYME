"""EEG research presets for record/export workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class EEGExperimentPreset:
    key: str
    title: str
    summary: str
    mode: str = "record_only"
    recommended_source: str = "hardware"
    session_condition: str = ""
    recommended_export: Tuple[str, ...] = ("xdf", "bids")
    marker_strategy: str = "external_lsl_markers"
    marker_names: Tuple[str, ...] = field(default_factory=tuple)
    block_structure: Tuple[str, ...] = field(default_factory=tuple)
    instructions: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "mode": self.mode,
            "recommended_source": self.recommended_source,
            "session_condition": self.session_condition or self.key,
            "recommended_export": list(self.recommended_export),
            "marker_strategy": self.marker_strategy,
            "marker_names": list(self.marker_names),
            "block_structure": list(self.block_structure),
            "instructions": list(self.instructions),
        }


_PRESETS: Tuple[EEGExperimentPreset, ...] = (
    EEGExperimentPreset(
        key="visual_p300_oddball",
        title="Visual P300 Oddball",
        summary="Target vs non-target ERP recording blocks for oddball-style attention experiments.",
        session_condition="p300_oddball",
        marker_names=(
            "p300/block_start",
            "p300/target",
            "p300/non_target",
            "p300/response",
            "p300/block_end",
        ),
        block_structure=(
            "10-20 s baseline eyes-open segment before the first stimulus block.",
            "Oddball block with frequent non-targets and sparse targets, markers on every stimulus onset.",
            "Optional response marker if the participant counts or presses on targets.",
            "15-30 s rest, then repeat 4-8 blocks with the same montage and subject metadata.",
        ),
        instructions=(
            "Run the visual stimulus presenter outside KYMA and send markers over LSL.",
            "Use KYMA to record the synchronized EEG stream, subject metadata, and block notes.",
            "Export the session to XDF or BIDS after the run for offline averaging and ERP review.",
        ),
    ),
    EEGExperimentPreset(
        key="visual_ssvep",
        title="Visual SSVEP",
        summary="Frequency-tagged gaze or focus blocks for steady-state response experiments.",
        session_condition="ssvep",
        marker_names=(
            "ssvep/block_start",
            "ssvep/freq_10hz",
            "ssvep/freq_12hz",
            "ssvep/freq_15hz",
            "ssvep/block_end",
        ),
        block_structure=(
            "5-10 s baseline before frequency-tagged trials begin.",
            "One labeled block per flicker condition, with a marker at each condition onset.",
            "Maintain fixed stimulus timing and trial duration across all frequency conditions.",
            "Insert short rest periods between blocks and keep condition order logged in notes or markers.",
        ),
        instructions=(
            "Keep the flicker or display logic in an external stimulus tool.",
            "Record each frequency condition as its own KYMA block and preserve markers.",
            "Use XDF or BIDS export for later spectral analysis instead of the live band decoder.",
        ),
    ),
    EEGExperimentPreset(
        key="visual_n170",
        title="Visual N170",
        summary="Face/object visual ERP blocks for early visual-response studies.",
        session_condition="n170",
        marker_names=(
            "n170/block_start",
            "n170/face",
            "n170/object",
            "n170/response",
            "n170/block_end",
        ),
        block_structure=(
            "10-20 s baseline before stimulus presentation.",
            "Present face and non-face/object trials with markers at every image onset.",
            "Keep image timing, inter-stimulus interval, and response window fixed across the run.",
            "Repeat balanced blocks so each subject has multiple full runs for session/subject holdout.",
        ),
        instructions=(
            "Use an external presenter for image timing and event markers.",
            "Record each condition as a separate KYMA session block with the same subject and run id.",
            "Review the exported data offline; this preset does not change KYMA's live EEG class labels.",
        ),
    ),
)


def list_eeg_experiment_presets() -> List[Dict[str, object]]:
    return [preset.to_dict() for preset in _PRESETS]
