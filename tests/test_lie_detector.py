import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from lie_detector import LieDetectorSession, classify_response


def test_classifies_target_closer_to_known_false_as_lie_like():
    result = classify_response(
        score=72.0,
        baseline=[10, 12, 11, 13, 12, 10],
        truth_controls=[18, 22, 20],
        lie_controls=[68, 75, 72],
    )

    assert result["label"] == "Lie-like"
    assert result["confidence"] > 0.5
    assert result["response_delta"] > 50


def test_classifies_target_closer_to_known_truth_as_truth_like():
    result = classify_response(
        score=21.0,
        baseline=[10, 12, 11, 13, 12, 10],
        truth_controls=[18, 22, 20],
        lie_controls=[68, 75, 72],
    )

    assert result["label"] == "Truth-like"
    assert result["confidence"] > 0.5


def test_requires_baseline_and_control_samples_before_deciding():
    result = classify_response(
        score=72.0,
        baseline=[10, 12],
        truth_controls=[18],
        lie_controls=[72],
    )

    assert result["label"] == "Inconclusive"
    assert "Train baseline" in result["reason"]


def test_session_collects_samples_and_scores_current_prediction():
    session = LieDetectorSession()
    for score in [10, 12, 11, 13, 12, 10]:
        session.add_sample("baseline", {"arousal": {"score": score}})
    for score in [18, 22]:
        session.add_sample("truth", {"arousal": {"score": score}})
    for score in [68, 75]:
        session.add_sample("lie", {"arousal": {"score": score}})

    session.start_question("Did you take it?")
    payload = session.score_question({"arousal": {"score": 73}}, answer="no")

    assert payload["ready"] is True
    assert payload["result"]["label"] == "Lie-like"
    assert payload["last_result"]["question"] == "Did you take it?"
