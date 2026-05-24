import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from biosignal_profiles import get_profile


def test_eda_profile_allows_live_cyton_hardware_streaming():
    profile = get_profile("eda")

    assert profile.hardware_supported is True
    assert "Cyton" in profile.support_notes
