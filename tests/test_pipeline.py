import pytest
import os
from pathlib import Path
from dotenv import load_dotenv
from secure_prompt.core.decision import DecisionCore


load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

with open(DATA_DIR / os.getenv("JAILBREAK_DATA_PATH"), "r") as f:
    JAILBREAK = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_DATA_PATH"), "r") as f:
    BENIGN = f.readlines()

hybrid = DecisionCore()
@pytest.mark.parametrize("text", JAILBREAK)
def test_jailbreak(text):
    decision = hybrid.decide([text])
    assert decision[0].verdict == "BLOCK"

@pytest.mark.parametrize("text", BENIGN)
def test_safe(text):
    decision = hybrid.decide([text])
    assert decision[0].verdict == "ALLOW"
