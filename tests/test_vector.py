import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from secure_prompt.guards.vector_guard import VectorGuard


load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

with open(DATA_DIR / os.getenv("JAILBREAK_DATA_PATH"), "r") as f:
    JAILBREAK = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_DATA_PATH"), "r") as f:
    BENIGN = f.readlines()

detector = VectorGuard(threshold=3.075)
@pytest.mark.parametrize("text", JAILBREAK)
def test_jailbreak(text):
    decision = detector.detect(text)
    assert decision.is_jailbreak

@pytest.mark.parametrize("text", BENIGN)
def test_safe(text):
    decision = detector.detect(text)
    assert not decision.is_jailbreak
