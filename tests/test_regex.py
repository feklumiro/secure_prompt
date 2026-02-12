import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from secure_prompt.guards.regex_guard import RegexGuard
from secure_prompt.core.preprocess import preprocess

load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

with open(DATA_DIR / os.getenv("JAILBREAK_DATA_PATH"), "r") as f:
    JAILBREAK = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_DATA_PATH"), "r") as f:
    BENIGN = f.readlines()


# ============== JAILBREAK TEST ==============
@pytest.mark.parametrize("text", JAILBREAK)
def test_jailbreak(text):
    norm = preprocess(text)
    r_guard = RegexGuard()
    for i in norm:
        r = r_guard.detect(norm[i])
        print(norm[i], r)
        if r.is_jailbreak:
            assert True
            return
    assert False


# ============== BENIGN TEST ==============
@pytest.mark.parametrize("text", BENIGN)
def test_safe(text):
    norm = preprocess(text)
    r_guard = RegexGuard()
    for i in norm:
        if r_guard.detect(norm[i]).is_jailbreak: assert False
    assert True

