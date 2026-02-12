import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv
from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.ml_guard import MLGuard


load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# чтобы тест можно было запускать напрямую
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with open(DATA_DIR / os.getenv("JAILBREAK_DATA_PATH"), "r") as f:
    JAILBREAK = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_DATA_PATH"), "r") as f:
    BENIGN = f.readlines()


# ============== JAILBREAK TEST ==============
@pytest.mark.parametrize("text", JAILBREAK)
def test_jailbreak(text):
    norm = preprocess(text)
    for i in norm:
        ml = MLGuard()
        ml_result = ml.detect(norm[i])
        print("JLB", norm[i], ml_result)
        if ml_result.is_jailbreak:
            assert True
            return
    assert False


# ============== BENIGN TEST ==============
@pytest.mark.parametrize("text", BENIGN)
def test_benign(text):
    norm = preprocess(text)
    for i in norm:
        ml = MLGuard()
        ml_result = ml.detect(norm[i])
        print("BNG", norm[i], ml_result)
        if ml_result.is_jailbreak:
            assert False
    assert True
