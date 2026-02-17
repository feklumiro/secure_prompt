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

ml = MLGuard(use_vector=False)
# ============== JAILBREAK TEST ==============
@pytest.mark.parametrize("text", JAILBREAK)
def test_jailbreak(text):
    res = ml.detect(preprocess([text]))
    for i in res:
        print(res)
        if i.is_jailbreak:
            assert True
            return
    assert False


# ============== BENIGN TEST ==============
@pytest.mark.parametrize("text", BENIGN)
def test_benign(text):
    res = ml.detect(preprocess([text]))
    for i in res:
        print(res)
        assert not i.is_jailbreak
    return True
