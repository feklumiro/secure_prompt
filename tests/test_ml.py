import sys
import pytest
from pathlib import Path
from tests.test_regex import JAILBREAK_CASES, SAFE_CASES
from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.ml_guard import MLGuard


# чтобы тест можно было запускать напрямую
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("text", JAILBREAK_CASES)
def test_jailbreak(text):
    norm = preprocess(text)
    for i in norm:
        ml = MLGuard()
        ml_result = ml.predict(i)
        print(ml_result)
        if ml_result.score >= 1:
            assert True
            return
    assert False


@pytest.mark.parametrize("text", SAFE_CASES)
def test_safe(text):
    norm = preprocess(text)
    for i in norm:
        ml = MLGuard()
        ml_result = ml.predict(i)
        if ml_result.score >= 1:
            assert False
    assert True


