import os
from pathlib import Path
from dotenv import load_dotenv
from secure_prompt.core.decision import DecisionCore
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

with open(DATA_DIR / os.getenv("JAILBREAK_TEST_PATH"), "r") as f:
    JAILBREAK = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_TEST_PATH"), "r") as f:
    BENIGN = f.readlines()


hybrid = DecisionCore()
y_test = [1] * len(JAILBREAK) + [0] * len(BENIGN)
result = hybrid.decide(JAILBREAK+BENIGN)
y_pred = [int(x.verdict == "BLOCK") for x in result]
y_prob = [x.probability for x in result]

print("=== Classification report ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("ACCURACY:", accuracy_score(y_test, y_pred))
