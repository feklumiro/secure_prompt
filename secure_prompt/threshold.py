from secure_prompt.ML.ml_guard import MLGuard
from secure_prompt.core.preprocess import preprocess
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# LOAD DATA
with open(DATA_DIR / os.getenv("JAILBREAK_TRAIN_PATH"), "r") as f:
    data = f.readlines()
with open(DATA_DIR / os.getenv("BENIGN_TRAIN_PATH"), "r") as f:
    bg = f.readlines()

model = MLGuard(use_vector=False)
X = [i.probability for i in model.detect(preprocess(data))]
XB = [i.probability for i in model.detect(preprocess(bg))]

# CALCULATE threshold
d = 0.0005
opt = -10
g = 0
p = -10.0
while p <= 10:
    t = 0
    for i in X:
        if i >= p:
            t += 1
    for i in XB:
        if i < p:
            t += 1
    if t >= g:
        opt = p
        g = t
    p += d
print("NON-VECTOR threshold:", opt, "|", g)

model = MLGuard(use_vector=True)
X = [i.probability for i in model.detect(preprocess(data))]
XB = [i.probability for i in model.detect(preprocess(bg))]
# CALCULATE OPT
d = 0.0005
opt = -10
g = 0
p = -10.0
while p <= 10:
    t = 0
    for i in X:
        if i >= p:
            t += 1
    for i in XB:
        if i < p:
            t += 1
    if t >= g:
        opt = p
        g = t
    p += d
print("VECTOR threshold:", opt, "|", g)

# 0.98
