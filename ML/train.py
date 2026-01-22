import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from ML.dataset import DatasetLoader


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def train():
    loader = DatasetLoader()
    X, y = loader.load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")

    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            print("!!!!", X_test[i], y_prob[i])


if __name__ == "__main__":
    train()
