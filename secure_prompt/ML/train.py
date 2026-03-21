import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from secure_prompt.features.feature_extractor import DatasetLoader


MODEL_PATH_VECTOR = Path(__file__).resolve().parent / "models" / "model_vector.pkl"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"


def train():
    loader = DatasetLoader()
    X, y, X_v, y_v = loader.load_dataset()

    if loader.vector_import:
        # VECTOR MODEL
        X_train, X_test, y_train, y_test = train_test_split(
            X_v, y_v,
            test_size=0.25,
            random_state=42,
            stratify=y_v
        )

        model = RandomForestClassifier(n_estimators=250, bootstrap=True, max_depth=15, min_samples_split=2, min_samples_leaf=5, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("=== Classification report ===")
        print(classification_report(y_test, y_pred))

        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
        print("ACCURACY:", accuracy_score(y_test, y_pred))

        with open(MODEL_PATH_VECTOR, "wb") as f:
            pickle.dump(model, f)

        print(f"Model with vector saved to {MODEL_PATH_VECTOR}")

    else:
        print("Exception while connecting to HuggingFace Hub, check your internet connection")

    # NON-VECTOR MODEL
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    model_n = LogisticRegression(class_weight="balanced", max_iter=1000)

    model_n.fit(X_train, y_train)

    y_pred = model_n.predict(X_test)
    y_prob = model_n.predict_proba(X_test)[:, 1]

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("ACCURACY:", accuracy_score(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_n, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
