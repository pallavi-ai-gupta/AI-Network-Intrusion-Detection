import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "dataset", "network_traffic.csv")
MODEL_PATH = os.path.join(BASE, "models", "intrusion_model.pkl")
REPORT_PATH = os.path.join(BASE, "reports", "model_report.json")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def train_model():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(how="all")

    if "label" not in df.columns:
        raise ValueError(
            "Dataset needs a 'label' column containing known classes."
        )

    df = df.dropna(subset=["label"])
    y = df["label"].astype(str).str.strip()
    X = df.drop(columns=["label"])

    valid = y.ne("") & y.ne("nan")
    X, y = X.loc[valid].copy(), y.loc[valid]

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if len(X) < 10 or y.nunique() < 2 or X.shape[1] == 0:
        raise ValueError(
            "Provide at least 10 usable rows, two target classes, "
            "and one nonconstant feature."
        )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    metrics = {}
    best_pipeline, best_name, best_f1 = None, None, -1

    algorithms = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        )
    }

    for name, classifier in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer(transformers)),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        precision = precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }

        if f1 > best_f1:
            best_pipeline, best_name, best_f1 = pipeline, name, f1

    predictions = best_pipeline.predict(X_test)
    labels = sorted(set(y_test.astype(str)) | set(predictions.astype(str)))

    report = {
        "best_model": best_name,
        "metrics": metrics,
        "rows": int(len(df)),
        "features": int(len(X.columns)),
        "target": "label",
        "classes": labels,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist()
    }

    joblib.dump({
        "model": best_pipeline,
        "features": list(X.columns),
        "classes": list(best_pipeline.classes_),
        "model_name": best_name
    }, MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Training completed.")
    print("Best model:", best_name)
    print(json.dumps(metrics, indent=2))
    return report


if __name__ == "__main__":
    train_model()