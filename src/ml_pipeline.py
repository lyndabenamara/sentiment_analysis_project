import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"


def ensure_nltk():
    """
    Ensure NLTK stopwords are available.
    """
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def text_process(mess: str) -> str:
    """
    Clean a single text string:
    1. Remove capitalized words
    2. Remove punctuation
    3. Remove stopwords
    """
    if not isinstance(mess, str):
        mess = str(mess)

    words = mess.split()
    # Remove words that start with a capital letter
    nocaps = [w for w in words if not w.istitle()]
    nocaps = " ".join(nocaps)

    # Remove punctuation
    nopunc = "".join(ch for ch in nocaps if ch not in string.punctuation)

    # Remove stopwords
    sw = set(stopwords.words("english"))
    nostop = [w for w in nopunc.split() if w.lower() not in sw]

    return " ".join(nostop)


def load_data() -> pd.DataFrame:
    csv_path = DATA_RAW / "rt_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run export_csv.py first to create it."
        )

    df = pd.read_csv(csv_path)
    expected_cols = {"PhraseId", "SentenceId", "Phrase", "Sentiment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing} in {csv_path}")

    return df


def basic_eda(df: pd.DataFrame):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Sentiment")
    plt.title("Sentiment label distribution")
    plt.tight_layout()
    out = REPORTS_DIR / "sentiment_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # Phrase length distribution
    df["Length"] = df["Phrase"].astype(str).str.split().str.len()
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="Length", bins=50, kde=True)
    plt.title("Phrase length distribution")
    plt.tight_layout()
    out = REPORTS_DIR / "phrase_length_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def build_pipelines():
    pipeline_rf = Pipeline(
        [
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    pipeline_logreg = Pipeline(
        [
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline_svm = Pipeline(
        [
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", LinearSVC()),
        ]
    )

    return {
        "RandomForest": pipeline_rf,
        "LogisticRegression": pipeline_logreg,
        "LinearSVM": pipeline_svm,
    }


def compare_models(X_train, X_test, y_train, y_test, models: dict):
    results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds))

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"model": model, "preds": preds, "accuracy": acc, "f1_macro": f1}
        print(f"{name}: accuracy={acc:.3f}, f1_macro={f1:.3f}")

    print("\nSummary:")
    for name, res in results.items():
        print(f"{name}: accuracy={res['accuracy']:.3f}, f1_macro={res['f1_macro']:.3f}")

    return results


def plot_confusion_matrix(y_test, preds, model_name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    out = REPORTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")





def main():
    ensure_nltk()
    df = load_data()
    basic_eda(df)

    X = df["Phrase"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_pipelines()
    results = compare_models(X_train, X_test, y_train, y_test, models)

    # Pick best by f1_macro
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = results[best_name]["model"]
    best_preds = results[best_name]["preds"]
    print(f"\nBest baseline model: {best_name}")
    plot_confusion_matrix(y_test, best_preds, best_name)

    


if __name__ == "__main__":
    main()