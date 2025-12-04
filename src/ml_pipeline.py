# importing all the stuff i need for the machine learning pipeline
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd #for processing the data bases
import seaborn as sns

# sklearn models and metrics
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

# for text processing
import nltk
from nltk.corpus import stopwords

# setting up file paths so i can find my data and save results
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"


def ensure_nltk():
    """
    just checking if i already have the stopwords downloaded, if not i'll get them
    """
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def text_process(mess: str) -> str:
    """
    clean up text by removing capitalized words, punctuation, and stopwords
    this helps the model focus on the actual content words
    """
    if not isinstance(mess, str):
        mess = str(mess)

    words = mess.split()
    # getting rid of words that start with capital letters (often proper nouns)
    nocaps = [w for w in words if not w.istitle()]
    nocaps = " ".join(nocaps)

    # removing all punctuation marks
    nopunc = "".join(ch for ch in nocaps if ch not in string.punctuation)

    # removing common words like 'the', 'and', 'is' that don't add much meaning
    sw = set(stopwords.words("english"))
    nostop = [w for w in nopunc.split() if w.lower() not in sw]

    return " ".join(nostop)


def load_data() -> pd.DataFrame:
    # looking for the csv file with the processed movie review data
    csv_path = DATA_RAW / "rt_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. run export_csv.py first to create it."
        )

    # loading the data and making sure it has all the columns we expect
    df = pd.read_csv(csv_path)
    expected_cols = {"PhraseId", "SentenceId", "Phrase", "Sentiment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {missing} in {csv_path}")

    return df


def basic_eda(df: pd.DataFrame):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # seeing how many reviews we have for each sentiment score (0-4)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Sentiment")
    plt.title("sentiment label distribution")
    plt.tight_layout()
    out = REPORTS_DIR / "sentiment_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"saved: {out}")

    # checking how long the phrases are (in number of words)
    df["Length"] = df["Phrase"].astype(str).str.split().str.len()
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="Length", bins=50, kde=True)
    plt.title("phrase length distribution")
    plt.tight_layout()
    out = REPORTS_DIR / "phrase_length_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"saved: {out}")


def build_pipelines():
    # setting up three different ml pipelines to compare
    # each one does: text -> word counts -> tfidf weights -> classifier
    
    # random forest: good for complex patterns, less prone to overfitting
    pipeline_rf = Pipeline(
        [
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # logistic regression: simple, fast, and often works really well
    pipeline_logreg = Pipeline(
        [
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # support vector machine: good at finding decision boundaries
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
    # training each model and collecting results to see which one works best
    results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        # training the model on training data
        model.fit(X_train, y_train)
        # making predictions on test data
        preds = model.predict(X_test)
        # showing detailed results (precision, recall, f1 for each class)
        print(classification_report(y_test, preds))

        # calculating overall performance metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"model": model, "preds": preds, "accuracy": acc, "f1_macro": f1}
        print(f"{name}: accuracy={acc:.3f}, f1_macro={f1:.3f}")

    # quick summary of all models to see which performed best
    print("\nsummary:")
    for name, res in results.items():
        print(f"{name}: accuracy={res['accuracy']:.3f}, f1_macro={res['f1_macro']:.3f}")

    return results


def plot_confusion_matrix(y_test, preds, model_name: str):
    # creating a confusion matrix to see where the model gets confused
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"confusion matrix - {model_name}")
    plt.tight_layout()
    out = REPORTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out)
    plt.close()
    print(f"saved: {out}")





def main():
    # running the whole pipeline step by step
    ensure_nltk()
    df = load_data()
    basic_eda(df)

    X = df["Phrase"]
    y = df["Sentiment"]

    # splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # building and comparing all the models
    models = build_pipelines()
    results = compare_models(X_train, X_test, y_train, y_test, models)

    # finding the best performing model based on f1 score
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = results[best_name]["model"]
    best_preds = results[best_name]["preds"]
    print(f"\nbest baseline model: {best_name}")
    plot_confusion_matrix(y_test, best_preds, best_name)

    


if __name__ == "__main__":
    main()