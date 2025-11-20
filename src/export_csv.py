import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"

def main():
    tsv_path = DATA_RAW / "train.tsv"

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Could not find {tsv_path}. "
            "Make sure you downloaded train.tsv from Kaggle and put it in data/raw/."
        )

    # Load TSV (tab-separated)
    trainmessages = pd.read_csv(tsv_path, sep="\t")

    print("Loaded train.tsv")
    print("Columns:", list(trainmessages.columns))
    print("First 5 rows:")
    print(trainmessages.head())

    # Export to CSV
    out_path = DATA_RAW / "rt_train.csv"
    trainmessages.to_csv(out_path, index=False)
    print(f"\nSaved CSV to: {out_path.resolve()}")

if __name__ == "__main__":
    main()