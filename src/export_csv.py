# converting the kaggle tsv file to csv so it's easier to work with
import pandas as pd
from pathlib import Path

# setting up paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"

def main():
    # looking for the original kaggle data file
    tsv_path = DATA_RAW / "train.tsv"

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"could not find {tsv_path}. "
            "make sure you downloaded train.tsv from kaggle and put it in data/raw/."
        )

    # loading the tab-separated file
    trainmessages = pd.read_csv(tsv_path, sep="\t")

    print("loaded train.tsv")
    print("columns:", list(trainmessages.columns))
    print("first 5 rows:")
    print(trainmessages.head())

    # saving as csv for easier handling
    out_path = DATA_RAW / "rt_train.csv"
    trainmessages.to_csv(out_path, index=False)
    print(f"\nsaved csv to: {out_path.resolve()}")

if __name__ == "__main__":
    main()