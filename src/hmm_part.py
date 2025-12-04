# Hidden Markov Model for sentiment analysis on movie reviews
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM

# Setting up paths to find data and save results
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"


def load_data() -> pd.DataFrame:
    # Looking for the CSV file with the processed movie review data
    csv_path = DATA_RAW / "rt_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run export_csv.py first to create it."
        )

    # Loading and checking we have all the columns we need
    df = pd.read_csv(csv_path)
    expected_cols = {"PhraseId", "SentenceId", "Phrase", "Sentiment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing} in {csv_path}")

    return df


def prepare_sequences(df: pd.DataFrame):
    # Making sure phrases in each sentence are in the right order
    df_sorted = df.sort_values(["SentenceId", "PhraseId"]).reset_index(drop=True)

    # Observations are the sentiment labels (0=very negative, 4=very positive)
    obs = df_sorted["Sentiment"].astype(int).to_numpy()
    obs_seq = obs.reshape(-1, 1)  # hmmlearn wants 2D array

    # Keeping track of how many phrases belong to each sentence
    lengths = df_sorted.groupby("SentenceId").size().tolist()

    return df_sorted, obs_seq, lengths


def fit_hmm_k(n_states: int, obs_seq: np.ndarray, lengths):
    # training an hmm with k hidden states
    model = CategoricalHMM(
        n_components=n_states,
        n_iter=100,  # how many training iterations
        random_state=42,  # for reproducible results
    )
    model.fit(obs_seq, lengths)  # actually train the model
    log_likelihood = model.score(obs_seq, lengths)  # how well it fits the data
    return model, log_likelihood


def compare_k_values(obs_seq, lengths, k_values=(2, 3, 4, 5)):
    # trying different numbers of hidden states to see which works best
    results = []
    for k in k_values:
        model_k, ll = fit_hmm_k(k, obs_seq, lengths)
        results.append({"n_states": k, "log_likelihood": ll})
        print(f"k={k}, log-likelihood={ll:.2f}")
    return pd.DataFrame(results)


def plot_matrix(mat: np.ndarray, title: str, xlabel: str, ylabel: str, filename: str):
    # making a nice heatmap to visualize transition or emission matrices
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = REPORTS_DIR / filename
    plt.savefig(out)
    plt.close()
    print(f"saved: {out}")


def main():
    # running the complete hmm analysis
    df = load_data()
    df_sorted, obs_seq, lengths = prepare_sequences(df)

    print(f"total observations: {obs_seq.shape[0]}, "
          f"number of sentences: {len(lengths)}")

    print("\ncomparing different numbers of hidden states...")
    results_df = compare_k_values(obs_seq, lengths, k_values=(2, 3, 4, 5))
    print("\nk vs log-likelihood:")
    print(results_df)

    # picking k=4 states (you can justify this choice in your report)
    best_k = 4
    model, ll = fit_hmm_k(best_k, obs_seq, lengths)
    print(f"\nfitted final hmm with k={best_k}, log-likelihood={ll:.2f}")

    # getting the learned matrices
    A = model.transmat_  # how states transition to each other
    B = model.emissionprob_  # what sentiments each state produces

    print("\ntransition matrix A:")
    print(A)
    print("\nemission matrix B (states x observed sentiments 0..4):")
    print(B)

    # plotting transition matrix
    plot_matrix(
        A,
        title=f"transition matrix (k={best_k})",
        xlabel="next state",
        ylabel="current state",
        filename=f"hmm_transition_K{best_k}.png",
    )

    # plotting emission matrix with better labels
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    # debug: check the shape and values of emission matrix
    print(f"\nemission matrix shape: {B.shape}")
    print(f"emission matrix min/max values: {B.min():.4f} / {B.max():.4f}")
    print(f"emission matrix first few rows:\n{B[:2, :5] if B.shape[1] >= 5 else B[:2]}")
    
    # make sure we're plotting the right dimensions
    # B should be (n_states, n_observations) where n_observations = 5 (sentiments 0-4)
    if B.shape[1] == 5:
        # this is correct - plot as is
        emission_to_plot = B
        x_labels = [str(i) for i in range(5)]
    elif B.shape[0] == 5:
        # matrix might be transposed - fix it
        emission_to_plot = B.T
        x_labels = [str(i) for i in range(5)]
    else:
        print(f"warning: unexpected emission matrix shape {B.shape}")
        emission_to_plot = B
        x_labels = [str(i) for i in range(B.shape[1])]
    
    sns.heatmap(
        emission_to_plot,
        annot=True,
        fmt=".3f",  # show 3 decimal places for better precision
        cmap="Greens",
        vmin=0,  # explicitly set color range
        vmax=1,
        yticklabels=[f"state {i}" for i in range(emission_to_plot.shape[0])],
        xticklabels=x_labels,
    )
    plt.title(f"emission probabilities (k={best_k})")
    plt.xlabel("observed sentiment (0–4)")
    plt.ylabel("hidden state")
    plt.tight_layout()
    out = REPORTS_DIR / f"hmm_emission_K{best_k}.png"
    plt.savefig(out, dpi=150)  # higher resolution
    plt.close()
    print(f"saved: {out}")

    # figuring out what hidden state each observation probably came from
    log_prob, hidden_states = model.decode(
        obs_seq,
        lengths=lengths,
        algorithm="viterbi",
    )
    print(f"\nviterbi log-prob: {log_prob:.2f}")
    df_sorted["hidden_state"] = hidden_states

    # seeing what sentiments each hidden state tends to produce
    dist = (
        df_sorted.groupby("hidden_state")["Sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print("\nsentiment distribution per hidden state (rows=states, cols=sentiment 0..4):")
    print(dist)

    # saving this analysis for the report
    dist_out = REPORTS_DIR / "hmm_state_sentiment_distribution.csv"
    dist.to_csv(dist_out)
    print(f"saved: {dist_out}")

    # simple forecasting: predicting the next sentiment based on current state
    last_state = hidden_states[-1]
    next_state_probs = A[last_state]  # where we'll probably go next
    next_obs_probs = next_state_probs @ B  # what sentiment that produces
    predicted_label = int(np.argmax(next_obs_probs))

    print("\nforecast example:")
    print("last hidden state index:", last_state)
    print("next-state distribution:", next_state_probs)
    print("next-observation (sentiment) distribution:", next_obs_probs)
    print("most likely next sentiment label:", predicted_label)


if __name__ == "__main__":
    main()