from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import MultinomialHMM

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"


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


def prepare_sequences(df: pd.DataFrame):
    # Sort so phrases in a sentence are in order
    df_sorted = df.sort_values(["SentenceId", "PhraseId"]).reset_index(drop=True)

    # Observations: sentiment labels 0..4
    obs = df_sorted["Sentiment"].astype(int).to_numpy()
    obs_seq = obs.reshape(-1, 1)

    # Sequence lengths: number of phrases per sentence
    lengths = df_sorted.groupby("SentenceId").size().tolist()

    return df_sorted, obs_seq, lengths


def fit_hmm_k(n_states: int, obs_seq: np.ndarray, lengths):
    model = MultinomialHMM(
        n_components=n_states,
        n_iter=100,
        random_state=42,
    )
    model.fit(obs_seq, lengths)
    log_likelihood = model.score(obs_seq, lengths)
    return model, log_likelihood


def compare_k_values(obs_seq, lengths, k_values=(2, 3, 4, 5)):
    results = []
    for k in k_values:
        model_k, ll = fit_hmm_k(k, obs_seq, lengths)
        results.append({"n_states": k, "log_likelihood": ll})
        print(f"K={k}, log-likelihood={ll:.2f}")
    return pd.DataFrame(results)


def plot_matrix(mat: np.ndarray, title: str, xlabel: str, ylabel: str, filename: str):
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
    print(f"Saved: {out}")


def main():
    df = load_data()
    df_sorted, obs_seq, lengths = prepare_sequences(df)

    print(f"Total observations: {obs_seq.shape[0]}, "
          f"Number of sentences: {len(lengths)}")

    print("\nComparing different numbers of hidden states...")
    results_df = compare_k_values(obs_seq, lengths, k_values=(2, 3, 4, 5))
    print("\nK vs log-likelihood:")
    print(results_df)

    # Choose a K (e.g., 4) – in a report you can justify this choice
    best_k = 4
    model, ll = fit_hmm_k(best_k, obs_seq, lengths)
    print(f"\nFitted final HMM with K={best_k}, log-likelihood={ll:.2f}")

    # Transition and emission matrices
    A = model.transmat_
    B = model.emissionprob_

    print("\nTransition matrix A:")
    print(A)
    print("\nEmission matrix B (states x observed sentiments 0..4):")
    print(B)

    plot_matrix(
        A,
        title=f"Transition matrix (K={best_k})",
        xlabel="Next state",
        ylabel="Current state",
        filename=f"hmm_transition_K{best_k}.png",
    )

    # For emissions, change labels on x-axis
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
    B,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    yticklabels=[f"State {i}" for i in range(best_k)],
    xticklabels=[str(i) for i in range(5)],
)
    plt.title(f"Emission probabilities (K={best_k})")
    plt.xlabel("Observed Sentiment (0–4)")
    plt.ylabel("Hidden State")
    plt.tight_layout()
    out = REPORTS_DIR / f"hmm_emission_K{best_k}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # Decode hidden states (Viterbi)
    log_prob, hidden_states = model.decode(
        obs_seq,
        lengths=lengths,
        algorithm="viterbi",
    )
    print(f"\nViterbi log-prob: {log_prob:.2f}")
    df_sorted["hidden_state"] = hidden_states

    # Summary: how each hidden state relates to sentiments
    dist = (
        df_sorted.groupby("hidden_state")["Sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print("\nSentiment distribution per hidden state (rows=states, cols=sentiment 0..4):")
    print(dist)

    # Save table to CSV for report
    dist_out = REPORTS_DIR / "hmm_state_sentiment_distribution.csv"
    dist.to_csv(dist_out)
    print(f"Saved: {dist_out}")

    # Simple forecasting example: next sentiment given last state
    last_state = hidden_states[-1]
    next_state_probs = A[last_state]
    next_obs_probs = next_state_probs @ B
    predicted_label = int(np.argmax(next_obs_probs))

    print("\nForecast example:")
    print("Last hidden state index:", last_state)
    print("Next-state distribution:", next_state_probs)
    print("Next-observation (sentiment) distribution:", next_obs_probs)
    print("Most likely next sentiment label:", predicted_label)


if __name__ == "__main__":
    main()