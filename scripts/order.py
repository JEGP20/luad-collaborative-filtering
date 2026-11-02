import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

EMBEDDINGS_PER_PATIENT = 128
SEED = 42
NON_FEAT_COLS = ["time", "event", "case_id", "primary_diagnosis", "image_path"]
INPUT_EMBEDDINGS_PKL = "lung_embeddings_uni_20x_unnormalized.pkl"
SAVE_PATH = "./"
ORDERS_DIR = "feature_orders"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_df(df):
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEAT_COLS]
    X = df[feat_cols].values.astype(np.float64)
    case_ids = df["case_id"].values
    return X, case_ids, feat_cols


def balanced_sample(X_all, case_ids, patients, n_per_patient, seed):
    rng = np.random.default_rng(seed)
    idx_all = []
    for p in patients:
        idx_p = np.where(case_ids == p)[0]
        if len(idx_p) == 0:
            continue
        if len(idx_p) <= n_per_patient:
            idx_all.extend(idx_p.tolist())
        else:
            idx_all.extend(rng.choice(idx_p, size=n_per_patient, replace=False).tolist())
    return np.array(idx_all, dtype=int)


def seriation_signed_permutation(X_sample):
    d = X_sample.shape[1]
    corr = np.corrcoef(X_sample, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - np.abs(np.clip(corr, -1, 1))
    np.fill_diagonal(dist, 0)
    y = squareform(dist, checks=False)
    Z = linkage(y, method="average")
    Z = optimal_leaf_ordering(Z, y)
    order = leaves_list(Z)

    signs = np.ones(d)
    for i in range(1, d):
        prev, curr = order[i - 1], order[i]
        effective_corr = corr[prev, curr] * signs[i - 1]
        if effective_corr < 0:
            signs[i] = -1

    return order, signs, corr


def mean_adjacent_abs_corr(corr, order):
    vals = [abs(corr[order[i], order[i + 1]]) for i in range(len(order) - 1)]
    return float(np.mean(vals))


def verify_orthogonal(X_sample, order, signs):
    a, b = X_sample[0], X_sample[min(5, len(X_sample) - 1)]
    dot_before = float(a @ b)
    norm_before = float(np.linalg.norm(a))
    a2 = a[order] * signs
    b2 = b[order] * signs
    dot_after = float(a2 @ b2)
    norm_after = float(np.linalg.norm(a2))
    return abs(dot_before - dot_after) < 1e-6 and abs(norm_before - norm_after) < 1e-6


def main():
    os.makedirs(ORDERS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"[1] Loading embeddings: {INPUT_EMBEDDINGS_PKL}")
    df = load_pkl(INPUT_EMBEDDINGS_PKL)
    X_all, case_ids, feat_cols = parse_df(df)
    print(f"  Total patches: {X_all.shape[0]:,} | Embedding dim: {X_all.shape[1]}")

    patients = sorted(pd.unique(case_ids))
    print(f"  Patients: {len(patients)}")

    print(f"\n[2] Sampling {EMBEDDINGS_PER_PATIENT} patches/patient to estimate correlation...")
    idx_sample = balanced_sample(X_all, case_ids, patients, EMBEDDINGS_PER_PATIENT, seed=SEED)
    X_sample = X_all[idx_sample]
    X_sample_norm = X_sample / (np.linalg.norm(X_sample, axis=1, keepdims=True) + 1e-12)
    print(f"  Sample size: {len(idx_sample):,} patches")

    print("\n[3] Computing signed seriation...")
    order, signs, corr = seriation_signed_permutation(X_sample_norm)
    ok = verify_orthogonal(X_sample_norm, order, signs)
    gain_before = mean_adjacent_abs_corr(corr, np.arange(len(order)))
    gain_after = mean_adjacent_abs_corr(corr, order)

    np.savez(os.path.join(ORDERS_DIR, "global.npz"), order=order, signs=signs)
    print(f"  Neighbor correlation -- before: {gain_before:.3f} -> after: {gain_after:.3f} | "
          f"geometry preserved: {ok}")

    df_summary = pd.DataFrame([{
        "n_patients": len(patients), "n_sampled": len(idx_sample),
        "adj_corr_before": gain_before, "adj_corr_after": gain_after,
        "orthogonal_check_ok": ok,
    }])
    df_summary.to_csv(os.path.join(TABLES_DIR, "feature_order_summary.csv"), index=False)

    # ============ Figures

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Original order\n(arbitrary)", "Seriated order\n(signed)"],
           [gain_before, gain_after], color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("Average absolute correlation between neighbors")
    ax.set_title("Seriation gain (single fit, no folds)")
    plt.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "feature_order_correlation_gain.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    im0 = axes[0].imshow(np.abs(corr), cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Original order (arbitrary)")
    im1 = axes[1].imshow(np.abs(corr[np.ix_(order, order)]), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Seriated order")
    fig.suptitle("|Correlation| between 1536 dim")
    fig.colorbar(im1, ax=axes, shrink=0.8, label="|correlation|")
    out2 = os.path.join(FIGURES_DIR, "feature_order_correlation_matrix.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nGenerated files:")
    print(f"  {ORDERS_DIR}/global.npz")
    print(f"  {TABLES_DIR}/feature_order_summary.csv")
    print(f"  {out1}")
    print(f"  {out2}")


if __name__ == "__main__":
    main()