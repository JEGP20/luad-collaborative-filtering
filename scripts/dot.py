import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


K_GROUPS = 32
RANDOM_SEED = 42
MB_N_INIT = 5
MB_BATCH_SIZE = 4096
MB_MAX_ITER = 300
NON_FEAT_COLS = ["time", "event", "case_id", "primary_diagnosis", "image_path"]
INPUT_PKL = "lung_embeddings_uni_20x_sin_normalizar.pkl"
SAVE_PATH = "./"


def load_df(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_df(df):
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEAT_COLS]
    X = df[feat_cols].values.astype(np.float64)
    keep = ["case_id"]
    for c in ("time", "event", "image_path"):
        if c in df.columns:
            keep.append(c)
    meta = df[keep].reset_index(drop=True)
    return X, meta, feat_cols



def repair_empty_clusters(X_norm, labels, k):
    labels = labels.copy()
    present = set(np.unique(labels))
    missing = [g for g in range(k) if g not in present]
    if not missing:
        return labels, 0
    n_repairs = 0
    for empty_g in missing:
        sizes = pd.Series(labels).value_counts()
        candidates = sizes[sizes > 1]
        if candidates.empty:
            break
        donor = candidates.idxmax()
        donor_idx = np.where(labels == donor)[0]
        donor_centroid = X_norm[donor_idx].mean(axis=0)
        donor_centroid = donor_centroid / (np.linalg.norm(donor_centroid) + 1e-12)
        dists = 1.0 - X_norm[donor_idx] @ donor_centroid
        farthest_local = donor_idx[np.argmax(dists)]
        labels[farthest_local] = empty_g
        n_repairs += 1
    return labels, n_repairs


def pairwise_variance_decomposition_closed_form(X_norm, labels):
    n, d = X_norm.shape
    total_pairs = n * (n - 1) / 2.0
    S_global = X_norm.sum(axis=0)
    sum_dot_global = (float(S_global @ S_global) - n) / 2.0

    sum_dot_same, sum_pairs_same = 0.0, 0.0
    per_group = []
    for k in np.unique(labels):
        idx_k = np.where(labels == k)[0]
        n_k = len(idx_k)
        if n_k > 1:
            S_k = X_norm[idx_k].sum(axis=0)
            dot_sum_k = (float(S_k @ S_k) - n_k) / 2.0
            pairs_k = n_k * (n_k - 1) / 2.0
            sum_dot_same += dot_sum_k
            sum_pairs_same += pairs_k
            intra_k = 1.0 - dot_sum_k / pairs_k
        else:
            intra_k = np.nan
        per_group.append({"group_id": int(k), "n_patches": n_k, "intra_var_group": intra_k})

    sum_dist_same = sum_pairs_same - sum_dot_same
    intra_var = (sum_dist_same / sum_pairs_same) if sum_pairs_same > 0 else np.nan

    sum_pairs_diff = total_pairs - sum_pairs_same
    sum_dot_diff = sum_dot_global - sum_dot_same
    sum_dist_diff = sum_pairs_diff - sum_dot_diff
    inter_var = (sum_dist_diff / sum_pairs_diff) if sum_pairs_diff > 0 else np.nan

    MIN_INTRA = 1e-6
    sep_ratio = (inter_var / intra_var) if (intra_var is not None and intra_var > MIN_INTRA) else np.nan

    return {
        "intra_var": intra_var, "inter_var": inter_var,
        "separation_ratio": sep_ratio, "per_group": per_group,
    }


def medoid_and_centroid_diagnostics(X_norm, labels, k):
    n, d = X_norm.shape
    overall_mean = X_norm.mean(axis=0)
    medoids = np.zeros((k, d))
    medoid_indices = np.full(k, -1, dtype=int)
    group_sizes = np.zeros(k, dtype=int)

    intra_classic_num, inter_classic_num, intra_medoid_num = 0.0, 0.0, 0.0
    total_n = 0

    for g in range(k):
        idx_g = np.where(labels == g)[0]
        n_g = len(idx_g)
        group_sizes[g] = n_g
        if n_g == 0:
            continue
        Xg = X_norm[idx_g]
        c = Xg.mean(axis=0)
        c_norm = c / (np.linalg.norm(c) + 1e-12)
        align = Xg @ c_norm
        local_medoid = int(np.argmax(align))
        medoid_indices[g] = idx_g[local_medoid]
        medoids[g] = Xg[local_medoid]

        intra_classic_num += np.sum((Xg - c) ** 2)
        inter_classic_num += n_g * np.sum((c - overall_mean) ** 2)
        intra_medoid_num += (1.0 - Xg @ medoids[g]).sum()
        total_n += n_g

    V_intra_classic = intra_classic_num / total_n if total_n > 0 else np.nan
    V_inter_classic = inter_classic_num / total_n if total_n > 0 else np.nan
    medoid_intra = intra_medoid_num / total_n if total_n > 0 else np.nan

    valid_g = np.where(group_sizes > 0)[0]
    if len(valid_g) > 1:
        M = medoids[valid_g]
        dist_m = 1.0 - np.clip(M @ M.T, -1.0, 1.0)
        iu = np.triu_indices(len(valid_g), k=1)
        medoid_inter = dist_m[iu].mean()
    else:
        medoid_inter = np.nan

    return {
        "medoids": medoids, "medoid_indices": medoid_indices,
        "V_intra_classic": V_intra_classic, "V_inter_classic": V_inter_classic,
        "medoid_intra": medoid_intra, "medoid_inter": medoid_inter,
    }



def generate_diagnostic_plots(global_stats, df_per_group, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = df_per_group.sort_values("n_patches", ascending=False)
    axes[0].bar(range(len(order)), order["n_patches"], color="#4C72B0")
    axes[0].set_title("Size of each global group")
    axes[0].set_xlabel("Group (order, largest to smallest)")
    axes[0].set_ylabel("Patches on the group")

    axes[1].bar(range(len(order)), order["n_unique_patients"], color="#55A868")
    axes[1].set_title("Different patients per group")
    axes[1].set_xlabel("Group")
    axes[1].set_ylabel("Different patients")
    axes[1].axhline(global_stats["n_patients"], color="red", linestyle="--",
                     label=f"Total of patients ({global_stats['n_patients']})")
    axes[1].legend()
    plt.tight_layout()
    out1 = os.path.join(out_dir, "cluster_size_distribution_exp2.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    order2 = df_per_group.sort_values("intra_var_group")
    ax.bar(range(len(order2)), order2["intra_var_group"], color="#4C72B0", label="intra_var del grupo")
    ax.axhline(global_stats["inter_var"], color="red", linestyle="--", linewidth=2,
               label=f"inter_var global = {global_stats['inter_var']:.3f}")
    ax.set_title("Variance intra vs. inter global ")
    ax.set_xlabel("Group (order intra_var, smallest to largest)")
    ax.set_ylabel("Distance coseno prom")
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(out_dir, "baseline_inter_intra_variance_exp2.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out1, out2


def main():
    print(f"[1] load embeddings: {INPUT_PKL}")
    df = load_df(INPUT_PKL)
    X_all, meta_all, feat_cols = parse_df(df)
    case_ids = meta_all["case_id"].values
    n_patients = len(pd.unique(case_ids))
    print(f"  Total patches: {X_all.shape[0]:,} | Dim embedding: {X_all.shape[1]} | Patients: {n_patients}")

    print("[2] Normalize (L2, coseno)...")
    X_norm = normalize(X_all, norm="l2", axis=1)

    print(f"[3] K-Means global (k={K_GROUPS}) sobre los {X_all.shape[0]:,} complete patches...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K_GROUPS, random_state=RANDOM_SEED,
                          n_init=MB_N_INIT, batch_size=MB_BATCH_SIZE, max_iter=MB_MAX_ITER)
    labels = km.fit_predict(X_norm)
    labels, n_repairs = repair_empty_clusters(X_norm, labels, K_GROUPS)
    t_fit = time.time() - t0
    print(f"  {t_fit:.1f}s | repair groups: {n_repairs}")

    print("[4] variance inter/intra")
    var_stats = pairwise_variance_decomposition_closed_form(X_norm, labels)
    diag_stats = medoid_and_centroid_diagnostics(X_norm, labels, K_GROUPS)
    print(f"  intra_var = {var_stats['intra_var']:.4f} | "
          f"inter_var = {var_stats['inter_var']:.4f} | "
          f"ratio = {var_stats['separation_ratio']:.2f}")


    print("[5]  Patient distribution per group...")
    group_patient_stats = []
    for g in range(K_GROUPS):
        mask_g = labels == g
        n_patches_g = int(mask_g.sum())
        cids_g = case_ids[mask_g]
        vc = pd.Series(cids_g).value_counts() if n_patches_g > 0 else pd.Series(dtype=int)
        group_patient_stats.append({
            "group_id": g,
            "n_patches": n_patches_g,
            "n_unique_patients": int(vc.shape[0]),
            "intra_var_group": next((p["intra_var_group"] for p in var_stats["per_group"]
                                      if p["group_id"] == g), np.nan),
        })
    df_per_group = pd.DataFrame(group_patient_stats)

    global_stats_dict = {
        "intra_var": var_stats["intra_var"],
        "inter_var": var_stats["inter_var"],
        "separation_ratio": var_stats["separation_ratio"],
        "V_intra_classic": diag_stats["V_intra_classic"],
        "V_inter_classic": diag_stats["V_inter_classic"],
        "medoid_intra": diag_stats["medoid_intra"],
        "medoid_inter": diag_stats["medoid_inter"],
        "n_patients": n_patients,
        "n_repairs": n_repairs,
    }

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    df_groups = meta_all.copy()
    df_groups["group_id"] = labels
    with open(os.path.join(SAVE_PATH, "block_matching_exp2_groups.pkl"), "wb") as f:
        pickle.dump(df_groups, f)

    with open(os.path.join(SAVE_PATH, "block_matching_exp2_summary.pkl"), "wb") as f:
        pickle.dump({"global": global_stats_dict, "per_group": df_per_group}, f)
    df_per_group.to_csv(os.path.join(SAVE_PATH, "block_matching_exp2_summary.csv"), index=False)

    fig1, fig2 = generate_diagnostic_plots(global_stats_dict, df_per_group)
    print(f"  Figuras: {fig1} | {fig2}")

    print("\nGenerated files:")
    print("  block_matching_exp2_groups.pkl   (case_id, group_id por parche, TODOS los pacientes)")
    print("  block_matching_exp2_summary.pkl / .csv  (varianza global + desglose por grupo)")
    print("  figures/cluster_size_distribution_exp2.png")
    print("  figures/baseline_inter_intra_variance_exp2.png")


if __name__ == "__main__":
    main()