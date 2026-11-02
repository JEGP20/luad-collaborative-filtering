import os
import pickle
import numpy as np
import pandas as pd

NON_FEAT_COLS = ["time", "event", "case_id", "primary_diagnosis", "image_path"]
CUTOFF_ABS = 0.25
MIN_ROWS_FOR_GROUP_FILTER = 4
INPUT_EMBEDDINGS_PKL = "lung_embeddings_uni_20x_sin_normalizar.pkl"
INPUT_GROUPS_PKL = "block_matching_exp2_groups.pkl"            
FEATURE_ORDER_NPZ = "feature_orders/global.npz"                 
SAVE_PATH = "./"


def load_pkl(path):
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


def haar_lowpass_level1(X, axis):
    n = X.shape[axis]
    if n < 2:
        return X.copy(), 0.0, n
    Xm = np.moveaxis(X, axis, 0)
    pairs = n // 2
    even = Xm[0:2*pairs:2]
    odd = Xm[1:2*pairs:2]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    approx = (even + odd) * inv_sqrt2
    detail = (even - odd) * inv_sqrt2
    e_detail = np.sum(detail ** 2)
    e_total = np.sum(approx ** 2) + e_detail
    if n % 2:
        leftover = Xm[-1:]
        e_total = e_total + np.sum(leftover ** 2)
    rec_pair = approx * inv_sqrt2
    Y = np.empty_like(Xm)
    Y[0:2*pairs:2] = rec_pair
    Y[1:2*pairs:2] = rec_pair
    if n % 2:
        Y[-1] = Xm[-1]
    frac_removed = float(e_detail / max(e_total, 1e-20))
    retained = pairs + (n % 2)
    return np.moveaxis(Y, 0, axis), frac_removed, int(retained)


def fft_lowpass_position(X, axis, cutoff_abs=CUTOFF_ABS):
    n = X.shape[axis]
    freqs = np.fft.fftfreq(n, d=1.0)
    mask = np.abs(freqs) <= cutoff_abs + 1e-12
    shape = [1] * X.ndim
    shape[axis] = n
    m = mask.reshape(shape)
    Z = np.fft.fft(X, axis=axis)
    e0 = np.sum(np.abs(Z) ** 2)
    removed = np.sum(np.abs(Z * (~m)) ** 2)
    Y = np.fft.ifft(Z * m, axis=axis).real
    frac_removed = float(removed / max(e0, 1e-20))
    return Y, frac_removed, int(mask.sum())


def order_by_medoid_similarity(X_norm):
    S = X_norm.sum(axis=0)
    align = X_norm @ S
    medoid_local = int(np.argmax(align))
    sims = X_norm @ X_norm[medoid_local]
    order = np.argsort(-sims)
    return order, medoid_local


def collaborative_filter_group_hybrid(G_k, order_feat, signs_feat, cutoff_abs=CUTOFF_ABS):
    n_k, d = G_k.shape
    norms = np.linalg.norm(G_k, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = G_k / norms

    order_grp, old_medoid_local = order_by_medoid_similarity(X_norm)
    Xo = X_norm[order_grp]

    if n_k >= MIN_ROWS_FOR_GROUP_FILTER:
        Y1, e_grp, k_grp = haar_lowpass_level1(Xo, axis=0)
        small_group = False
    else:
        Y1, e_grp, k_grp = Xo.copy(), 0.0, n_k
        small_group = True

    Y1_ser = Y1[:, order_feat] * signs_feat[None, :]
    Y2_ser, e_feat, k_feat = fft_lowpass_position(Y1_ser, axis=1, cutoff_abs=cutoff_abs)

    inv_order_feat = np.argsort(order_feat)
    Y2 = (Y2_ser * signs_feat[None, :])[:, inv_order_feat]

    inv_order_grp = np.argsort(order_grp)
    Y_final_norm = Y2[inv_order_grp]

    Yn = Y_final_norm / (np.linalg.norm(Y_final_norm, axis=1, keepdims=True) + 1e-12)
    c = Yn.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    new_medoid_local = int(np.argmax(Yn @ c))

    G_filtered = Y_final_norm * norms

    stats = {
        "n_patches": n_k,
        "small_group": small_group,
        "row_haar_detail_energy_removed_frac": e_grp,
        "col_fourier_energy_removed_frac": e_feat,
        "relative_reconstruction_error": float(
            np.linalg.norm(Y_final_norm - X_norm) / max(np.linalg.norm(X_norm), 1e-20)
        ),
        "medoid_changed": bool(new_medoid_local != old_medoid_local),
    }
    return G_filtered, new_medoid_local, old_medoid_local, stats


def main():
    print(f"[1] Loading original embeddings: {INPUT_EMBEDDINGS_PKL}")
    df_emb = load_pkl(INPUT_EMBEDDINGS_PKL)
    X_all, meta_all, feat_cols = parse_df(df_emb)
    case_ids = meta_all["case_id"].values
    print(f"  Total patches: {X_all.shape[0]:,} | Dim embedding: {X_all.shape[1]}")

    print(f"[2] Loading goups: {INPUT_GROUPS_PKL}")
    df_groups = load_pkl(INPUT_GROUPS_PKL)
    assert len(df_groups) == X_all.shape[0]
    group_ids = df_groups["group_id"].values

    print(f"[3] Loading column order: {FEATURE_ORDER_NPZ}")
    z = np.load(FEATURE_ORDER_NPZ)
    order_feat, signs_feat = z["order"], z["signs"]

    print("[4]Filtering the 32 global groups (Haar + Fourier )...")
    X_filtered = np.zeros_like(X_all)
    representative_rows = []
    group_stats_rows = []

    unique_groups = np.unique(group_ids)
    for gid in unique_groups:
        idx_g = np.where(group_ids == gid)[0]
        if len(idx_g) == 0:
            continue
        G_k = X_all[idx_g]
        n_patients_g = len(pd.unique(case_ids[idx_g]))

        G_filtered, new_med_local, old_med_local, stats = collaborative_filter_group_hybrid(
            G_k, order_feat, signs_feat
        )
        X_filtered[idx_g] = G_filtered

        representative_rows.append({
            "group_id": int(gid),
            "orig_row_idx": int(idx_g[new_med_local]),
            "old_medoid_row_idx": int(idx_g[old_med_local]),
            "case_id_representative": case_ids[idx_g[new_med_local]],
            "medoid_changed": stats["medoid_changed"],
        })
        group_stats_rows.append({"group_id": int(gid), "n_patients": n_patients_g, **stats})

        print(f"  Grupo {gid}: {len(idx_g):,} parches de {n_patients_g} pacientes | "
              f"Haar={100*stats['row_haar_detail_energy_removed_frac']:.1f}% "
              f"Fourier={100*stats['col_fourier_energy_removed_frac']:.1f}% "
              f"medoide_cambio={stats['medoid_changed']}", flush=True)

    df_stats = pd.DataFrame(group_stats_rows)
    print(f"\n[Resume global]")
    print(f"  Energy Haar deleted      : {100*df_stats['row_haar_detail_energy_removed_frac'].mean():.1f}%")
    print(f"  Energy Fourier deleted : {100*df_stats['col_fourier_energy_removed_frac'].mean():.1f}%")
    print(f"  % of medoids that changed: {100*df_stats['medoid_changed'].mean():.1f}%")

    # ============ Guardar

    df_filtered_feats = pd.DataFrame(X_filtered, columns=feat_cols)
    df_filtered = pd.concat([meta_all.reset_index(drop=True), df_filtered_feats], axis=1)
    df_filtered["group_id"] = group_ids
    with open(os.path.join(SAVE_PATH, "block_matching_exp2_block2_hybrid_filtered.pkl"), "wb") as f:
        pickle.dump(df_filtered, f)

    df_rep = pd.DataFrame(representative_rows)
    df_rep_feats = X_all[df_rep["orig_row_idx"].values]
    df_rep_full = pd.concat([df_rep.reset_index(drop=True), pd.DataFrame(df_rep_feats, columns=feat_cols)], axis=1)
    with open(os.path.join(SAVE_PATH, "block_matching_exp2_block2_hybrid_representatives.pkl"), "wb") as f:
        pickle.dump(df_rep_full, f)

    df_stats.to_csv(os.path.join(SAVE_PATH, "block_matching_exp2_block2_hybrid_stats.csv"), index=False)

    print("\nArchivos generados:")
    print("  block_matching_exp2_block2_hybrid_filtered.pkl")
    print("  block_matching_exp2_block2_hybrid_representatives.pkl")
    print("  block_matching_exp2_block2_hybrid_stats.csv")


if __name__ == "__main__":
    main()