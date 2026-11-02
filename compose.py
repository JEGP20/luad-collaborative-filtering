########################Compose k= 32 groups)#################################################
import os
import re
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


NAS_DIR = "/home/jguarama/SegCode/Nas/Data/LUAD/BM3D/32globales/nosplit/csv"
GROUPS_PKL = os.path.join(NAS_DIR, "block_matching_exp2_block2_hybrid_filtered.pkl")
N_GROUPS = 32
PATIENT_ID = "TCGA-49-AAR9"
PATIENT_SEED = 42
CELL_SIZE = 48
MAX_CANVAS_DIM = 6000     
MIN_CELL_SIZE = 6
GROUP_ALPHA = 0.50
SAVE_PATH = "./phenotype_figures"



def build_distinct_palette():
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        colors.extend([cmap(i) for i in range(cmap.N)])
    return colors


PALETTE = build_distinct_palette()


def group_color_rgb255(group_id):
    r, g, b, _a = PALETTE[int(group_id) % len(PALETTE)]
    return int(r * 255), int(g * 255), int(b * 255)


def group_color_hex(group_id):
    r, g, b = group_color_rgb255(group_id)
    return f"#{r:02x}{g:02x}{b:02x}"


PATCH_NAME_RE = re.compile(r"_(\d+)x_(\d+)px_x(\d+)_y(\d+)\.png$")


def parse_patch_filename(path):
    m = PATCH_NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    mag, px, x, y = m.groups()
    return int(mag), int(px), int(x), int(y)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"[1] Loading group assignments: {GROUPS_PKL}")
    df = load_pkl(GROUPS_PKL)

    required_cols = {"case_id", "image_path", "group_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Group file is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"    Total patches: {len(df):,} | patients: {df['case_id'].nunique()}")

    if PATIENT_ID is None:
        rng = random.Random(PATIENT_SEED)
        patient_id = rng.choice(sorted(df["case_id"].unique().tolist()))
        print(f"[2] Randomly selected patient (seed={PATIENT_SEED}): {patient_id}")
    else:
        patient_id = PATIENT_ID
        if patient_id not in set(df["case_id"].unique()):
            raise ValueError(f"'{patient_id}' does not exist in {GROUPS_PKL}")
        print(f"[2] Fixed patient: {patient_id}")

    df_pat = df[df["case_id"] == patient_id].reset_index(drop=True)
    n_patches_total = len(df_pat)
    print(f"    Patches for this patient: {n_patches_total:,}")

    counts = df_pat["group_id"].value_counts().reindex(range(N_GROUPS), fill_value=0)
    quant = pd.DataFrame({
        "group_id": counts.index,
        "n_patches": counts.values,
        "pct": (counts.values / n_patches_total * 100.0) if n_patches_total > 0 else 0.0,
    }).sort_values("n_patches", ascending=False).reset_index(drop=True)

    quant_csv_path = os.path.join(SAVE_PATH, f"phenotypic_composition_{patient_id}.csv")
    quant.to_csv(quant_csv_path, index=False)
    print(f"[3] Quantification saved: {quant_csv_path}")
    print(quant.head(10).to_string(index=False))
    print("[4] Parsing patch coordinates...")
    parsed = df_pat["image_path"].apply(parse_patch_filename)
    ok_mask = parsed.notna()
    n_unparsed = int((~ok_mask).sum())
    if n_unparsed > 0:
        print(f"    WARNING: {n_unparsed:,} patches do not match the expected naming pattern -- skipped from mosaic (still counted in quantification above).")

    df_geo = df_pat[ok_mask].copy()
    coords = parsed[ok_mask].apply(pd.Series)
    coords.columns = ["magnification", "patch_size_px", "coord_x", "coord_y"]
    df_geo = pd.concat([df_geo.reset_index(drop=True), coords.reset_index(drop=True)], axis=1)

    patch_size_px = int(df_geo["patch_size_px"].mode().iloc[0])
    print(f"    Most common patch_size_px: {patch_size_px} (used as grid step)")

    # Grid indices based on actual position (x, y)
    df_geo["col"] = (df_geo["coord_x"] / patch_size_px).round().astype(int)
    df_geo["row"] = (df_geo["coord_y"] / patch_size_px).round().astype(int)
    df_geo["col"] -= df_geo["col"].min()
    df_geo["row"] -= df_geo["row"].min()

    n_cols = int(df_geo["col"].max()) + 1
    n_rows = int(df_geo["row"].max()) + 1
    print(f"    Grid: {n_cols} columns x {n_rows} rows")
    cell = CELL_SIZE
    max_dim_cells = max(n_cols, n_rows)
    if max_dim_cells * cell > MAX_CANVAS_DIM:
        cell = max(MIN_CELL_SIZE, MAX_CANVAS_DIM // max_dim_cells)
        print(f"    Large grid -- reducing CELL_SIZE to {cell}px to avoid exceeding {MAX_CANVAS_DIM}px side length")

    canvas_w = n_cols * cell
    canvas_h = n_rows * cell
    print(f"    Final canvas: {canvas_w} x {canvas_h} px")


    base = Image.new("RGB", (canvas_w, canvas_h), color=(235, 235, 235))
    overlay = Image.new("RGBA", (canvas_w, canvas_h), color=(0, 0, 0, 0))

    n_loaded, n_failed = 0, 0
    alpha255 = int(GROUP_ALPHA * 255)

    for _, row in df_geo.iterrows():
        x0 = int(row["col"]) * cell
        y0 = int(row["row"]) * cell

        try:
            img = Image.open(row["image_path"]).convert("RGB").resize((cell, cell), Image.LANCZOS)
            base.paste(img, (x0, y0))
            n_loaded += 1
        except Exception:
            n_failed += 1

        r, g, b = group_color_rgb255(row["group_id"])
        overlay.paste((r, g, b, alpha255), (x0, y0, x0 + cell, y0 + cell))

    print(f"[5] Actual mosaic: {n_loaded:,} patches loaded, {n_failed:,} failed")

    composite = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    mosaic_path = os.path.join(SAVE_PATH, f"phenotype_mosaic_{patient_id}.png")
    composite.save(mosaic_path, quality=95)
    print(f"    Overlay mosaic saved: {mosaic_path}")

    present = quant[quant["n_patches"] > 0].reset_index(drop=True)

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.35)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(composite)
    ax_img.set_title(f"Phenotypic composition -- patient {patient_id}\n"
                      f"{n_patches_total:,} patches | {len(present)} groups present",
                      fontsize=12)
    ax_img.axis("off")


    ax_leg = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(present))[::-1]
    colors_bar = [group_color_hex(g) for g in present["group_id"]]
    ax_leg.barh(y_pos, present["pct"], color=colors_bar, edgecolor="black", linewidth=0.3)

    max_pct = float(present["pct"].max()) if len(present) else 1.0
    ax_leg.set_xlim(0, max_pct * 1.55)
    for y, g, n, p in zip(y_pos, present["group_id"], present["n_patches"], present["pct"]):
        ax_leg.text(p + max_pct * 0.03, y, f"G{int(g)}  {p:.1f}% ({n:,})",
                    va="center", ha="left", fontsize=8)

    ax_leg.set_yticks([])
    ax_leg.set_xlabel("% of patient patches")
    ax_leg.set_title("Quantification by group", fontsize=11)
    ax_leg.grid(axis="x", alpha=0.2)
    for spine in ("top", "right", "left"):
        ax_leg.spines[spine].set_visible(False)

    fig_path = os.path.join(SAVE_PATH, f"phenotype_composition_{patient_id}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6] Final figure (mosaic + quantification) saved: {fig_path}")

    print("\nDone. Generated files:")
    print(f"  - {mosaic_path}  (overlay mosaic only, full size)")
    print(f"  - {fig_path}  (mosaic + quantification bar, for review/reporting)")
    print(f"  - {quant_csv_path}  (table: group_id, n_patches, pct)")


if __name__ == "__main__":
    main()