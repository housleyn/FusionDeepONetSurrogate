from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Filename parsers
# -----------------------------
ORION_RE = re.compile(
    r"""^orion_data_
        AoA(?P<AoA>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_
        Mach(?P<Mach>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \.csv$""",
    re.VERBOSE,
)

X43_RE = re.compile(
    r"""^x_43_
        a2(?P<a2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_
        a3(?P<a3>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_
        a4(?P<a4>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \.csv$""",
    re.VERBOSE,
)


# -----------------------------
# Scan folders + assign File #
# -----------------------------
def scan_orion(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))

    rows = []
    file_id = 1

    for p in files:
        m = ORION_RE.match(p.name)
        if m:
            rows.append({
                "Inference": file_id,
                "AoA": float(m.group("AoA")),
                "Mach": float(m.group("Mach")),
            })
            file_id += 1

    return pd.DataFrame(rows)


def scan_x43(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))

    rows = []
    file_id = 1

    for p in files:
        m = X43_RE.match(p.name)
        if m:
            rows.append({
                "Inference": file_id,
                "a2": float(m.group("a2")),
                "a3": float(m.group("a3")),
                "a4": float(m.group("a4")),
            })
            file_id += 1

    return pd.DataFrame(rows)


# -----------------------------
# Save dataframe as PNG table
# -----------------------------
def save_table_png(df, filename, title=None):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(len(df.columns)*2, len(df)*0.5 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df.round(6).values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------
def main(orion_dir, x43_dir, out_dir="combo_outputs"):

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Build tables
    orion_table = scan_orion(orion_dir)
    x43_table = scan_x43(x43_dir)

    summary = pd.DataFrame([
        {"dataset": "orion", "n_files": len(orion_table)},
        {"dataset": "x_43", "n_files": len(x43_table)},
    ])

    # ---- CSV export (for Word)
    orion_table.to_csv(out_dir / "orion_table.csv", index=False)
    x43_table.to_csv(out_dir / "x43_table.csv", index=False)
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)

    # ---- Optional PNG tables
    save_table_png(orion_table, out_dir / "orion_table.png", "Orion Files")
    save_table_png(x43_table, out_dir / "x43_table.png", "X-43 Files")
    save_table_png(summary, out_dir / "comparison_summary.png", "Dataset Summary")

    print("Tables saved to:", out_dir.resolve())



if __name__ == "__main__":
    ORION_DIR = r"Data/orion_unseen_32"
    X43_DIR   = r"Data/x_43_unseen_20"

    main(ORION_DIR, X43_DIR)
