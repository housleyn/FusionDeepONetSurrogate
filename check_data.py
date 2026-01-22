import os
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
FOLDER = os.path.join("Data", "spheres_data_150")

FLOW_COLS = [
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Absolute Pressure (Pa)",
    "Density (kg/m^3)",
    "Temperature (K)",
]

# Global "extremely large" fallback threshold
GLOBAL_THRESHOLD = 1e10

# Optional: per-column thresholds (recommended). Set None to disable.
PER_COL_THRESHOLDS = {
    "Velocity[i] (m/s)": 1e6,
    "Velocity[j] (m/s)": 1e6,
    "Velocity[k] (m/s)": 1e6,
    "Absolute Pressure (Pa)": 1e12,
    "Density (kg/m^3)": 1e6,
    "Temperature (K)": 1e6,
}

# Extra safety: values this large can overflow if squared/variance computed naively
DANGER_SQUARE_THRESHOLD = 1e150

# Print examples of offending rows (first N)
PRINT_ROW_EXAMPLES = True
MAX_EXAMPLE_ROWS = 3


# =========================
# HELPERS
# =========================
def safe_read_csv(path: str) -> pd.DataFrame:
    # Keep default engine; if you have malformed CSVs, uncomment engine="python"
    return pd.read_csv(path)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Convert to numeric; any non-numeric tokens become NaN
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def check_file(path: str) -> dict:
    df = safe_read_csv(path)

    present_cols = [c for c in FLOW_COLS if c in df.columns]
    rep = {
        "file": os.path.basename(path),
        "checked_cols": present_cols,
        "missing_cols": [c for c in FLOW_COLS if c not in df.columns],
        "row_count": int(len(df)),
        "nan_total": 0,
        "inf_total": 0,
        "too_large_total": 0,
        "danger_square_total": 0,
        "nan_by_col": {},
        "inf_by_col": {},
        "too_large_by_col": {},
        "danger_square_by_col": {},
        "max_abs_by_col": {},
        "examples": [],
        "is_corrupted": False,
        "notes": [],
    }

    if not present_cols:
        rep["notes"].append("No FLOW_COLS found in this CSV.")
        return rep

    num = coerce_numeric(df, present_cols)
    data = num.to_numpy(dtype=np.float64)

    # Counts
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)

    rep["nan_total"] = int(nan_mask.sum())
    rep["inf_total"] = int(inf_mask.sum())

    # Per-column stats + thresholds
    for j, c in enumerate(present_cols):
        col = num[c].to_numpy(dtype=np.float64)

        rep["nan_by_col"][c] = int(np.isnan(col).sum())
        rep["inf_by_col"][c] = int(np.isinf(col).sum())

        # Max abs (ignore NaNs)
        rep["max_abs_by_col"][c] = float(np.nanmax(np.abs(col))) if np.any(~np.isnan(col)) else float("nan")

        # Too large threshold
        thr = PER_COL_THRESHOLDS.get(c, GLOBAL_THRESHOLD) if PER_COL_THRESHOLDS else GLOBAL_THRESHOLD
        too_large_mask = np.isfinite(col) & (np.abs(col) > thr)
        rep["too_large_by_col"][c] = int(too_large_mask.sum())

        # Danger for squaring/variance overflow
        danger_mask = np.isfinite(col) & (np.abs(col) > DANGER_SQUARE_THRESHOLD)
        rep["danger_square_by_col"][c] = int(danger_mask.sum())

    rep["too_large_total"] = int(sum(rep["too_large_by_col"].values()))
    rep["danger_square_total"] = int(sum(rep["danger_square_by_col"].values()))

    # Mark corrupted
    rep["is_corrupted"] = (
        rep["nan_total"] > 0
        or rep["inf_total"] > 0
        or rep["too_large_total"] > 0
        or rep["danger_square_total"] > 0
    )

    # Notes about likely cause
    if rep["nan_total"] > 0:
        rep["notes"].append("NaNs found (could be real NaNs or non-numeric tokens coerced to NaN).")
    if rep["inf_total"] > 0:
        rep["notes"].append("Infs found.")
    if rep["danger_square_total"] > 0:
        rep["notes"].append("Values exceed 1e150; variance/std computations can overflow if squaring is used.")
    if rep["missing_cols"]:
        rep["notes"].append(f"Missing columns: {rep['missing_cols']}")

    # Example rows (first N rows that violate any condition)
    if PRINT_ROW_EXAMPLES and rep["is_corrupted"]:
        bad_rows = set()

        # rows with NaN/Inf in any checked col
        bad_rows.update(np.where(np.any(nan_mask | inf_mask, axis=1))[0].tolist())

        # rows with too-large or danger-square values
        for c in present_cols:
            thr = PER_COL_THRESHOLDS.get(c, GLOBAL_THRESHOLD) if PER_COL_THRESHOLDS else GLOBAL_THRESHOLD
            col = num[c].to_numpy(dtype=np.float64)
            too_large = np.where(np.isfinite(col) & (np.abs(col) > thr))[0]
            danger = np.where(np.isfinite(col) & (np.abs(col) > DANGER_SQUARE_THRESHOLD))[0]
            bad_rows.update(too_large.tolist())
            bad_rows.update(danger.tolist())

        bad_rows = sorted(bad_rows)[:MAX_EXAMPLE_ROWS]
        for r in bad_rows:
            # include a compact snapshot of the flow cols
            rep["examples"].append({"row": int(r), **{c: df.loc[r, c] for c in present_cols}})

    return rep


def main():
    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder not found: {FOLDER}")

    files = [f for f in os.listdir(FOLDER) if f.lower().endswith(".csv")]
    files.sort()

    print(f"Checking {len(files)} files in: {FOLDER}")
    print("Columns checked:", ", ".join(FLOW_COLS))
    print()

    corrupted = []
    for f in files:
        full_path = os.path.join(FOLDER, f)
        rep = check_file(full_path)
        if rep["is_corrupted"]:
            corrupted.append(rep)

    print("\n===========================")
    print(" Summary: Corrupted Files")
    print("===========================\n")

    if not corrupted:
        print("✅ No corrupted files found!")
        return

    print(f"❌ {len(corrupted)} corrupted files found:\n")

    for rep in corrupted:
        print(f"- {rep['file']}  (rows={rep['row_count']})")

        if rep["nan_total"] or rep["inf_total"]:
            print(f"  NaN total: {rep['nan_total']} | Inf total: {rep['inf_total']}")

        if rep["too_large_total"]:
            print(f"  Too-large total: {rep['too_large_total']} (per-col thresholds if enabled)")

        if rep["danger_square_total"]:
            print(f"  Danger-square total (>1e150): {rep['danger_square_total']}")

        # Show per-column max abs (compact)
        max_abs = ", ".join([f"{k}={rep['max_abs_by_col'][k]:.3e}" for k in rep["checked_cols"]])
        print(f"  Max |x|: {max_abs}")

        if rep["notes"]:
            print("  Notes:", " | ".join(rep["notes"]))

        if rep["examples"]:
            print("  Examples:")
            for ex in rep["examples"]:
                row = ex.pop("row")
                print(f"    row {row}: {ex}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
