import os
import pandas as pd
import numpy as np

# Path to the folder
folder = os.path.join("Data", "orion_low_fi_data")

# Threshold for "extremely large" values
THRESHOLD = 1e10   # adjust if needed

# Columns we actually care about
FLOW_COLS = [
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Absolute Pressure (Pa)",
    "Density (kg/m^3)",
    "Temperature (K)",
]

def check_file(path):
    df = pd.read_csv(path)

    # Only check flow-field columns
    flow_cols = [c for c in FLOW_COLS if c in df.columns]

    if not flow_cols:
        return {
            "file": os.path.basename(path),
            "checked_cols": [],
            "nan_count": 0,
            "inf_count": 0,
            "too_large": 0,
        }

    data = df[flow_cols].values

    report = {
        "file": os.path.basename(path),
        "checked_cols": flow_cols,
        "nan_count": np.isnan(data).sum(),
        "inf_count": np.isinf(data).sum(),
        "too_large": np.sum(np.abs(data) > THRESHOLD),
    }

    return report

def main():
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    print(f"Checking {len(files)} files (velocity/pressure/density/temp only)...\n")

    corrupted_files = []

    for f in files:
        full_path = os.path.join(folder, f)
        rep = check_file(full_path)


        # Track corrupted files
        if rep["too_large"] > 0 or rep["nan_count"] > 0 or rep["inf_count"] > 0:
            corrupted_files.append(rep["file"])

    # === FINAL SUMMARY ===
    print("\n===========================")
    print(" Summary: Corrupted Files")
    print("===========================\n")

    if not corrupted_files:
        print("✅ No corrupted files found!")
    else:
        print("❌ Files containing values above threshold, NaNs, or Infs:")
        for f in corrupted_files:
            print("  -", f)

if __name__ == "__main__":
    main()
