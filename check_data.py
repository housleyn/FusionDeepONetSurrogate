import os
import pandas as pd
import numpy as np

# Path to the folder
folder = os.path.join("Data", "spheres_HF_60")

# Threshold for "extremely large" values
THRESHOLD = 1e10   # adjust if needed

def check_file(path):
    df = pd.read_csv(path)

    report = {
        "file": os.path.basename(path),
        "nan_count": df.isna().sum().sum(),
        "inf_count": np.isinf(df.values).sum(),
        "too_large": np.sum(np.abs(df.values) > THRESHOLD)
    }

    return report

def main():
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    print(f"Checking {len(files)} files...\n")

    for f in files:
        full_path = os.path.join(folder, f)
        rep = check_file(full_path)

        print(f"--- {rep['file']} ---")
        print(f" NaNs:       {rep['nan_count']}")
        print(f" Infs:       {rep['inf_count']}")
        print(f" >{THRESHOLD}: {rep['too_large']}")
        print()

if __name__ == "__main__":
    main()
