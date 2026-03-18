import pandas as pd
import os
import csv

folder = "Data/waverider_hf_data"

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        if "Z_slice" not in df.columns:

            if "Z (m)" not in df.columns or "Mach" not in df.columns:
                print(f"Skipping {file} (missing required columns)")
                continue

            # Copy column
            df["Z_slice"] = df["Z (m)"]

            # Move column after Mach
            cols = list(df.columns)
            cols.remove("Z_slice")
            mach_index = cols.index("Mach")
            cols.insert(mach_index + 1, "Z_slice")
            df = df[cols]

        # Save with quotes preserved
        df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_ALL
        )

        print(f"Updated {file}")