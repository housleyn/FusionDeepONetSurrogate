import os
import re
import shutil
import numpy as np
from scipy.stats import qmc


def select_and_copy_subset_files(
    source_folder,
    n_select,
    subset_folder_name="x43_subset",
    dry_run=False,
    random_seed=42,
):
    """
    Select files named like either:
        x_43_a21.234131463_a314.25106432_a415.37972993.csv
    or
        x_43_a21.150067306_a322.39658073_a44.094249377_low_fi.csv

    Uses Latin Hypercube Sampling over (a2, a3, a4), maps each sampled point
    to the nearest real file, then COPIES those files into a sibling folder
    named `subset_folder_name`.
    """

    pattern = re.compile(
        r"^x_43_a2([-+]?\d*\.?\d+)_a3([-+]?\d*\.?\d+)_a4([-+]?\d*\.?\d+)(?:_low_fi)?\.csv$"
    )

    records = []
    for fname in os.listdir(source_folder):
        match = pattern.match(fname)
        if match:
            a2 = float(match.group(1))
            a3 = float(match.group(2))
            a4 = float(match.group(3))
            records.append({
                "filename": fname,
                "a2": a2,
                "a3": a3,
                "a4": a4,
            })

    if not records:
        raise ValueError(
            f"No matching files found in source folder: {source_folder}"
        )

    if n_select > len(records):
        raise ValueError(
            f"Requested {n_select} files, but only {len(records)} matching files exist."
        )

    a2_vals = np.array([r["a2"] for r in records], dtype=float)
    a3_vals = np.array([r["a3"] for r in records], dtype=float)
    a4_vals = np.array([r["a4"] for r in records], dtype=float)

    a2_min, a2_max = a2_vals.min(), a2_vals.max()
    a3_min, a3_max = a3_vals.min(), a3_vals.max()
    a4_min, a4_max = a4_vals.min(), a4_vals.max()

    def normalize(vals, vmin, vmax):
        if np.isclose(vmax, vmin):
            return np.zeros_like(vals)
        return (vals - vmin) / (vmax - vmin)

    real_points = np.column_stack([
        normalize(a2_vals, a2_min, a2_max),
        normalize(a3_vals, a3_min, a3_max),
        normalize(a4_vals, a4_min, a4_max),
    ])

    sampler = qmc.LatinHypercube(d=3, seed=random_seed)
    lhs_points = sampler.random(n=n_select)

    available = set(range(len(records)))
    selected_indices = []

    for target in lhs_points:
        avail_list = np.array(sorted(available))
        avail_points = real_points[avail_list]

        dists = np.linalg.norm(avail_points - target, axis=1)
        best_local_idx = np.argmin(dists)
        best_global_idx = avail_list[best_local_idx]

        selected_indices.append(best_global_idx)
        available.remove(best_global_idx)

    selected_files = sorted([records[i]["filename"] for i in selected_indices])

    parent_dir = os.path.dirname(os.path.abspath(source_folder))
    subset_folder = os.path.join(parent_dir, subset_folder_name)

    if dry_run:
        print(f"[DRY RUN] Would copy {len(selected_files)} files to:")
        print(subset_folder)
        print()
        for fname in selected_files:
            print(fname)
        return selected_files

    os.makedirs(subset_folder, exist_ok=True)

    for fname in selected_files:
        src = os.path.join(source_folder, fname)
        dst = os.path.join(subset_folder, fname)
        shutil.copy2(src, dst)
        print(f"Copied: {fname}")

    print(f"\nCopied {len(selected_files)} files to: {subset_folder}")
    return selected_files


if __name__ == "__main__":

    base_sources = [
        r"Data/x_43_data",
        r"Data/x_43_low_fi_data",
    ]

    subset_sizes = [200]

    for source_folder in base_sources:
        dataset_name = os.path.basename(source_folder)

        for n_select in subset_sizes:
            subset_name = f"{dataset_name}_subset_{n_select}"

            print("\n----------------------------------")
            print(f"Creating subset: {subset_name}")
            print("----------------------------------")

            select_and_copy_subset_files(
                source_folder=source_folder,
                n_select=n_select,
                subset_folder_name=subset_name,
                dry_run=False,   # set True first if you want to preview
                random_seed=42,
            )