import os
import re
import shutil
import numpy as np
from scipy.stats import qmc


def select_and_move_unseen_files(
    source_folder,
    n_select,
    comparison_folder=None,
    unseen_folder_name="waverider_unseen",
    dry_run=False,
    random_seed=42,
):
    """
    Select files named like:
        3D_slice_Mach_5.0600_z_-3.471429.csv

    Uses Latin Hypercube Sampling over (Mach, z), maps each sampled point
    to the nearest real file, then MOVES those files into a sibling folder
    named `waverider_unseen`.

    If comparison_folder is provided, any file whose NAME already exists in
    that folder will be excluded from selection.

    Parameters
    ----------
    source_folder : str
        Folder containing the CSV files.
    n_select : int
        Number of files to move out as unseen/evaluation files.
    comparison_folder : str or None
        Folder to check for duplicate filenames. If a filename exists there,
        it will not be selected.
    unseen_folder_name : str
        Name of sibling folder to create beside source_folder.
    dry_run : bool
        If True, only prints selected files without moving them.
    random_seed : int
        Random seed for reproducibility.
    """

    pattern = re.compile(
        r"^3D_slice_Mach_([-+]?\d*\.?\d+)_z_([-+]?\d*\.?\d+)\.csv$"
    )

    # --- Collect filenames to exclude from comparison folder ---
    comparison_filenames = set()
    if comparison_folder is not None:
        if not os.path.isdir(comparison_folder):
            raise ValueError(f"comparison_folder does not exist: {comparison_folder}")
        comparison_filenames = {
            fname for fname in os.listdir(comparison_folder)
            if pattern.match(fname)
        }

    # --- Read source records ---
    records = []
    excluded_due_to_comparison = []

    for fname in os.listdir(source_folder):
        match = pattern.match(fname)
        if not match:
            continue

        if fname in comparison_filenames:
            excluded_due_to_comparison.append(fname)
            continue

        mach = float(match.group(1))
        z = float(match.group(2))
        records.append({
            "filename": fname,
            "mach": mach,
            "z": z,
        })

    if not records:
        raise ValueError(
            "No matching files available in source folder after filtering duplicates."
        )

    if n_select > len(records):
        raise ValueError(
            f"Requested {n_select} files, but only {len(records)} eligible files exist "
            f"after excluding duplicates from comparison folder."
        )

    mach_vals = np.array([r["mach"] for r in records], dtype=float)
    z_vals = np.array([r["z"] for r in records], dtype=float)

    mach_min, mach_max = mach_vals.min(), mach_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()

    def normalize(vals, vmin, vmax):
        if np.isclose(vmax, vmin):
            return np.zeros_like(vals)
        return (vals - vmin) / (vmax - vmin)

    real_points = np.column_stack([
        normalize(mach_vals, mach_min, mach_max),
        normalize(z_vals, z_min, z_max),
    ])

    sampler = qmc.LatinHypercube(d=2, seed=random_seed)
    lhs_points = sampler.random(n=n_select)

    available = set(range(len(records)))
    selected_indices = []

    for target in lhs_points:
        if not available:
            raise RuntimeError("Ran out of available files before completing selection.")

        avail_list = np.array(sorted(available))
        avail_points = real_points[avail_list]

        dists = np.linalg.norm(avail_points - target, axis=1)
        order = np.argsort(dists)

        chosen_global_idx = None
        for local_idx in order:
            candidate_global_idx = avail_list[local_idx]
            candidate_fname = records[candidate_global_idx]["filename"]

            # Extra safety check in case comparison folder changed during runtime
            if candidate_fname not in comparison_filenames:
                chosen_global_idx = candidate_global_idx
                break

        if chosen_global_idx is None:
            raise RuntimeError(
                "Could not find a valid non-duplicate file for one of the LHS samples."
            )

        selected_indices.append(chosen_global_idx)
        available.remove(chosen_global_idx)

    selected_files = sorted([records[i]["filename"] for i in selected_indices])

    parent_dir = os.path.dirname(os.path.abspath(source_folder))
    unseen_folder = os.path.join(parent_dir, unseen_folder_name)

    if dry_run:
        print(f"[DRY RUN] Would move {len(selected_files)} files to:")
        print(unseen_folder)
        print()

        if comparison_folder is not None:
            print(f"Checked duplicates against: {comparison_folder}")
            print(f"Excluded {len(excluded_due_to_comparison)} duplicate-name files.\n")

        for fname in selected_files:
            print(fname)

        return selected_files

    os.makedirs(unseen_folder, exist_ok=True)

    for fname in selected_files:
        src = os.path.join(source_folder, fname)
        dst = os.path.join(unseen_folder, fname)

        if os.path.exists(dst):
            raise FileExistsError(f"Destination file already exists: {dst}")

        shutil.move(src, dst)
        print(f"Moved: {fname}")

    print(f"\nMoved {len(selected_files)} files to: {unseen_folder}")

    if comparison_folder is not None:
        print(f"Checked duplicates against: {comparison_folder}")
        print(f"Excluded {len(excluded_due_to_comparison)} duplicate-name files.")

    return selected_files


if __name__ == "__main__":
    source_folder = r"Data/waverider_hf_data"
    comparison_folder = r"Data/waverider_hf_data_subset_200"   # change this as needed
    n_select = 50

    select_and_move_unseen_files(
        source_folder=source_folder,
        n_select=n_select,
        comparison_folder=comparison_folder,
        unseen_folder_name=f"waverider_unseen_{n_select}",
        dry_run=False,   # set True first to preview
        random_seed=42,
    )