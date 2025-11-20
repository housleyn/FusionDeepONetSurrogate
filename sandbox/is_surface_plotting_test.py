import os
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(script_dir, "is_surface_test.csv")

def find_is_surface_column(df):
    # prefer exact name, handle common misspelling
    for name in ("is_surface", "is_surace", "isSurface"):
        if name in df.columns:
            return name
    # fallback: any column with 'surface' in its name
    for c in df.columns:
        if "surface" in c.lower():
            return c
    return None

def find_area_vector_columns(df):
    """Find area vector components in the dataframe"""
    # Look for common area vector naming patterns
    patterns = [
        ("Area Vector[i] (m^2)", "Area Vector[j] (m^2)", "Area Vector[k] (m^2)"),
        ("AreaVector[i]", "AreaVector[j]", "AreaVector[k]"),
        ("area_x", "area_y", "area_z"),
        ("Ax", "Ay", "Az"),
        ("Area_X", "Area_Y", "Area_Z")
    ]
    
    # Check for exact matches
    for ax_col, ay_col, az_col in patterns:
        if all(col in df.columns for col in [ax_col, ay_col, az_col]):
            return ax_col, ay_col, az_col
    
    # Look for columns containing area vector keywords
    area_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['area', 'vector'])]
    
    if len(area_cols) >= 3:
        # Try to identify x, y, z components
        x_comp = next((c for c in area_cols if any(x in c.lower() for x in ['[i]', 'x', '_i'])), None)
        y_comp = next((c for c in area_cols if any(y in c.lower() for y in ['[j]', 'y', '_j'])), None)
        z_comp = next((c for c in area_cols if any(z in c.lower() for z in ['[k]', 'z', '_k'])), None)
        
        if x_comp and y_comp and z_comp:
            return x_comp, y_comp, z_comp
    
    return None, None, None

def choose_xy_columns(df, iscol):
    # Look specifically for coordinate columns first
    coord_patterns = [
        ("X (m)", "Y (m)"),  # Your typical coordinate format
        ("x", "y"),
        ("X", "Y"),
        ("lon", "lat"),
        ("longitude", "latitude")
    ]
    
    # Check for exact matches first
    for x_col, y_col in coord_patterns:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    
    # Check for columns containing coordinate keywords
    x_candidates = [c for c in df.columns if any(keyword in c.lower() for keyword in ['x', 'longitude', 'lon'])]
    y_candidates = [c for c in df.columns if any(keyword in c.lower() for keyword in ['y', 'latitude', 'lat'])]
    
    # Filter out velocity and other non-coordinate columns
    exclude_keywords = ['velocity', 'vel', 'speed', 'pressure', 'temperature', 'density', 'mach', 'area', 'vector']
    x_candidates = [c for c in x_candidates if not any(exc in c.lower() for exc in exclude_keywords)]
    y_candidates = [c for c in y_candidates if not any(exc in c.lower() for exc in exclude_keywords)]
    
    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]
    
    # Last resort: use first two numeric columns excluding is_surface
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != iscol]
    # Also exclude velocity and other flow field columns
    numeric = [c for c in numeric if not any(exc in c.lower() for exc in exclude_keywords)]
    
    if len(numeric) >= 2:
        return numeric[0], numeric[1]
    
    return None, None

def plot_area_vectors(df, xcol, ycol, ax_col, ay_col, az_col, iscol):
    """Plot area vectors as arrows on the coordinate system - SURFACE ONLY"""
    # Get coordinates and area vector components
    x = df[xcol].values
    y = df[ycol].values
    ax_vals = df[ax_col].values
    ay_vals = df[ay_col].values
    az_vals = df[az_col].values if az_col else np.zeros_like(ax_vals)
    
    # Get is_surface values for filtering
    is_vals = pd.to_numeric(df[iscol], errors="coerce").fillna(0).astype(int)
    
    # Calculate vector magnitudes
    vector_mag = np.sqrt(ax_vals**2 + ay_vals**2 + az_vals**2)
    
    # FILTER 1: Only surface points (is_surface == 1)
    surface_mask = (is_vals == 1)
    print(f"Total points: {len(df)}")
    print(f"Surface points: {np.sum(surface_mask)}")
    
    if not np.any(surface_mask):
        print("⚠️  No surface points found (is_surface == 1)")
        return None
    
    # Apply surface filter
    x_surf = x[surface_mask]
    y_surf = y[surface_mask]
    ax_surf = ax_vals[surface_mask]
    ay_surf = ay_vals[surface_mask]
    az_surf = az_vals[surface_mask]
    mag_surf = vector_mag[surface_mask]
    
    # FILTER 2: Check for ridiculously high values
    mag_stats = {
        'mean': np.mean(mag_surf),
        'std': np.std(mag_surf),
        'max': np.max(mag_surf),
        'min': np.min(mag_surf[mag_surf > 0])  # Exclude zeros
    }
    
    # Define "ridiculous" thresholds
    mean_threshold = 100 * mag_stats['mean']  # 100x the mean
    std_threshold = mag_stats['mean'] + 10 * mag_stats['std']  # Mean + 10 std devs
    absolute_threshold = 1e6  # Absolute large number threshold
    
    # Check for outliers
    high_values_mask = (mag_surf > mean_threshold) | (mag_surf > std_threshold) | (mag_surf > absolute_threshold)
    
    if np.any(high_values_mask):
        high_count = np.sum(high_values_mask)
        high_max = np.max(mag_surf[high_values_mask])
        print(f"🚨 ERROR: Found {high_count} ridiculously high area vector values!")
        print(f"   Max ridiculous value: {high_max:.2e}")
        print(f"   Mean value: {mag_stats['mean']:.2e}")
        print(f"   Std dev: {mag_stats['std']:.2e}")
        print(f"   Thresholds exceeded:")
        print(f"     - Mean threshold (100x): {mean_threshold:.2e}")
        print(f"     - Std threshold (mean + 10σ): {std_threshold:.2e}")
        print(f"     - Absolute threshold: {absolute_threshold:.2e}")
        
        # Option: Remove ridiculous values or exit
        choice = input("Continue by filtering out high values? (y/n): ")
        if choice.lower() != 'y':
            return None
        else:
            # Filter out the ridiculous values
            good_mask = ~high_values_mask
            x_surf = x_surf[good_mask]
            y_surf = y_surf[good_mask]
            ax_surf = ax_surf[good_mask]
            ay_surf = ay_surf[good_mask]
            az_surf = az_surf[good_mask]
            mag_surf = mag_surf[good_mask]
            print(f"   Filtered to {len(x_surf)} reasonable surface points")
    
    # FILTER 3: Remove zero vectors
    non_zero_mask = mag_surf > 1e-10
    x_surf = x_surf[non_zero_mask]
    y_surf = y_surf[non_zero_mask]
    ax_surf = ax_surf[non_zero_mask]
    ay_surf = ay_surf[non_zero_mask]
    az_surf = az_surf[non_zero_mask]
    mag_surf = mag_surf[non_zero_mask]
    
    print(f"Final surface vectors to plot: {len(x_surf)}")
    
    if len(x_surf) == 0:
        print("⚠️  No valid surface vectors to plot after filtering")
        return None
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Area vectors as arrows (SURFACE ONLY)
    # Scale arrows for visibility
    coord_range = max(np.ptp(x_surf), np.ptp(y_surf))
    if coord_range > 0 and np.max(mag_surf) > 0:
        arrow_scale = 0.1 * coord_range / np.mean(mag_surf)
    else:
        arrow_scale = 1.0
    
    # Plot surface area vectors only
    ax1.quiver(x_surf, y_surf, ax_surf, ay_surf,
              scale=1/arrow_scale, scale_units='xy', angles='xy',
              color='red', alpha=0.8, label=f'Surface Normals (n={len(x_surf)})', 
              width=0.004)
    
    # Also plot the surface points as background
    ax1.scatter(x_surf, y_surf, s=30, c='red', alpha=0.3, 
                label='Surface Points', zorder=1)
    
    ax1.set_xlabel(xcol)
    ax1.set_ylabel(ycol)
    ax1.set_title('Surface Area Vectors Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Vector magnitude distribution (SURFACE ONLY)
    ax2.hist(mag_surf, bins=min(50, len(mag_surf)//2), alpha=0.7, 
             color='red', edgecolor='black')
    ax2.set_xlabel('Area Vector Magnitude (Surface Only)')
    ax2.set_ylabel('Count')
    ax2.set_title('Surface Area Vector Magnitude Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Surface Points: {len(x_surf)}\n'
    stats_text += f'Mean: {np.mean(mag_surf):.2e}\n'
    stats_text += f'Max: {np.max(mag_surf):.2e}\n'
    stats_text += f'Min: {np.min(mag_surf):.2e}\n'
    stats_text += f'Std: {np.std(mag_surf):.2e}'
    ax2.text(0.65, 0.7, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    out_png = "surface_area_vectors_plot.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved surface area vectors plot to {out_png}")
    
    return fig

def main():
    if not os.path.exists(CSV_FILE):
        print(f"CSV file not found: {CSV_FILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    iscol = find_is_surface_column(df)
    if iscol is None:
        print("Could not find an 'is_surface' column in the CSV.", file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    # Ensure is_surface is numeric (0/1)
    try:
        is_vals = pd.to_numeric(df[iscol], errors="coerce").fillna(0).astype(int)
    except Exception:
        is_vals = df[iscol].astype(int)

    # decide plotting mode
    xcol, ycol = choose_xy_columns(df, iscol)

    # Original is_surface plot
    plt.figure(figsize=(8, 6))
    base_size = 20
    large_size = 140
    sizes = np.where(is_vals == 1, large_size, base_size)
    colors = np.where(is_vals == 1, "C1", "C0")

    if xcol and ycol:
        plt.scatter(df[xcol], df[ycol], s=sizes, c=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        title_extra = f" ({xcol} vs {ycol})"
    else:
        # fallback: plot index vs is_surface and scale markers
        plt.scatter(df.index, is_vals, s=sizes, c=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
        plt.xlabel("index")
        plt.ylabel(iscol)
        title_extra = " (index vs is_surface)"

    plt.title(f"is_surface plot{title_extra}")
    # legend
    legend_elements = [
        Line2D([12], [13], marker="o", color="w", label="is_surface = 1", markerfacecolor="C1", markersize=8),
        Line2D([2], [13], marker="o", color="w", label="is_surface = 0", markerfacecolor="C0", markersize=6),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.tight_layout()

    out_png = "is_surface_plot.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")
    plt.show()

    # Area vectors plot
    ax_col, ay_col, az_col = find_area_vector_columns(df)
    if ax_col and ay_col and xcol and ycol:
        print(f"Found area vector columns: {ax_col}, {ay_col}, {az_col}")
        plot_area_vectors(df, xcol, ycol, ax_col, ay_col, az_col, iscol)
        plt.show()
    else:
        print("Could not find area vector columns in the CSV.")
        print("Looking for columns with 'area' and 'vector' keywords...")
        area_related = [c for c in df.columns if any(keyword in c.lower() for keyword in ['area', 'vector'])]
        if area_related:
            print("Found area/vector related columns:", area_related)
        else:
            print("No area vector columns found.")


if __name__ == "__main__":
    main()