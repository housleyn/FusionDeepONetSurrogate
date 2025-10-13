import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_raw_field(csv_path, field_name, distance_column='distanceToEllipse', 
                   mask_threshold=0.0, figsize=(15, 5)):
    """
    Plot raw field values as scatter points with optional masking
    
    Parameters:
    - csv_path: path to CSV file
    - field_name: name of field to plot (e.g., 'Density (kg/m^3)')
    - distance_column: name of distance column
    - mask_threshold: threshold for masking (points with distance <= threshold are masked)
    - figsize: figure size
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract coordinates
    x = df['X (m)'].values
    y = df['Y (m)'].values
    
    # Extract field and distance data
    field_values = df[field_name].values
    distance_values = df[distance_column].values
    
    # Create mask
    mask = distance_values <= mask_threshold
    
    # Split data into inside/outside points
    x_outside = x[~mask]
    y_outside = y[~mask]
    field_outside = field_values[~mask]
    
    x_inside = x[mask]
    y_inside = y[mask]
    field_inside = field_values[mask]
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: All points
    scatter1 = ax1.scatter(x, y, c=field_values, cmap='viridis', s=1)
    ax1.set_title(f'{field_name}\n(All Points)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot 2: Distance values showing inside/outside
    scatter2 = ax2.scatter(x, y, c=distance_values, cmap='RdYlBu', s=1)
    # Highlight the boundary points
    boundary_points = np.abs(distance_values) < 0.001  # Points very close to boundary
    if np.any(boundary_points):
        ax2.scatter(x[boundary_points], y[boundary_points], c='red', s=3, marker='x')
    ax2.set_title(f'{distance_column}\n(Red X = boundary)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot 3: Only outside points (masked)
    if len(x_outside) > 0:
        scatter3 = ax3.scatter(x_outside, y_outside, c=field_outside, cmap='viridis', s=1)
        ax3.set_title(f'{field_name}\n(Outside Only - Masked)')
        plt.colorbar(scatter3, ax=ax3)
    else:
        ax3.set_title(f'{field_name}\n(No outside points)')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Total data points: {len(x)}")
    print(f"Field range: {field_values.min():.2e} to {field_values.max():.2e}")
    print(f"Distance range: {distance_values.min():.2e} to {distance_values.max():.2e}")
    print(f"Points inside (distance <= {mask_threshold}): {np.sum(mask)} ({100*np.sum(mask)/len(mask):.1f}%)")
    print(f"Points outside (distance > {mask_threshold}): {np.sum(~mask)} ({100*np.sum(~mask)/len(mask):.1f}%)")

if __name__ == "__main__":
    csv_file = "Data/ellipse_data/ellipse_data_test1.csv"
    field = "Density (kg/m^3)"
    
    plot_raw_field(csv_file, field, distance_column='distanceToEllipse', 
                   mask_threshold=0.0, figsize=(15, 5))