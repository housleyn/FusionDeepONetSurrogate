import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def relative_rmse_per_field(csv_true, csv_pred, columns):
    df_true = pd.read_csv(csv_true)
    df_pred = pd.read_csv(csv_pred)

    assert np.allclose(
        df_true[["X (m)", "Y (m)", "Z (m)"]].values,
        df_pred[["X (m)", "Y (m)", "Z (m)"]].values
    ), "Coordinate mismatch"

    data = []
    for col in columns:
        error = df_pred[col].values - df_true[col].values
        rel_rmse = np.sqrt(np.sum(error**2) / np.sum(df_true[col].values**2))
        data.append([col, f"{rel_rmse * 100:.2f}%"])

    return data

def plot_table(data):
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(
        cellText=data,
        colLabels=["Field", "% Error"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig("rmse_table.png", dpi=300)
    plt.show()

# === Usage ===
columns_to_compare = [
    "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
    "Absolute Pressure (Pa)", "Density (kg/m^3)"
]

results = relative_rmse_per_field("sphere_data_075.csv", "predicted_output.csv", columns_to_compare)
plot_table(results)
