import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_inference_errors(self, field_aggregates):
    # Prepare output directory
    plots_dir = os.path.join(self.project_root, "Outputs", self.project_name, "all_inference_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot scatter for each field and save ---
    averages = {}  # store averages for later in all-in-one plot
    colors = plt.cm.tab10.colors

    for idx, (field, list_of_errors) in enumerate(field_aggregates.items()):
        y = [float(e) for e in list_of_errors]
        x = np.arange(len(y))
        avg = np.mean(y)
        averages[field] = avg

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x, y, s=80,
            color=colors[idx % len(colors)],
            edgecolors='black',
            alpha=0.8,
            label=f"avg = {avg:.2f}%"
        )
        plt.xlabel("Inferences")
        plt.ylabel("Relative L2 Error (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=9, title_fontsize=10)

        safe_field = "".join(c if (c.isalnum() or c in "._-") else "_" for c in field)
        out_path = os.path.join(plots_dir, f"{safe_field}_l2_errors.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved scatter for '{field}' -> {out_path}")

    # --- All-in-one summary plot with field averages ---
    plt.figure(figsize=(10, 6))

    for idx, (field, list_of_errors) in enumerate(field_aggregates.items()):
        y = [float(e) for e in list_of_errors]
        x = np.arange(len(y))
        avg = averages[field]
        plt.scatter(
            x, y, s=70,
            color=colors[idx % len(colors)],
            edgecolors='black',
            alpha=0.8,
            label=f"{field} (avg = {avg:.2f}%)"
        )

    plt.xticks(x, x + 1)
    plt.xlabel("Inferences")
    plt.ylabel("Relative L2 Error (%)")
    plt.title("All Fields")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Fields and Averages", bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=9, title_fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "all_fields_l2_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved combined comparison plot with averages -> {out_path}")

def plot_loss_history(self, low_fidelity=False):
    plt.plot(self.loss_history, label='Training Loss')
    plt.plot(self.test_loss_history, label='Testing Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    if low_fidelity:
        self.loss_history_file_name = "loss_history_low_fidelity.png"
        plt.title("Low Fidelity Training and Testing Loss History")
    else:
        self.loss_history_file_name = "loss_history.png"
        plt.title("Training and Testing Loss History")
    plt.legend()
    plt.grid(True)
    fig_dir = os.path.join("Outputs",f"{self.project_name}")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, self.loss_history_file_name))
    plt.close()