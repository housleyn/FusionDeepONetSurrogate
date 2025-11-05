from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import matplotlib.tri as tri
import matplotlib.colors as mcolors

class MethodsPostprocess:
    def run(self, dimension):
        self._calculate_error()
        self._define_ouput_folders()
        fields = self.fields
        if dimension == 3:
            fields.insert(2, "Velocity[k] (m/s)")

        for field in fields:
            self._calculate_relative_l2_error(field)

        self._create_table()

        self.plot_fields()
    
    def _calculate_error(self):
        error = np.abs(self.df_true - self.df_pred)
        error["X (m)"] = self.df_true["X (m)"]
        error["Y (m)"] = self.df_true["Y (m)"]
        self.error = error

    def _calculate_relative_l2_error(self, field):
        u_true = self.df_true[field].values
        u_pred = self.df_pred[field].values
        rel_l2 = 100 * np.linalg.norm(u_true-u_pred) / np.linalg.norm(u_true)
        self.errors[field] = rel_l2

    def _define_ouput_folders(self):
        self.figures_dir = os.path.join("Outputs", self.project_name)
        self.tables_dir = os.path.join("Outputs", self.project_name)
        os.makedirs(self.figures_dir, exist_ok=True)

    def _create_table(self):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis('off')
        table_data = [(k, f"{v:.2f}%") for k, v in self.errors.items()]
        table = ax.table(cellText=table_data,
                         colLabels=["Field", "Relative L2 Norm % Error"],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        os.makedirs(self.tables_dir, exist_ok=True)
        plt.savefig(os.path.join(self.tables_dir, "relative_l2_errors.png"), bbox_inches='tight')
        plt.close()

    def plot_fields(self):

        x_col = self.df_true["X (m)"].values
        y_col = self.df_true["Y (m)"].values
        if "distanceToSurface" in self.df_true.columns:
            dist_col = self.df_true["distanceToSurface"].values
        elif "distanceToEllipse" in self.df_true.columns:
            dist_col = self.df_true["distanceToEllipse"].values
        else:
            raise ValueError("Neither 'distanceToSurface' nor 'distanceToEllipse' column found in the dataset")

        x, y = x_col, y_col
        dist = dist_col
        triang = tri.Triangulation(x,y)

        xtri, ytri = x[triang.triangles], y[triang.triangles]
        edge_lengths = np.sqrt((xtri[:, None, :] - xtri[:, :, None])**2 +
                       (ytri[:, None, :] - ytri[:, :, None])**2)
        
        max_edge = np.max(edge_lengths, axis=(1,2))
        dist_tri = dist[triang.triangles]
        dist_centroid = dist_tri.mean(axis=1)

        edge_threshold = np.percentile(max_edge, self.edge_percentile)
        dist_threshold = self.dist_threshold

        mask = (max_edge > edge_threshold) & (dist_centroid < dist_threshold)
        triang.set_mask(mask)

        for field in self.fields:
            zi_true = self.df_true[field].values
            zi_pred = self.df_pred[field].values
            zi_error = self.error[field].values

            # shared normalization for predicted and true
            vmin_tp = np.nanmin([zi_true, zi_pred])
            vmax_tp = np.nanmax([zi_true, zi_pred])
            norm_tp = mcolors.Normalize(vmin=vmin_tp, vmax=vmax_tp)

            # normalization for error
            vmin_err = np.nanmin(zi_error)
            vmax_err = np.nanmax(zi_error)
            norm_err = mcolors.Normalize(vmin=vmin_err, vmax=vmax_err)
            
            
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            titles = ["Predicted", "True", "Error"]
            datasets = [zi_pred, zi_true, zi_error]
            cmaps = ["inferno"] * 3
            norms = [norm_tp, norm_tp, norm_err]
            levels = np.linspace(vmin_tp, vmax_tp, 100)
            

            for ax, data, title, cmap, norm in zip(axs, datasets, titles, cmaps, norms):
                if title == "Error":
                    levels = 100
                contour = ax.tricontourf(triang, data, levels=levels, cmap=cmap, norm=norm)
                cbar = fig.colorbar(contour, ax=ax)

                locator = MaxNLocator(nbins=8, prune=None)
                cbar.locator = locator
                cbar.update_ticks()

                cbar.set_label(
                    f"{field} Color Scale" if title != "Error" else "Error Color Scale"
                )
                cbar.ax.tick_params(labelsize=12)
                ax.set_title(f"{field} - {title}")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_xlim(self.x_lim)
                ax.set_ylim(self.y_lim)

            plt.tight_layout()
            safe_field = (
                field.replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            error_dir = os.path.join(self.figures_dir, "error_figures")
            os.makedirs(error_dir, exist_ok=True)
            plt.savefig(os.path.join(error_dir, f"{safe_field}_comparison.png"))
            plt.close()
            print(f"Saved comparison plot for {field} to {error_dir}/{safe_field}_comparison.png")

            # # === Histogram Plot ===
            fig, ax = plt.subplots(figsize=(8, 6))
            # vals = z_denorm
            

            # # Use log scale for better spread visualization if needed
            
            ax.hist(zi_error, bins=100, color='darkorange', alpha=0.85, edgecolor='black')
            
            ax.set_yscale('log')
            ax.set_xlabel(f"{field}")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"{field} Error Distribution")
            ax.grid(True, which="both", ls="--", alpha=0.5)

            hist_dir = os.path.join(self.figures_dir,"error_figures", "histograms")
            os.makedirs(hist_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{safe_field}_error_histogram.png"))
            plt.close()
            print(f"Saved residual histogram for {field} to {hist_dir}/{safe_field}_error_histogram.png")

        if self.model_type == "low_fi_fusion":
                self._plot_residual_analysis(mask)

    def _plot_residual_analysis(self, mask):
        residual_path = os.path.join("Outputs", self.project_name, "residual.npz")
        residual_data = np.load(residual_path, allow_pickle=True)

        outputs       = residual_data["outputs"]        # (num_sims, num_points, num_fields)
        outputs_mean  = residual_data["outputs_mean"]   # (num_fields,)
        outputs_std   = residual_data["outputs_std"]    # (num_fields,)
        params_all    = residual_data["params"]         # (num_sims, num_params)
        coords_all    = residual_data["coords"]         # (num_sims, num_points, 3)
        fields        = self.fields

        # Identify which simulation matches this case
        true_params = self.df_true[self.param_columns].iloc[0].to_numpy()
        idx = np.where(np.all(np.isclose(params_all, true_params, atol=1e-6), axis=1))[0]
        if len(idx) == 0:
            raise ValueError(f"No simulation found in residuals for parameters {true_params}")
        sim_idx = idx[0]

        # Extract coords and outputs
        coords = coords_all[sim_idx]
        x, y = coords[:, 0], coords[:, 1]
        triang = tri.Triangulation(x, y)

        # Recompute mask using same logic from your plot_fields() method
        xtri, ytri = x[triang.triangles], y[triang.triangles]
        edge_lengths = np.sqrt((xtri[:, None, :] - xtri[:, :, None])**2 +
                            (ytri[:, None, :] - ytri[:, :, None])**2)
        max_edge = np.max(edge_lengths, axis=(1, 2))

        # Optional: compute distance if your data includes it
        if "distanceToSurface" in self.df_true.columns:
            dist = self.df_true["distanceToSurface"].values
        elif "distanceToEllipse" in self.df_true.columns:
            dist = self.df_true["distanceToEllipse"].values
        else:
            dist = np.zeros_like(x)

        dist_tri = dist[triang.triangles]
        dist_centroid = dist_tri.mean(axis=1)

        edge_threshold = np.percentile(max_edge, self.edge_percentile)
        dist_threshold = self.dist_threshold
        mask_new = (max_edge > edge_threshold) & (dist_centroid < dist_threshold)
        triang.set_mask(mask_new)

        outputs_sim = outputs[sim_idx]

        # Loop through each field
        for i, field in enumerate(fields):
            z_raw = outputs_sim[:, i]
            z_denorm = z_raw * outputs_std[i] + outputs_mean[i]
            z_abs = np.abs(z_denorm)

            # Independent normalizations
            norm_raw = mcolors.Normalize(vmin=np.nanmin(z_raw), vmax=np.nanmax(z_raw))
            norm_denorm = mcolors.Normalize(vmin=np.nanmin(z_denorm), vmax=np.nanmax(z_denorm))
            norm_abs = mcolors.Normalize(vmin=np.nanmin(z_abs), vmax=np.nanmax(z_abs))

            # Create contour plots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            titles = ["Raw Outputs", "Denormalized Outputs", "Absolute Outputs"]
            datasets = [z_raw, z_denorm, z_abs]
            cmaps = ["viridis", "viridis", "inferno"]
            norms = [norm_raw, norm_denorm, norm_abs]

            for ax, data, title, cmap, norm in zip(axs, datasets, titles, cmaps, norms):
                contour = ax.tricontourf(triang, data, levels=100, cmap=cmap, norm=norm)
                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label(f"{field} {title} Color Scale")
                cbar.ax.tick_params(labelsize=12)
                ax.set_title(f"{field} - {title}")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_xlim(self.x_lim)
                ax.set_ylim(self.y_lim)

            plt.tight_layout()
            safe_field = (
                field.replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            out_dir = os.path.join(self.figures_dir, "residual_analysis")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{safe_field}_residual_analysis.png"))
            plt.close()
            print(f"Saved residual analysis for {field} to {out_dir}/{safe_field}_residual_analysis.png")

            # === Histogram Plot ===
            fig, ax = plt.subplots(figsize=(8, 6))
            vals = z_denorm
            # abs_vals = abs_vals[~np.isnan(abs_vals)]  # remove NaNs

            # Use log scale for better spread visualization if needed
            # bins = np.logspace(np.log10(max(vals.min(), 1e-6)), np.log10(vals.max()), 60)
            ax.hist(vals, bins=100, color='darkorange', alpha=0.85, edgecolor='black')
            # ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f"{field} (Denormalized Residual)")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"{field} – Residual Magnitude Distribution")
            ax.grid(True, which="both", ls="--", alpha=0.5)

            hist_dir = os.path.join(self.figures_dir, "residual_analysis", "histograms")
            os.makedirs(hist_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{safe_field}_residual_histogram.png"))
            plt.close()
            print(f"Saved residual histogram for {field} to {hist_dir}/{safe_field}_residual_histogram.png")

        residual_data.close()




    

    

