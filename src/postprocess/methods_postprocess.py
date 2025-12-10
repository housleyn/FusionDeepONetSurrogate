from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.tri as tri
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import pandas as pd

class MethodsPostprocess:
    def run(self, dimension):
        self._calculate_error()
        self._define_output_folders()
        fields = self.fields
        if dimension == 3:
            fields.insert(2, "Velocity[k] (m/s)")

        for field in fields:
            self._calculate_relative_l2_error(field)

        self._create_table()

        self.plot_fields()
        surface_metrics = self.compute_surface_percent_differences()

        self._create_surface_metrics_table(
            metrics=surface_metrics,
            save_path=os.path.join(self.tables_dir, "surface_metric_errors.png")
        )


    def get_errors(self):
        self._calculate_error()
        fields = self.fields 
        for field in fields:
            self._calculate_relative_l2_error(field)
        return self.errors.items()
    
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

    def _define_output_folders(self):
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
        iscol = "is_on_surface"
        ax_col, ay_col, az_col = "Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"

        for col in (ax_col, ay_col, az_col):
            vals = self.df_true[col].values
            bad = np.abs(vals) > 1e6
            self.df_true.loc[bad, col] = np.nan

        x, y = x_col, y_col
        is_surface = (pd.to_numeric(self.df_true[iscol], errors="coerce").fillna(0).astype(int).to_numpy().astype(bool))
        triangles = tri.Triangulation(x,y)
        surf_tri_mask = is_surface[triangles.triangles].any(axis=1)

        

        

        for field in self.fields:
            zi_true = self.df_true[field].values
            zi_pred = self.df_pred[field].values
            zi_error = self.error[field].values

            node_invalid = ~np.isfinite(zi_true)
            field_tri_mask = node_invalid[triangles.triangles].any(axis=1)

            combined_mask = surf_tri_mask | field_tri_mask
            triangles.set_mask(combined_mask)
            

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
                contour = ax.tricontourf(triangles, data, levels=levels, cmap=cmap, norm=norm)
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

            # Percent error histogram
            zi_percent_error = np.abs((zi_error) / (zi_true + 1e-8)) * 100
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(zi_percent_error, bins=200, color='steelblue', alpha=0.8, edgecolor='black')
            ax.set_yscale('log')
            ax.set_xlabel(f"{field} Percent Error (%)")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"{field} Percent Error Distribution")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{safe_field}_percent_error_histogram.png"))
            plt.close()

        if self.model_type == "low_fi_fusion":
                self._plot_residual_analysis()

    def _plot_residual_analysis(self):
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
            # optional debug print:
            print(f"[Residual analysis] No simulation found in residuals for parameters {true_params}")
            residual_data.close()  
            return  
        sim_idx = idx[0]

        iscol = "is_on_surface"
        ax_col, ay_col, az_col = "Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"

        for col in (ax_col, ay_col, az_col):
            vals = self.df_true[col].values
            bad = np.abs(vals) > 1e6
            self.df_true.loc[bad, col] = np.nan

        coords = coords_all[sim_idx]
        x_res = coords[:, 0]
        y_res = coords[:, 1]
        outputs_sim = outputs[sim_idx]


        # build is_surface_res from df_true
        x_true = self.df_true["X (m)"].values
        y_true = self.df_true["Y (m)"].values
        is_surface_true = self.df_true["is_on_surface"].astype(bool).values

        true_xy = np.column_stack([x_true, y_true])
        res_xy  = np.column_stack([x_res, y_res])

        tree = cKDTree(true_xy)
        dist, nn_idx = tree.query(res_xy, k=1)
        is_surface_res = is_surface_true[nn_idx]

        triang = tri.Triangulation(x_res, y_res)
        base_geom_mask = is_surface_res[triang.triangles].any(axis=1)

        # Loop through each field
        for i, field in enumerate(fields):
            z_raw = outputs_sim[:, i]
            z_denorm = z_raw * outputs_std[i] + outputs_mean[i]
            z_abs = np.abs(z_denorm)

            node_invalid = ~(np.isfinite(z_raw) & np.isfinite(z_denorm) & np.isfinite(z_abs))
            field_tri_mask = node_invalid[triang.triangles].any(axis=1)
            combined_mask = base_geom_mask | field_tri_mask
            triang.set_mask(combined_mask)

            # Independent normalizations
            norm_raw = mcolors.Normalize(vmin=np.nanmin(z_raw), vmax=np.nanmax(z_raw))
            norm_denorm = mcolors.Normalize(vmin=np.nanmin(z_denorm), vmax=np.nanmax(z_denorm))
            norm_abs = mcolors.Normalize(vmin=np.nanmin(z_abs), vmax=np.nanmax(z_abs))

            # Create contour plots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            titles = ["Raw Outputs", "Denormalized Outputs", "Absolute Outputs"]
            datasets = [z_raw, z_denorm, z_abs]
            cmaps = ["inferno", "inferno", "inferno"]
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

    def compute_surface_percent_differences(
        self,
        area_threshold=1e100,
        coef_eps=1e-12,
    ):
        """
        Compute percent difference in CD, CL, and wall temperature
        (using area-weighted temperature proxy).
        
        - Surface cells: is_on_surface == 1 AND area normal < threshold
        - CD/CL: pressure * area vector integrations
        - Heat: sum(T * |A|) comparison
        """

        # --- Basic checks ---
        if self.df_true is None:
            raise ValueError("df_true is None; surface metrics require true data.")
        if self.df_pred is None:
            raise ValueError("df_pred is None; surface metrics require predicted data.")

        required_cols_true = [
            "Absolute Pressure (Pa)",
            "Temperature (K)",
            "X (m)", "Y (m)", "Z (m)",
            "is_on_surface",
            "Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)",
        ]
        for col in required_cols_true:
            if col not in self.df_true.columns:
                raise ValueError(f"True data missing required column: {col}")

        required_cols_pred = [
            "Absolute Pressure (Pa)",
            "Temperature (K)",
        ]
        for col in required_cols_pred:
            if col not in self.df_pred.columns:
                raise ValueError(f"Predicted data missing column: {col}")

        # --- Surface mask from normals ---
        area_vec_all = self.df_true[
            ["Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"]
        ].to_numpy()
        area_norm_all = np.linalg.norm(area_vec_all, axis=1)

        is_surface = (self.df_true["is_on_surface"] == 1)
        valid_area = (
            np.isfinite(area_norm_all)
            & (area_norm_all > 0.0)
            & (area_norm_all < area_threshold)
        )
        surface_mask = is_surface & valid_area

        if not np.any(surface_mask):
            raise ValueError("No valid surface cells found.")

        # --- Extract surface values ---
        p_true_s = self.df_true["Absolute Pressure (Pa)"].to_numpy()[surface_mask]
        p_pred_s = self.df_pred["Absolute Pressure (Pa)"].to_numpy()[surface_mask]

        T_true_s = self.df_true["Temperature (K)"].to_numpy()[surface_mask]
        T_pred_s = self.df_pred["Temperature (K)"].to_numpy()[surface_mask]

        A_vec_s = area_vec_all[surface_mask]
        A_norm_s = area_norm_all[surface_mask]

        # --- Directions for Cd/Cl ---
        if "AoA" in self.df_true.columns:
            aoa_deg = float(self.df_true["AoA"].iloc[0])
        else:
            aoa_deg = 0.0

        aoa_rad = np.deg2rad(aoa_deg)
        e_inf = np.array([np.cos(aoa_rad), 0, np.sin(aoa_rad)])
        e_inf /= (np.linalg.norm(e_inf) + 1e-15)

        e_drag = e_inf
        e_lift = np.array([-np.sin(aoa_rad), 0, np.cos(aoa_rad)])
        e_lift /= (np.linalg.norm(e_lift) + 1e-15)

        # --- Integrate forces ---
        F_true = -np.sum(p_true_s[:, None] * A_vec_s, axis=0)
        F_pred = -np.sum(p_pred_s[:, None] * A_vec_s, axis=0)

        Cd_true = float(np.dot(F_true, e_drag))
        Cd_pred = float(np.dot(F_pred, e_drag))
        Cl_true = float(np.dot(F_true, e_lift))
        Cl_pred = float(np.dot(F_pred, e_lift))

        def percent_diff(pred, true):
            return float(np.abs(pred - true) / (np.abs(true) + coef_eps) * 100)

        Cd_percent = percent_diff(Cd_pred, Cd_true)
        Cl_percent = percent_diff(Cl_pred, Cl_true)

        # --- Wall Temperature Proxy (NO gradients) ---
        Q_true = float(np.sum(T_true_s * A_norm_s))
        Q_pred = float(np.sum(T_pred_s * A_norm_s))
        Heat_percent = percent_diff(Q_pred, Q_true)

        return {
            "Cd_percent_diff": Cd_percent,
            "Cl_percent_diff": Cl_percent,
            "Heat_percent_diff": Heat_percent,
            "Cd_true_proxy": Cd_true,
            "Cd_pred_proxy": Cd_pred,
            "Cl_true_proxy": Cl_true,
            "Cl_pred_proxy": Cl_pred,
            "Q_true_proxy": Q_true,
            "Q_pred_proxy": Q_pred,
            "num_surface_cells": int(surface_mask.sum()),
        }

    
    def _create_surface_metrics_table(self, metrics: dict, save_path: str):
        """
        Table image reporting Cd, Cl, and Wall Temperature % errors.
        """

        import matplotlib.pyplot as plt

        Cd_percent = metrics.get("Cd_percent_diff", None)
        Cl_percent = metrics.get("Cl_percent_diff", None)
        Heat_percent = metrics.get("Heat_percent_diff", None)

        table_data = [
            ["Metric", "Percent Error (%)"],
            ["Cd Error", f"{Cd_percent:.2f}"],
            ["Cl Error", f"{Cl_percent:.2f}"],
            ["Wall Temperature Error", f"{Heat_percent:.2f}"],
        ]

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")

        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        

        table.scale(1, 2)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)





    

    

