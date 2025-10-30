import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

class MethodsPostprocess:
    def run(self, dimension, shape):
        error = self._calculate_error()
        self._define_ouput_folders()
        fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)",
                "Absolute Pressure (Pa)", "Density (kg/m^3)", "Temperature (K)"]
        if dimension == 3:
            fields.insert(2, "Velocity[k] (m/s)")

        for field in fields:
            self._calculate_relative_l2_error(field)

        self._create_table()

        # Fix the shape selection logic
        if shape == "ellipse":
            plot_func = self._plot_fields
        elif shape == "spheres": 
            plot_func = self._plot_fields_spheres
        else:  # shape == "None" or any other value
            # Use point cloud plotting (no interpolation, no masking)
            plot_func = lambda field, error: self.plot_point_cloud(field, error)
        
        for field in fields:
            plot_func(field, error)

    def _calculate_error(self):
        error = np.abs(self.df_true - self.df_pred)
        error["X (m)"] = self.df_true["X (m)"]
        error["Y (m)"] = self.df_true["Y (m)"]
        return error

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

    def _plot_fields(self, field, error):
        self._plot_generic(field, error, mask_type="ellipse")

    def _plot_fields_spheres(self, field, error):
        self._plot_generic(field, error, mask_type="spheres")

    def _plot_generic(self, field, error, mask_type):
        x = self.df_true["X (m)"].values
        y = self.df_true["Y (m)"].values
        xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)) #fix put the actual coordinates in here

        zi_true = griddata((x, y), self.df_true[field].values, (xi, yi), method='cubic')
        zi_pred = griddata((x, y), self.df_pred[field].values, (xi, yi), method='cubic')
        zi_error = np.abs(griddata((x, y), error[field].values, (xi, yi), method='cubic'))

        if mask_type == "ellipse":
            a = self.df_true["a"].values[0]
            b = self.df_true["b"].values[0]
            x0, y0 = -2.5, 0
            mask = ((xi - x0)**2 / a**2 + (yi - y0)**2 / b**2) <= 1
        elif mask_type == "spheres":
            x1, y1 = 0, 0
            x2, y2 = self.df_true["x"].values[0], self.df_true["y"].values[0]
            radius = 1.0
            mask = ((xi - x1)**2 + (yi - y1)**2 <= radius**2) | ((xi - x2)**2 + (yi - y2)**2 <= radius**2)
        else:
            self.plot_point_cloud(field, error)
            return

        for z in [zi_true, zi_pred, zi_error]:
            z[mask] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["Predicted", "True", "Error"]
        datasets = [zi_pred, zi_true, zi_error]
        cmaps = ["inferno"] * 3

        vmin_tp = np.nanmin([zi_true])
        vmax_tp = np.nanmax([zi_true])
        ticks_tp = np.linspace(vmin_tp, vmax_tp, 9)
        tick_labels_tp = [f"{t:.3f}" for t in ticks_tp]

        vmin_err = np.nanmin(zi_error)
        vmax_err = np.nanmax(zi_error)
        ticks_err = np.linspace(vmin_err, vmax_err, 9)
        tick_labels_err = [f"{t:.3f}" for t in ticks_err]

        for ax, data, title, cmap in zip(axs, datasets, titles, cmaps):
            if title == "Error":
                contour = ax.contourf(xi, yi, data, levels=100, cmap=cmap, vmin=vmin_err, vmax=vmax_err)
                cbar = fig.colorbar(contour, ax=ax, ticks=ticks_err)
                cbar.ax.set_yticklabels(tick_labels_err)
                cbar.set_label("Error Color Scale")
            else:
                levels = np.linspace(vmin_tp, vmax_tp, 100)
                contour = ax.contourf(xi, yi, data, levels=levels, cmap=cmap, vmin=vmin_tp, vmax=vmax_tp)
                cbar = fig.colorbar(contour, ax=ax, ticks=ticks_tp)
                cbar.ax.set_yticklabels(tick_labels_tp)
                cbar.set_label(f"{field} Color Scale")
            cbar.ax.tick_params(labelsize=14)
            ax.set_title(f"{field} - {title}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        plt.tight_layout()
        safe_field = field.replace(' ', '_').replace('[', '').replace(']', '')\
                          .replace('(', '').replace(')', '').replace('/', '_')
        error_dir = os.path.join(self.figures_dir, "error_figures")
        os.makedirs(error_dir, exist_ok=True)
        plt.savefig(os.path.join(error_dir, f"{safe_field}_comparison.png"))
        plt.close()
        print(f"Saved comparison plot for {field} to {error_dir}/{safe_field}_comparison.png")

        if self.model_type == "low_fi_fusion":
            residual_data = np.load(os.path.join("Outputs", self.project_name, "residual.npz"), allow_pickle=True)
            coords_all = residual_data["coords"]
            residuals_all = residual_data["outputs"]
            mu_r = residual_data["outputs_mean"]
            std_r = residual_data["outputs_std"]
            params_all = residual_data["params"]

            # === Match validation case parameters ===
            if not hasattr(self, "params") or self.params is None or len(self.params) == 0:
                print("⚠ No parameter columns provided in self.params — cannot match residual case.")
                return

            # Ensure order of columns in self.params matches how params were saved
            target_param = self.df_true[self.params].iloc[0].to_numpy().astype(float)

            # find matching samples in residual file
            tol = 1e-6
            param_mask = np.all(np.isclose(params_all, target_param, atol=tol), axis=1)

            if not np.any(param_mask):
                # helpful debug info
                unique_params = np.unique(params_all, axis=0)
                print(f"⚠ No residual entries matched parameters: {target_param}")
                print(f"Available unique param vectors in residual.npz:\n{unique_params[:5]}")
                return

            coords = coords_all[param_mask]
            residuals = residuals_all[param_mask]

            # === Denormalize residuals for this case ===
            r_norm = residuals
            r_denorm = residuals * std_r + mu_r

            # === Compute residual per field ===
            field_map = {
                "Velocity[i] (m/s)": 0,
                "Velocity[j] (m/s)": 1,
                "Absolute Pressure (Pa)": 2,
                "Density (kg/m^3)": 3,
                "Temperature (K)": 4
            }

            if field not in field_map:
                print(f"⚠ Field '{field}' not recognized in residual field map.")
                return

            idx = field_map[field]
            residual_field = r_denorm[..., idx].flatten()
            residual_norm_field = r_norm[..., idx].flatten()

            # === Interpolate residuals to same grid as other plots ===
            coords_flat = coords.reshape(-1, coords.shape[-1])
            zi_residual = griddata((coords_flat[:, 0], coords_flat[:, 1]), 
                                residual_field, (xi, yi), method='cubic')
            zi_residual_norm = griddata((coords_flat[:, 0], coords_flat[:, 1]), 
                                residual_norm_field, (xi, yi), method='cubic')

            # Apply the same mask used for other plots
            zi_residual[mask] = np.nan
            zi_residual_norm[mask] = np.nan

            # === Plot contour plots ===
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Changed to 3 subplots

            # Plot 1: Denormalized residual contour
            vmin_res = np.nanmin(zi_residual)
            vmax_res = np.nanmax(zi_residual)
            ticks_res = np.linspace(vmin_res, vmax_res, 9)
            tick_labels_res = [f"{t:.3f}" for t in ticks_res]

            levels_res = np.linspace(vmin_res, vmax_res, 100)
            contour1 = axs[0].contourf(xi, yi, zi_residual, levels=levels_res, 
                                    cmap='inferno', vmin=vmin_res, vmax=vmax_res)
            cbar1 = fig.colorbar(contour1, ax=axs[0], ticks=ticks_res)
            cbar1.ax.set_yticklabels(tick_labels_res)
            cbar1.set_label(f"Residual {field}")
            cbar1.ax.tick_params(labelsize=14)
            axs[0].set_title(f"Residual Field – {field}")
            axs[0].set_xlabel("X (m)")
            axs[0].set_ylabel("Y (m)")

            # Plot 2: Normalized residual contour
            vmin_res_norm = np.nanmin(zi_residual_norm)
            vmax_res_norm = np.nanmax(zi_residual_norm)
            ticks_res_norm = np.linspace(vmin_res_norm, vmax_res_norm, 9)
            tick_labels_res_norm = [f"{t:.3f}" for t in ticks_res_norm]

            levels_res_norm = np.linspace(vmin_res_norm, vmax_res_norm, 100)
            contour2 = axs[1].contourf(xi, yi, zi_residual_norm, levels=levels_res_norm, 
                                    cmap='inferno', vmin=vmin_res_norm, vmax=vmax_res_norm)
            cbar2 = fig.colorbar(contour2, ax=axs[1], ticks=ticks_res_norm)
            cbar2.ax.set_yticklabels(tick_labels_res_norm)
            cbar2.set_label(f"Residual {field} (Normalized)")
            cbar2.ax.tick_params(labelsize=14)
            axs[1].set_title(f"Normalized Residual Field – {field}")
            axs[1].set_xlabel("X (m)")
            axs[1].set_ylabel("Y (m)")

            # Plot 3: Absolute residual contour
            zi_abs_residual = np.abs(zi_residual)
            vmin_abs = np.nanmin(zi_abs_residual)
            vmax_abs = np.nanmax(zi_abs_residual)
            ticks_abs = np.linspace(vmin_abs, vmax_abs, 9)
            tick_labels_abs = [f"{t:.3f}" for t in ticks_abs]

            levels_abs = np.linspace(vmin_abs, vmax_abs, 100)
            contour3 = axs[2].contourf(xi, yi, zi_abs_residual, levels=levels_abs, 
                                    cmap='inferno', vmin=vmin_abs, vmax=vmax_abs)
            cbar3 = fig.colorbar(contour3, ax=axs[2], ticks=ticks_abs)
            cbar3.ax.set_yticklabels(tick_labels_abs)
            cbar3.set_label(f"|Residual| {field}")
            cbar3.ax.tick_params(labelsize=14)
            axs[2].set_title(f"Absolute Residual Field – {field}")
            axs[2].set_xlabel("X (m)")
            axs[2].set_ylabel("Y (m)")

            plt.tight_layout()
            plt.savefig(os.path.join(error_dir, f"{safe_field}_residual_contours.png"))
            plt.close()

            print(f"✅ Residual plots saved for field '{field}' and case params {target_param}")


        
    def _plot_predicted_only(self, params):
        self._define_ouput_folders()
        fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                  "Absolute Pressure (Pa)", "Density (kg/m^3)", "Temperature (K)"]
        for field in fields:
            self._plot_single_prediction(field, params)

    def _plot_single_prediction(self, field, params):
        x = self.df_pred["X (m)"].values
        y = self.df_pred["Y (m)"].values
        z = self.df_pred[field].values
        xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500))
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        a, b = params[0], params[1]
        x0, y0 = -2.5, 0
        mask = ((xi - x0)**2 / a**2 + (yi - y0)**2 / b**2) <= 1
        zi[mask] = np.nan

        fig, ax = plt.subplots(figsize=(6, 6))
        contour = ax.contourf(xi, yi, zi, levels=100, cmap="inferno")
        cbar = fig.colorbar(contour)
        cbar.set_label(f"{field} Color Scale")
        ax.set_title(f"{field} - Predicted Only")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        safe_field = field.replace(' ', '_').replace('[', '').replace(']', '')\
                          .replace('(', '').replace(')', '').replace('/', '_')
        pred_dir = os.path.join(self.figures_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        plt.savefig(os.path.join(pred_dir, f"{safe_field}_predicted.png"))
        plt.close()
        print(f"Saved predicted plot for {field} to {pred_dir}/{safe_field}_predicted.png")

    def plot_point_cloud(self, field, error):
        """
        Plot field comparison using raw point data without interpolation
        """
        x = self.df_true["X (m)"].values
        y = self.df_true["Y (m)"].values
        
        # Get field values
        true_values = self.df_true[field].values
        pred_values = self.df_pred[field].values
        error_values = error[field].values
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(36, 12))
        
        # Data for each subplot
        datasets = [pred_values, true_values, error_values]
        titles = ["Predicted", "True", "Error"]
        cmaps = ["inferno", "inferno", "inferno"]
        
        # Calculate consistent color scales
        vmin_tp = min(np.min(true_values), np.min(pred_values))
        vmax_tp = max(np.max(true_values), np.max(pred_values))
        
        vmin_err = np.min(error_values)
        vmax_err = np.max(error_values)
        
        # Create scatter plots
        for i, (ax, data, title, cmap) in enumerate(zip(axs, datasets, titles, cmaps)):
            if title == "Error":
                # Use error color scale
                scatter = ax.scatter(x, y, c=data, cmap=cmap, s=1, 
                                vmin=vmin_err, vmax=vmax_err)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label("Error Color Scale")
            else:
                # Use true/predicted color scale
                scatter = ax.scatter(x, y, c=data, cmap=cmap, s=1, 
                                vmin=vmin_tp, vmax=vmax_tp)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label(f"{field} Color Scale")
            
            cbar.ax.tick_params(labelsize=14)
            ax.set_title(f"{field} - {title}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save the plot
        safe_field = field.replace(' ', '_').replace('[', '').replace(']', '')\
                        .replace('(', '').replace(')', '').replace('/', '_')
        error_dir = os.path.join(self.figures_dir, "error_figures")
        os.makedirs(error_dir, exist_ok=True)
        plt.savefig(os.path.join(error_dir, f"{safe_field}_pointcloud_comparison.png"), dpi=150)
        plt.close()
        print(f"Saved point cloud comparison plot for {field} to {error_dir}/{safe_field}_pointcloud_comparison.png")

