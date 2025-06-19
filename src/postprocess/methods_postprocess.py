import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

class MethodsPostprocess:
    def run(self):
        error = self._calculate_error()
        self._define_ouput_folders()
        fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                "Absolute Pressure (Pa)", "Density (kg/m^3)"]
        
        for field in fields:
            self._calculate_relative_l2_error(field)
        self._create_table()
        for field in fields:
            self._plot_fields(field, error)

    def _calculate_error(self):
        error = (self.df_true - self.df_pred)
        error["X (m)"] = self.df_true["X (m)"]
        error["Y (m)"] = self.df_true["Y (m)"]
        return error

    def _calculate_relative_l2_error(self, field):
            u_true = self.df_true[field].values
            u_pred = self.df_pred[field].values
            rel_l2 = 100 * np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
            self.errors[field] = rel_l2

    def _define_ouput_folders(self):
        self.figures_dir = os.path.join("Tables_and_Figures", "figures")
        self.tables_dir = os.path.join("Tables_and_Figures", "tables")
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


    def _plot_fields(self,field, error):
        x = self.df_true["X (m)"].values
        y = self.df_true["Y (m)"].values

        xi = np.linspace(x.min(), x.max(), 500)
        yi = np.linspace(y.min(), y.max(), 500)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate true, predicted, and error values
        zi_true = griddata((x, y), self.df_true[field].values, (xi, yi), method='cubic')
        zi_pred = griddata((x, y), self.df_pred[field].values, (xi, yi), method='cubic')
        zi_error = griddata((x, y), error[field].values, (xi, yi), method='cubic')
        zi_error = np.abs(zi_error)

            # Define semi-ellipse mask
        a = 1.44  # replace with actual
        b = 1.16  # replace with actual
        x0, y0 = -2.5, 0
        mask = ((xi - x0)**2 / a**2 + (yi - y0)**2 / b**2) <= 1

        # Mask the data
        zi_true[mask] = np.nan
        zi_pred[mask] = np.nan
        zi_error[mask] = np.nan

        

        # Plot all three as subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["Predicted", "True", "Error"]
        datasets = [zi_pred, zi_true, zi_error]
        cmaps = ["inferno", "inferno", "inferno"]
        # Set shared vmin, vmax for true/pred
        vmin_tp = np.nanmin([zi_true, zi_pred])
        vmax_tp = np.nanmax([zi_true, zi_pred])
        ticks_tp = np.linspace(vmin_tp, vmax_tp, 9)
        tick_labels_tp = [f"{t:.3f}" for t in ticks_tp]  # same format

        # Separate error settings
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
                cbar.ax.tick_params(labelsize=14)
            else:
                contour = ax.contourf(xi, yi, data, levels=100, cmap=cmap, vmin=vmin_tp, vmax=vmax_tp)
                cbar = fig.colorbar(contour, ax=ax, ticks=ticks_tp)
                cbar.ax.set_yticklabels(tick_labels_tp)
                cbar.set_label(f"{field} Color Scale")
                cbar.ax.tick_params(labelsize=14)

            ax.set_title(f"{field} - {title}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")


        plt.tight_layout()
        safe_field = field.replace(' ', '_')\
                    .replace('[','')\
                    .replace(']','')\
                    .replace('(','')\
                    .replace(')','')\
                    .replace('/','_')
        error_dir = os.path.join(self.figures_dir, "error_figures")
        os.makedirs(error_dir, exist_ok=True)
        plt.savefig(os.path.join(error_dir, f"{safe_field}_comparison.png"))
        plt.close()
        print(f"Saved comparison plot for {field} to {error_dir}/{safe_field}_comparison.png")

    def _plot_predicted_only(self):
        self._define_ouput_folders()
        fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                "Absolute Pressure (Pa)", "Density (kg/m^3)"]

        for field in fields:
            x = self.df_pred["X (m)"].values
            y = self.df_pred["Y (m)"].values
            z = self.df_pred[field].values

            xi = np.linspace(x.min(), x.max(), 500)
            yi = np.linspace(y.min(), y.max(), 500)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='cubic')

            # Mask ellipse
            a, b = 1.44, 1.16
            x0, y0 = -2.5, 0
            mask = ((xi - x0)**2 / a**2 + (yi - y0)**2 / b**2) <= 1
            zi[mask] = np.nan

            # Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            contour = ax.contourf(xi, yi, zi, levels=100, cmap="inferno")
            cbar = fig.colorbar(contour)
            cbar.set_label(f"{field} Color Scale")
            ax.set_title(f"{field} - Predicted Only")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            safe_field = field.replace(' ', '_').replace('[','').replace(']','')\
                            .replace('(','').replace(')','').replace('/','_')
            pred_dir = os.path.join(self.figures_dir, "predicted_only")
            os.makedirs(pred_dir, exist_ok=True)
            plt.savefig(os.path.join(pred_dir, f"{safe_field}_predicted.png"))
            plt.close()
            print(f"Saved predicted plot for {field} to {pred_dir}/{safe_field}_predicted.png")
