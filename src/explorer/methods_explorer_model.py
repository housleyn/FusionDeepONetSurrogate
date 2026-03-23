from pathlib import Path
import csv
import os

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from ..models.fusion_model import FusionDeepONet
from ..models.vanilla_model import VanillaDeepONet
from ..models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet


class MethodsExplorerModel:
    def _load_model_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.config_path = str(config_path)
        self.project_name = config["project_name"]

        self.coord_dim = config["coord_dim"]
        self.param_dim = config["param_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.output_dim = config["output_dim"]
        self.model_type = config["model_type"]
        self.distance_dim = config["distance_dim"]
        self.param_columns = config["param_columns"]
        self.distance_columns = config["distance_columns"]

        self.batch_size = config.get("batch_size", None)
        self.num_epochs = config.get("num_epochs", None)
        self.test_size = config.get("test_size", None)
        self.print_every = config.get("print_every", None)
        self.shuffle = config.get("shuffle", None)
        self.dimension = config.get("dimension", None)
        self.lhs_sample = config.get("lhs_sample", None)
        self.lr = config.get("lr", None)
        self.lr_gamma = config.get("lr_gamma", None)
        self.low_fi_dropout = config.get("low_fi_dropout", 0.0)
        self.dropout = config.get("dropout", 0.0)

    def _set_model_paths_from_config(self):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.outputs_dir = os.path.join(self.project_root, "Outputs", self.project_name)
        self.model_dir = os.path.join(self.outputs_dir, "model")

        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.stats_path = os.path.join(self.outputs_dir, "processed_data.npz")
        self.predicted_output_file = os.path.join(self.outputs_dir, "predicted_output.csv")

        if self.model_type == "vanilla":
            self.model_path = os.path.join(self.model_dir, "vanilla_deeponet.pt")
            self.low_fi_model_path = None
            self.low_fi_stats_path = None

        elif self.model_type == "FusionDeepONet":
            self.model_path = os.path.join(self.model_dir, "fusion_deeponet.pt")
            self.low_fi_model_path = None
            self.low_fi_stats_path = None

        elif self.model_type == "low_fi_fusion":
            self.model_path = os.path.join(self.model_dir, "fusion_deeponet.pt")
            self.low_fi_model_path = os.path.join(self.model_dir, "low_fi_fusion_deeponet.pt")
            self.low_fi_stats_path = os.path.join(self.outputs_dir, "processed_low_fi_data.npz")
            self.residual_stats_path = os.path.join(self.outputs_dir, "residual.npz")

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _load_model_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32).to(self.device),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32).to(self.device),
        }

    def _denormalize_outputs(self, output):
        return output * self.stats["outputs_std"] + self.stats["outputs_mean"]

    def _low_fi_denormalize(self, output):
        return output * self.low_fi_stats["outputs_std"] + self.low_fi_stats["outputs_mean"]

    def _residual_denormalize(self, output):
        return output * self.residual_stats["outputs_std"] + self.residual_stats["outputs_mean"]

    def _build_model_instance(self):
        with np.load(self.stats_path) as data:
            param_dim_from_stats = data["params"].shape[1]

        if self.model_type == "vanilla":
            model = VanillaDeepONet(
                self.coord_dim,
                param_dim_from_stats,
                self.hidden_size,
                self.num_hidden_layers,
                self.output_dim,
            )
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            return model

        if self.model_type == "FusionDeepONet":
            model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
            )
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            return model

        if self.model_type == "low_fi_fusion":
            model_1 = Low_Fidelity_FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                npz_path=self.stats_path,
            )
            model_1.load_state_dict(torch.load(self.low_fi_model_path, map_location=self.device))
            model_1.eval()

            model_2 = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                aux_dim=self.output_dim,
            )
            model_2.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model_2.eval()

            return model_1, model_2

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def initialize_model(self, config_path):
        self._load_model_config(config_path)
        self._set_model_paths_from_config()

        self.stats = self._load_model_stats(self.stats_path)

        if self.model_type == "low_fi_fusion":
            self.low_fi_stats = self._load_model_stats(self.low_fi_stats_path)
            self.residual_stats = self._load_model_stats(self.residual_stats_path)

            self.model_1, self.model_2 = self._build_model_instance()
            self.model_1 = self.model_1.to(self.device)
            self.model_2 = self.model_2.to(self.device)
            self.model = None
        else:
            self.low_fi_stats = None
            self.residual_stats = None

            self.model = self._build_model_instance()
            self.model = self.model.to(self.device)
            self.model_1 = None
            self.model_2 = None

    def load_explorer_csv_input(self, csv_path):
        df = pd.read_csv(csv_path)
        coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values
        params_np = df[self.param_columns].iloc[0].values.astype(np.float32)
        sdf_np = df[self.distance_columns].values
        return df, coords_np, params_np, sdf_np

    def predict_from_arrays(self, coords_np, params_np, sdf_np):
        coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)   # [1, N, 3]
        params = torch.tensor(params_np, dtype=torch.float32).unsqueeze(0).to(self.device)   # [1, P]
        sdf = torch.tensor(sdf_np, dtype=torch.float32).unsqueeze(0).to(self.device)         # [1, N, 1]

        if self.model_type == "low_fi_fusion":
            with torch.no_grad():
                low_fi_pred = self.model_1(coords, params, sdf)
                low_fi_pred_denorm = self._low_fi_denormalize(low_fi_pred)
                low_fi_pred_residual_norm = (
                    low_fi_pred_denorm - self.residual_stats["outputs_mean"]
                ) / self.residual_stats["outputs_std"]
                residual_pred = self.model_2(coords, params, sdf, aux=low_fi_pred_residual_norm)
                pred = self._residual_denormalize(residual_pred) + self._low_fi_denormalize(low_fi_pred)
        else:
            with torch.no_grad():
                pred = self.model(coords, params, sdf)
            pred = self._denormalize_outputs(pred)

        return pred.squeeze(0).cpu().numpy()

    def save_prediction_csv(self, coords_np, output_np, out_path):
        headers = [
            "Density (kg/m^3)",
            "Velocity[i] (m/s)",
            "Velocity[j] (m/s)",
            "Velocity[k] (m/s)",
            "Absolute Pressure (Pa)",
            "Temperature (K)",
            "X (m)",
            "Y (m)",
            "Z (m)",
        ]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, mode="w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)

            for i in range(coords_np.shape[0]):
                density = output_np[i, 0]
                vel = output_np[i, 1:4]
                pressure = output_np[i, 4]
                temperature = output_np[i, 5]
                row = [density] + list(vel) + [pressure, temperature] + list(coords_np[i])
                writer.writerow(row)

    def predict_from_csv(
        self,
        input_csv_path,
        output_csv_path=None,
    ):
        df_in, coords_np, params_np, sdf_np = self.load_explorer_csv_input(input_csv_path)
        output_np = self.predict_from_arrays(coords_np, params_np, sdf_np)

        pred_df = pd.DataFrame({
            "Density (kg/m^3)": output_np[:, 0],
            "Velocity[i] (m/s)": output_np[:, 1],
            "Velocity[j] (m/s)": output_np[:, 2],
            "Velocity[k] (m/s)": output_np[:, 3],
            "Absolute Pressure (Pa)": output_np[:, 4],
            "Temperature (K)": output_np[:, 5],
            "X (m)": coords_np[:, 0],
            "Y (m)": coords_np[:, 1],
            "Z (m)": coords_np[:, 2],
        })

        if output_csv_path is None:
            output_csv_path = self.predicted_output_file

        self.save_prediction_csv(coords_np, output_np, output_csv_path)
        return pred_df, df_in

    def plot_predicted_field(
        self,
        pred_df,
        input_df,
        field_name,
        output_plot,
        levels=100,
        cmap="inferno",
        vmin=None,
        vmax=None,
    ):
        x = pred_df["X (m)"].to_numpy()
        y = pred_df["Y (m)"].to_numpy()
        values = pred_df[field_name].to_numpy()
        is_surface = input_df["is_on_surface"].to_numpy().astype(int)

        triang = tri.Triangulation(x, y)
        surf_tri_mask = is_surface[triang.triangles].any(axis=1)
        triang.set_mask(surf_tri_mask)

        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.tricontourf(
            triang,
            values,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(field_name)

        surface_mask = is_surface == 1
        ax.scatter(
            x[surface_mask],
            y[surface_mask],
            s=4,
            c="white",
            linewidths=0,
            alpha=0.8,
            label="Surface points",
        )

        ax.set_title(f"Predicted {field_name}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(output_plot, dpi=250)
        plt.close(fig)

        return output_plot

    def predict_and_plot_single_field(
        self,
        input_csv_path,
        field_name,
        output_csv_path=None,
        output_plot_path=None,
        levels=100,
        cmap="viridis",
    ):
        pred_df, input_df = self.predict_from_csv(
            input_csv_path=input_csv_path,
            output_csv_path=output_csv_path,
        )

        plot_path = None
        if output_plot_path is not None:
            plot_path = self.plot_predicted_field(
                pred_df=pred_df,
                input_df=input_df,
                field_name=field_name,
                output_plot=output_plot_path,
                levels=levels,
                cmap=cmap,
            )

        return pred_df, input_df, plot_path

    def predict_and_plot_all_fields(
        self,
        input_csv_path,
        output_csv_path=None,
        plot_dir=None,
        levels=100,
        cmap="viridis",
    ):
        pred_df, input_df = self.predict_from_csv(
            input_csv_path=input_csv_path,
            output_csv_path=output_csv_path,
        )

        plot_paths = {}
        if plot_dir is not None:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)

            field_names = [
                "Density (kg/m^3)",
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Velocity[k] (m/s)",
                "Absolute Pressure (Pa)",
                "Temperature (K)",
            ]

            for field_name in field_names:
                safe_name = (
                    field_name.replace("/", "_")
                    .replace("[", "")
                    .replace("]", "")
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                plot_path = plot_dir / f"{safe_name}.png"
                self.plot_predicted_field(
                    pred_df=pred_df,
                    input_df=input_df,
                    field_name=field_name,
                    output_plot=plot_path,
                    levels=levels,
                    cmap=cmap,
                )
                plot_paths[field_name] = plot_path

        return pred_df, input_df, plot_paths