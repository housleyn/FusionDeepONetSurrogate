from pathlib import Path
import torch
import numpy as np

from .methods_explorer import (
    build_single_case,
    plot_distance_field,
    build_position_path,
    save_gif_from_frames,
)
from .methods_explorer_model import MethodsExplorerModel


class Explorer(MethodsExplorerModel):
    def __init__(
        self,
        project_name="ExplorerProject",
        output_dir="Explorer_outputs",
        x_lim=(-2.5, 17.5),
        y_lim=(-15.0, 15.0),
        radius=1.0,
        mach=7.5,
        total_points=30000,
        n_surface_each=180,
        seed=42,
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.radius = radius
        self.mach = mach
        self.total_points = total_points
        self.n_surface_each = n_surface_each
        self.seed = seed

        self.center_main = (0.0, 0.0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.df = None
        self.points = None
        self.surface_points = None
        self.is_surface = None
        self.secondary_center = None

        self.model = None
        self.model_1 = None
        self.model_2 = None
        self.stats = None
        self.low_fi_stats = None
        self.residual_stats = None

    def create_single_case(
        self,
        x_secondary,
        y_secondary,
        csv_path=None,
    ):
        self.secondary_center = (float(x_secondary), float(y_secondary))

        if csv_path is None:
            csv_path = self.output_dir / "generated_single_case.csv"
        else:
            csv_path = Path(csv_path)

        result = build_single_case(
            x_secondary=x_secondary,
            y_secondary=y_secondary,
            output_csv=csv_path,
            mach=self.mach,
            total_points=self.total_points,
            n_surface_each=self.n_surface_each,
            seed=self.seed,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            radius=self.radius,
        )

        self.df = result["df"]
        self.points = result["points"]
        self.surface_points = result["surface_points"]
        self.is_surface = result["is_surface"]

        return self.df

    def plot_single_case_distance(
        self,
        plot_name="generated_single_case_distance_plot.png",
        levels=100,
    ):
        if self.df is None:
            raise ValueError("No case has been created yet. Run create_single_case() first.")

        return plot_distance_field(
            df=self.df,
            is_surface=self.is_surface,
            surface_points=self.surface_points,
            output_plot=self.output_dir / plot_name,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            levels=levels,
        )

    def create_case_series(
        self,
        positions,
        csv_subdir="csv_frames",
        plot_subdir="plot_frames",
        levels=100,
    ):
        csv_dir = self.output_dir / csv_subdir
        plot_dir = self.output_dir / plot_subdir
        csv_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []

        for i, (x_secondary, y_secondary) in enumerate(positions):
            csv_name = csv_dir / f"case_{i:04d}.csv"
            plot_name = plot_dir / f"frame_{i:04d}.png"

            self.create_single_case(
                x_secondary=x_secondary,
                y_secondary=y_secondary,
                csv_name=csv_name,
            )

            frame_path = self.plot_single_case_distance(
                plot_name=plot_name,
                levels=levels,
            )
            frame_paths.append(frame_path)

        return frame_paths

    def build_linear_path(self, start, end, n_frames):
        return build_position_path(start=start, end=end, n_frames=n_frames)

    def make_gif_frames(
        self,
        positions,
        gif_name="distance_animation.gif",
        csv_subdir="csv_frames",
        plot_subdir="plot_frames",
        fps=10,
        levels=100,
    ):
        frame_paths = self.create_case_series(
            positions=positions,
            csv_subdir=csv_subdir,
            plot_subdir=plot_subdir,
            levels=levels,
        )

        gif_path = self.output_dir / gif_name
        save_gif_from_frames(frame_paths=frame_paths, gif_path=gif_path, fps=fps)
        return gif_path


    def make_prediction_gif(
        self,
        positions,
        field_name="Absolute Pressure (Pa)",
        gif_name="predicted_field.gif",
        csv_subdir="csv_frames",
        pred_csv_subdir="predicted_csv_frames",
        plot_subdir="predicted_plot_frames",
        fps=10,
        levels=100,
        cmap="viridis",
    ):
        csv_dir = self.output_dir / csv_subdir
        pred_csv_dir = self.output_dir / pred_csv_subdir
        plot_dir = self.output_dir / plot_subdir

        csv_dir.mkdir(parents=True, exist_ok=True)
        pred_csv_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        safe_name = (
            field_name.replace("/", "_")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        pred_results = []
        all_values = []

        # -----------------------------
        # Pass 1: create inputs + predict
        # -----------------------------
        for i, (x_secondary, y_secondary) in enumerate(positions):
            input_csv = csv_dir / f"case_{i:04d}.csv"
            pred_csv = pred_csv_dir / f"pred_{i:04d}.csv"

            self.create_single_case(
                x_secondary=x_secondary,
                y_secondary=y_secondary,
                csv_path=input_csv,
            )

            pred_df, input_df = self.predict_from_csv(
                input_csv_path=input_csv,
                output_csv_path=pred_csv,
            )

            pred_results.append((pred_df, input_df, i))
            all_values.append(pred_df[field_name].values)

        # -----------------------------
        # Compute global color scale
        # -----------------------------
        all_values = np.concatenate(all_values)
        vmin = np.min(all_values)
        vmax = np.max(all_values)

        # Optional robust scaling instead:
        # vmin = np.percentile(all_values, 1)
        # vmax = np.percentile(all_values, 99)

        # -----------------------------
        # Pass 2: plot with fixed scale
        # -----------------------------
        frame_paths = []

        for pred_df, input_df, i in pred_results:
            plot_path = plot_dir / f"{safe_name}_{i:04d}.png"

            frame_path = self.plot_predicted_field(
                pred_df=pred_df,
                input_df=input_df,
                field_name=field_name,
                output_plot=plot_path,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            frame_paths.append(frame_path)

        # -----------------------------
        # Build GIF
        # -----------------------------
        gif_path = self.output_dir / gif_name
        save_gif_from_frames(frame_paths=frame_paths, gif_path=gif_path, fps=fps)

        return gif_path