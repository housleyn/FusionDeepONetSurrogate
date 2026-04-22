from src.inference import Inference
from src.postprocess import Postprocess
import os
import numpy as np
from .plotting_surrogate import (plot_all_inference_errors)
import pandas as pd

def infer_and_validate(self, file):
        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            low_fi_stats_path = os.path.join(
                self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            inference = Inference(self.project_name, config_path=self.config_path, model_path=self.model_path, stats_path=stats_path, low_fi_stats_path=low_fi_stats_path, low_fi_model_path=self.low_fi_model_path if self.model_type=="low_fi_fusion" else None)
        else:
            stats_path = self.npz_path
            inference = Inference(self.project_name, config_path=self.config_path, model_path=self.model_path, stats_path=stats_path, low_fi_model_path=self.low_fi_model_path if self.model_type=="low_fi_fusion" else None)
        coords_np, params_np, sdf_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params, sdf_np)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        print(f"Inference complete. Output saved to {self.predicted_output_file}.")
        print("Beginning postprocessing...")
        postprocess = Postprocess(config_path=self.config_path, path_true=file, path_pred=self.predicted_output_file)
        postprocess.run(self.dimension)
    
def inference(self, file):
    inference = Inference(self.project_name,config_path=self.config_path, model_path=self.model_path, stats_path=self.npz_path, low_fi_model_path=self.low_fi_model_path if self.model_type=="low_fi_fusion" else None)
    coords_np, params_np, sdf_np = inference.load_csv_input(file)
    params = params_np[1]
    output = inference.predict(coords_np, params)
    inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
    print(f"Inference complete. Output saved to {self.predicted_output_file}.")
    print("Beginning postprocessing...")
    postprocess = Postprocess(config_path=self.config_path, path_true=None, path_pred=self.predicted_output_file) 
    postprocess._plot_predicted_only(params)
    
def infer_all_unseen(self, folder):
    errors = []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(
                self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            low_fi_stats_path = stats_path
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                low_fi_stats_path=low_fi_stats_path,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
            )
        else:
            stats_path = self.npz_path
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
            )

        coords_np, params_np, sdf_np = inference.load_csv_input(file_path)
        params = params_np[1]
        output = inference.predict(coords_np, params, sdf_np)

        predicted_output = os.path.join(
            self.project_root, "Outputs", self.project_name, "all_inferences", "predicted_" + filename
        )
        inference.save_to_csv(coords_np, output, out_path=predicted_output)
        print(f"Inference complete. Output saved to {predicted_output}.")

        postprocess = Postprocess(
            config_path=self.config_path,
            path_true=file_path,
            path_pred=predicted_output
        )
        file_errors = dict(postprocess.get_errors(self.dimension))
        errors.append(file_errors)

    field_aggregates = {}
    for fe in errors:
        if not isinstance(fe, dict):
            continue

        for field, val in fe.items():
            scalar = float(np.asarray(val).squeeze())
            field_aggregates.setdefault(field, []).append(scalar)

    plot_all_inference_errors(self, field_aggregates)

def infer_full_flowfield_l2(self, folder):
    """
    Infer all files in a folder, concatenate all slices/files together,
    and compute one global L2 error per field.

    This is intended to answer:
    'How well does the model predict the full 3D flowfield?'
    """

    true_fields = {}
    pred_fields = {}

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(folder, filename)

        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(
                self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            low_fi_stats_path = stats_path
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                low_fi_stats_path=low_fi_stats_path,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
            )
        else:
            stats_path = self.npz_path
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
            )

        coords_np, params_np, sdf_np = inference.load_csv_input(file_path)
        params = params_np[1]
        output = inference.predict(coords_np, params, sdf_np)

        predicted_output = os.path.join(
            self.project_root, "Outputs", self.project_name, "all_inferences", "predicted_" + filename
        )
        inference.save_to_csv(coords_np, output, out_path=predicted_output)
        print(f"Inference complete. Output saved to {predicted_output}.")

        # Read true and predicted CSVs
        df_true = pd.read_csv(file_path)
        df_pred = pd.read_csv(predicted_output)

        # Map friendly field names to CSV column names
        field_map = {
            "u": "Velocity[i] (m/s)",
            "v": "Velocity[j] (m/s)",
            "w": "Velocity[k] (m/s)",
            "p": "Absolute Pressure (Pa)",
            "rho": "Density (kg/m^3)",
            "T": "Temperature (K)",
        }

        for field_name, col in field_map.items():
            if col not in df_true.columns or col not in df_pred.columns:
                continue

            true_fields.setdefault(field_name, []).append(df_true[col].to_numpy().ravel())
            pred_fields.setdefault(field_name, []).append(df_pred[col].to_numpy().ravel())

    global_errors = {}

    for field_name in true_fields:
        y_true = np.concatenate(true_fields[field_name], axis=0)
        y_pred = np.concatenate(pred_fields[field_name], axis=0)

        denom = np.linalg.norm(y_true)
        if denom == 0:
            global_errors[field_name] = np.nan
        else:
            global_errors[field_name] = 100.0 * np.linalg.norm(y_true - y_pred) / denom

    print("\n=== Global L2 Error Across Full 3D Flowfield ===")
    for field_name, err in global_errors.items():
        print(f"{field_name}: {err:.6f}%")

    return global_errors

def infer_and_validate_3d(self, file):
    """
    Run inference on a single full 3D CSV file by passing one row at a time
    through the model, then combine all predictions and compute global L2
    errors for the full 3D flowfield.

    No plotting.
    """

    if self.model_type == "low_fi_fusion":
        stats_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
        )
        low_fi_stats_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
        )
        inference = Inference(
            self.project_name,
            config_path=self.config_path,
            model_path=self.model_path,
            stats_path=stats_path,
            low_fi_stats_path=low_fi_stats_path,
            low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
        )
    else:
        stats_path = self.npz_path
        inference = Inference(
            self.project_name,
            config_path=self.config_path,
            model_path=self.model_path,
            stats_path=stats_path,
            low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
        )

    coords_np, params_np, sdf_np = inference.load_csv_input(file)

    all_outputs = []

    n_points = coords_np.shape[0]
    print(f"Beginning 3D row-wise inference for {n_points} points...")

    for i in range(n_points):
        coord_i = coords_np[i:i+1]   # shape: (1, 3)
        param_i = params_np[i]       # shape: (param_dim,)
        sdf_i = sdf_np[i:i+1]        # shape: (1, sdf_dim)

        output_i = inference.predict(coord_i, param_i, sdf_i)
        all_outputs.append(output_i)

        if (i + 1) % 10000 == 0 or (i + 1) == n_points:
            print(f"Inferred {i + 1}/{n_points} points")

    output = np.vstack(all_outputs)

    predicted_output = os.path.join(
        self.project_root, "Outputs", self.project_name, "predicted_output_3d.csv"
    )
    inference.save_to_csv(coords_np, output, out_path=predicted_output)
    print(f"Inference complete. Output saved to {predicted_output}.")

    df_true = pd.read_csv(file)
    df_pred = pd.read_csv(predicted_output)

    field_map = {
        "rho": ("Density (kg/m^3)", "Density (kg/m^3)"),
        "u": ("Velocity[i] (m/s)", "Velocity[i] (m/s)"),
        "v": ("Velocity[j] (m/s)", "Velocity[j] (m/s)"),
        "w": ("Velocity[k] (m/s)", "Velocity[k] (m/s)"),
        "p": ("Absolute Pressure (Pa)", "Absolute Pressure (Pa)"),
        "T": ("Temperature (K)", "Temperature (K)"),
    }

    global_errors = {}

    print("\n=== Global L2 Error Across Full 3D Flowfield ===")
    for field_name, (true_col, pred_col) in field_map.items():
        if true_col not in df_true.columns:
            print(f"Skipping {field_name}: true column '{true_col}' not found.")
            continue
        if pred_col not in df_pred.columns:
            print(f"Skipping {field_name}: predicted column '{pred_col}' not found.")
            continue

        y_true = df_true[true_col].to_numpy().ravel()
        y_pred = df_pred[pred_col].to_numpy().ravel()

        denom = np.linalg.norm(y_true)
        if denom == 0:
            err = np.nan
        else:
            err = 100.0 * np.linalg.norm(y_true - y_pred) / denom

        global_errors[field_name] = err
        print(f"{field_name}: {err:.6f}%")

    return global_errors