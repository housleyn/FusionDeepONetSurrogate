from src.inference import Inference
from src.postprocess import Postprocess
import os
import numpy as np
from .plotting_surrogate import (plot_all_inference_errors)

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
    postprocess = Postprocess(self.project_name, path_true=None, path_pred=self.predicted_output_file) #wrong
    postprocess._plot_predicted_only(params)
    
def infer_all_unseen(self, folder):
    errors = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz")
            low_fi_stats_path = stats_path
            inference = Inference(
                self.project_name, config_path=self.config_path, model_path=self.model_path,
                stats_path=stats_path, low_fi_stats_path=low_fi_stats_path,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None
            )
        else:
            stats_path = self.npz_path
            inference = Inference(
                self.project_name, config_path=self.config_path, model_path=self.model_path,
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

        postprocess = Postprocess(config_path=self.config_path, path_true=file_path, path_pred=predicted_output)
        file_errors = dict(postprocess.get_errors())
        errors.append(file_errors)

    
    field_aggregates = {}
    for fe in errors:
        if not isinstance(fe, dict):
            continue
        for field, vals in fe.items():
            arr = np.asarray(vals).ravel()
            field_aggregates.setdefault(field, []).append(arr)

    
    plot_all_inference_errors(field_aggregates)
