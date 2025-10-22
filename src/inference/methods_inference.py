import torch
from ..models.fusion_model import FusionDeepONet
from ..models.vanilla_model import VanillaDeepONet
from ..models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet
import csv
import numpy as np 
import pandas as pd
import os

class MethodsInference:
    def _load_model(self, path, stats_path):
        with np.load(stats_path) as data:
            param_dim = data["params"].shape[1]
        if self.model_type == "vanilla":
            model = VanillaDeepONet(self.coord_dim, param_dim, self.hidden_size, self.num_hidden_layers, self.output_dim)
        elif self.model_type == "FusionDeepONet":
            model = FusionDeepONet(coord_dim=self.coord_dim+self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim)
        elif self.model_type == "low_fi_fusion":
            model_1 = Low_Fidelity_FusionDeepONet(coord_dim=self.coord_dim+self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim, npz_path=self.npz_path)
            model_1.load_state_dict(torch.load(self.low_fi_model_path, map_location=self.device))
            model_1.eval()
            model_2 = FusionDeepONet(coord_dim=self.coord_dim+self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim, aux_dim=self.output_dim)
            model_2.load_state_dict(torch.load(path, map_location=self.device))
            model_2.eval()
            return model_1, model_2
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

    def _denormalize(self, output):
        return output * self.stats["outputs_std"] + self.stats["outputs_mean"]
    def _low_fi_denormalize(self, output):
        return output * self.low_fi_stats["outputs_std"] + self.low_fi_stats["outputs_mean"]
    def _residual_denormalize(self, output):
        return output * self.residual_stats["outputs_std"] + self.residual_stats["outputs_mean"]

    def load_csv_input(self, csv_path):
        df = pd.read_csv(csv_path)
        coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values
        params_np = df[self.param_columns].values
        sdf_np = df[self.distance_columns].values
        return coords_np, params_np, sdf_np
    
    def predict(self, coords_np, params, sdf):
        coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(self.device)
        sdf = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0).to(self.device)
        if self.model_type == "low_fi_fusion":
            with torch.no_grad():
                low_fi_pred = self.model_1(coords, params, sdf)
                
                residual_pred = self.model_2(coords, params, sdf, aux=low_fi_pred) 

                pred = self._residual_denormalize(residual_pred) + self._low_fi_denormalize(low_fi_pred)

        else:
            with torch.no_grad():
                pred = self.model(coords, params, sdf)
            pred = self._denormalize(pred)
        
        return pred.squeeze(0).cpu().numpy()
    
    def save_to_csv(self, coords_np, output_np, out_path=None):

        headers = [
            'Density (kg/m^3)', 'Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)',
            'Absolute Pressure (Pa)', 'Temperature (K)', "X (m)", "Y (m)", "Z (m)"
        ]

        if out_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "Outputs")
            out_path = os.path.join(outputs_dir,self.project_name, "predicted_output.csv")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, mode='w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
            for i in range(coords_np.shape[0]):
                density = output_np[i, 0]
                vel = output_np[i, 1:4]
                pressure = output_np[i, 4]
                temperature = output_np[i, 5]

                row = [density] + list(vel) + [pressure, temperature] + list(coords_np[i])
                writer.writerow(row)


    
