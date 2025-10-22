import torch
import os
import yaml
class BaseInference:
    def __init__(self, project_name, config_path ,model_path, stats_path, low_fi_stats_path=None, param_columns=None, distance_columns=None, low_fi_model_path=None):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.coord_dim = config["coord_dim"] 
        self.param_dim = config["param_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.output_dim = config["output_dim"]
        self.model_type = config["model_type"]
        self.distance_dim = config["distance_dim"]
        self.low_fi_model_path = low_fi_model_path
        
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param_columns = param_columns
        self.distance_columns = distance_columns
        self.npz_path = stats_path
        if self.model_type == "low_fi_fusion":
            # Load both LF and HF models in a single call where `path` points to the HF checkpoint.
            # `_load_model` will internally use `self.low_fi_model_path` for the LF model and `path` for the HF model.

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "Outputs", project_name, "model", "fusion_deeponet.pt"
            )
            residual_stats_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "Outputs", project_name, "residual.npz"
            )
            self.model_1, self.model_2 = self._load_model(model_path, stats_path)
        else:
            self.model = self._load_model(model_path, stats_path)
        self.low_fi_stats = self._load_stats(low_fi_stats_path) if low_fi_stats_path else None
        self.stats = self._load_stats(stats_path)
        self.residual_stats = self._load_stats(residual_stats_path) if self.model_type == "low_fi_fusion" else None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "Outputs")
        self.out_path = os.path.join(outputs_dir, "predicted_output.csv")
        self.project_name = project_name
