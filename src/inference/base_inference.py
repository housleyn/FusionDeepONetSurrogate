import torch
import os
import yaml
class BaseInference:
    def __init__(self, project_name, config_path ,model_path, stats_path="Data/processed_data.npz", param_columns=None, distance_columns=None):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.coord_dim = config["coord_dim"] + 1 #accounts for distance to surface being added to trunk
        self.param_dim = config["param_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.output_dim = config["output_dim"]
        
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param_columns = param_columns
        self.distance_columns = distance_columns
        self.model = self._load_model(model_path, stats_path)
        self.stats = self._load_stats(stats_path)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "Outputs")
        self.out_path = os.path.join(outputs_dir, "predicted_output.csv")
        self.project_name = project_name
