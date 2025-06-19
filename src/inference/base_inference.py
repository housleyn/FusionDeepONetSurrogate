import torch
import os
class BaseInference:
    def __init__(self, project_name, model_path="src/model/fusion_deeponet.pt", stats_path="Data/processed_data.npz", param_columns=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.stats = self._load_stats(stats_path)
        self.param_columns = param_columns
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "Outputs")
        self.out_path = os.path.join(outputs_dir, "predicted_output.csv")
        self.project_name = project_name
