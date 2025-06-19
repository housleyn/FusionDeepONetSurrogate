import torch
class BaseInference:
    def __init__(self, model_path="src/model/fusion_deeponet.pt", stats_path="Data/processed_data.npz", param_columns=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.stats = self._load_stats(stats_path)
        self.param_columns = param_columns
        self.out_path = "predicted_output.csv"