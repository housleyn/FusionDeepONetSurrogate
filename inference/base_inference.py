"""Base functionality for performing inference with a trained model."""

import torch


class BaseInference:
    def __init__(self, model_path="fusion_deeponet.pt", stats_path="processed_data.npz", device="cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.stats = self._load_stats(stats_path)
        self.model.to(self.device)

