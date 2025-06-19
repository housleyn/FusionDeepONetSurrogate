import torch
import os
import yaml
class BaseSurrogate:
    def __init__(self, config_path):
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.project_name = config["project_name"]
        self.data_folder = config["data_folder"]
        self.files = self._get_data_files()
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.output_dim = config["output_dim"]
        self.test_size = config["test_size"]
        self.coord_dim = config["coord_dim"]
        self.param_dim = config["param_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.print_every = config["print_every"]
        self.shuffle = config["shuffle"]
        self.dimension = config["dimension"]
        self.param_columns = config["param_columns"]
        self.loss_history_file_name = "loss_history.png"
        self.lhs_sample = config["lhs_sample"]

        self.npz_path = "Data/processed_data.npz"
        self.output_path = "Data/processed_data.npz"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = f"Outputs/{self.project_name}/model/fusion_deeponet.pt"
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.predicted_output_file = os.path.join(
            project_root, "Outputs", self.project_name, "predicted_output.csv"
        )