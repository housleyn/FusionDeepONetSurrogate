import torch
import os
import yaml
class BaseSurrogate:
    def __init__(self, config_path, ddp_info=None):
        self.config_path = config_path
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.project_name = config["project_name"]
        self.data_folder = config["data_folder"]
        self.low_fi_data_folder = config.get("low_fi_data_folder", None)
        self.files = self._get_data_files()
        self.low_fi_files = self._get_low_fi_data_files() if self.low_fi_data_folder else None
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.output_dim = config["output_dim"]
        self.test_size = config["test_size"]
        self.coord_dim = config["coord_dim"]
        self.distance_dim = config["distance_dim"]
        self.distance_columns = config["distance_columns"]
        self.param_dim = config["param_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.print_every = config["print_every"]
        self.shuffle = config["shuffle"]
        self.dimension = config["dimension"]
        self.param_columns = config["param_columns"]
        self.loss_history_file_name = "loss_history.png"
        self.lhs_sample = config["lhs_sample"]
        self.lr = config["lr"]
        self.lr_gamma = config["lr_gamma"]
        self.model_type = config["model_type"]
        self.loss_type = config["loss_type"]
        self.low_fi_dropout = config.get("low_fi_dropout", 0.0)
        self.dropout = config.get("dropout", 0.0)
        self.dist_threshold = config.get("dist_threshold", 0.01)
        self.edge_percentile = config.get("edge_percentile", 99.5)
        self.x_lim = config.get("x_lim", None)
        self.y_lim = config.get("y_lim", None)


        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        os.makedirs(os.path.join(self.project_root, "Outputs", self.project_name), exist_ok=True)

        self.npz_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_data.npz"
        )
        self.output_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_data.npz"
        )
        self.low_fi_output_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
        )
        self.ddp_info = ddp_info or {"is_ddp": False, "rank": 0, "local_rank": 0, "world_size": 1}
        if self.ddp_info.get("is_ddp") and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.ddp_info.get('local_rank', 0)}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = self.ddp_info.get("rank", 0)
        if self.model_type == "low_fi_fusion":
            self.low_fi_model_path = f"Outputs/{self.project_name}/model/low_fi_fusion_deeponet.pt"
        self.model_path = f"Outputs/{self.project_name}/model/fusion_deeponet.pt"
        
        self.predicted_output_file = os.path.join(
            self.project_root, "Outputs", self.project_name, "predicted_output.csv"
        )
        