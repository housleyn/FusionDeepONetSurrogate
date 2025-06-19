import torch
import os
class BaseSurrogate:
    def __init__(self):

        self.project_name = "semi_ellipse1"
        self.data_folder = "Data/ellipse_data"
        self.files = self._get_data_files()
        self.npz_path = "Data/processed_data.npz"
        self.batch_size = 1
        self.num_epochs = 10
        self.output_dim = 5  # u,v,w,rho, and p (not in that order)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_size = 0.2
        self.coord_dim = 3
        self.param_dim = 2
        self.hidden_size = 32
        self.num_hidden_layers = 3
        self.print_every = 1
        self.shuffle = False
        self.dimension = 2  # 2D or 3D problem
        self.output_path = "Data/processed_data.npz"
        self.param_columns = ["a", "b"]
        
        self.loss_history_file_name = "loss_history_test.png"
        self.model_path = f"Outputs/{self.project_name}/model/fusion_deeponet.pt"
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.predicted_output_file = os.path.join(
            project_root, "Outputs",self.project_name, "predicted_output.csv"
        )
        self.lhs_sample = 500000
        
