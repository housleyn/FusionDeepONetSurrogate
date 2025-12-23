from src.models.fusion_model import FusionDeepONet
from src.models.vanilla_model import VanillaDeepONet
from src.models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet
from src.dataloader import Data
from src.trainer import Trainer
from src.preprocess import Preprocess
import torch 
import numpy as np
import os
import glob
from .residual_calculations import (make_residual_dataset)
from .inference_functions import (infer_and_validate, inference, infer_all_unseen)
from .plotting_surrogate import (plot_loss_history)

class MethodsSurrogate:
    
    def _train(self):
        self._preprocess_data()
        self._load_data()
        self._create_model()
        self._train_model()
        
    def _preprocess_data(self):
        preprocess = Preprocess(files=self.files, output_path=self.output_path, param_columns=self.param_columns, distance_columns=self.distance_columns)
        preprocess.run_all(overwrite=self.overwrite)
        if self.model_type == "low_fi_fusion":
            preprocess_low_fi = Preprocess(files=self.low_fi_files, output_path=self.low_fi_output_path, param_columns=self.param_columns, distance_columns=self.distance_columns)
            preprocess_low_fi.run_all(overwrite=self.overwrite)
        print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        self.train_loader, self.test_loader = data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
        if self.model_type == "low_fi_fusion" and self.low_fi_output_path:
            data_low_fi = Data(self.low_fi_output_path)
            self.train_loader_low_fi, self.test_loader_low_fi = data_low_fi.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
        else:
            self.train_loader_low_fi, self.test_loader_low_fi = None, None
        print("Data loaded in dataloader.")

    def _create_model(self):
        if self.model_type == "vanilla":
            print("Using Vanilla DeepONet model.")
            self.model = VanillaDeepONet(self.coord_dim, self.param_dim, self.hidden_size, self.num_hidden_layers, self.output_dim)
        if self.model_type == "FusionDeepONet":
            print("Using Fusion DeepONet model.")
            self.model = FusionDeepONet(coord_dim=self.coord_dim + self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size, 
                                        num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim)
        if self.model_type == "low_fi_fusion":
            print("Using Low Fidelity Fusion DeepONet model.")
            self.model = Low_Fidelity_FusionDeepONet(coord_dim=self.coord_dim + self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size,
                                                     num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim, npz_path=self.low_fi_output_path, dropout=self.low_fi_dropout)

    def _train_model(self):
        if self.model_type == "low_fi_fusion":
            trainer_low_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader_low_fi, device=self.device, lr=self.lr, 
                                     lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer_low_fi.train(self.train_loader_low_fi, self.test_loader_low_fi, self.num_epochs, print_every=self.print_every)
            trainer_low_fi.save_model(low_fi=True)
            plot_loss_history(self, low_fidelity=True)
            print("Training low_fidelity complete. Loss history and model saved.")
            print("Evaluating low_fidelity model on high_fidelity data...")
            
            residual_npz = os.path.join(self.project_root, "Outputs", self.project_name, "residual.npz")
            make_residual_dataset(self, hf_npz_out=residual_npz,  low_fi_stats_path=self.low_fi_output_path, high_fi_stats_path=self.npz_path)
            residual = Data(residual_npz)
            res_train_loader, res_test_loader = residual.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
            print("Residual dataset created and loaded.")
            self.model = FusionDeepONet(coord_dim=self.coord_dim + self.distance_dim, param_dim=self.param_dim, hidden_size=self.hidden_size,
                                        num_hidden_layers=self.num_hidden_layers, out_dim=self.output_dim, aux_dim=self.output_dim, dropout=self.dropout).to(self.device)
            trainer_hi_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=res_train_loader, device=self.device, lr=self.lr, 
                                    lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer_hi_fi.train(res_train_loader, res_test_loader, self.num_epochs, print_every=self.print_every)
            trainer_hi_fi.save_model()
            plot_loss_history(self, low_fidelity=True)
            print("Training high_fidelity complete. Loss history and model saved.")
        else:
            trainer = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
            trainer.save_model()
            plot_loss_history(self)
            print("Training complete. Loss history and model saved.")
    
    def _infer_and_validate(self, file):
        infer_and_validate(self, file)
        
    def _inference(self, file):
        inference(self, file)  
    
    def _infer_all_unseen(self, folder):
        infer_all_unseen(self, folder)

    def _get_data_files(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    
    def _get_low_fi_data_files(self):
        if self.low_fi_data_folder is None:
            return []
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.low_fi_data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    
    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

