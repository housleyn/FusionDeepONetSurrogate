from src.models.fusion_model import FusionDeepONet
from src.models.vanilla_model import VanillaDeepONet
from src.models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet
from src.dataloader import Data
from src.trainer import Trainer
from src.preprocess import Preprocess
from src.inference import Inference
from src.postprocess import Postprocess
import matplotlib.pyplot as plt
import torch 
import numpy as np
import os
import glob

class MethodsSurrogate:
    
    def _train(self):
        self._preprocess_data()
        self._load_data()
        self._create_model()
        self._train_model()
        
    
    def _preprocess_data(self):
        preprocess = Preprocess(files=self.files ,dimension=self.dimension, output_path=self.output_path, param_columns=self.param_columns, distance_columns=self.distance_columns, lhs_sample=self.lhs_sample)
        preprocess.run_all()
        if self.model_type == "low_fi_fusion":
            preprocess_low_fi = Preprocess(files=self.low_fi_files ,dimension=self.dimension, output_path=self.low_fi_output_path, param_columns=self.param_columns, distance_columns=self.distance_columns, lhs_sample=self.lhs_sample)
            preprocess_low_fi.run_all()
        print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        data_low_fi = Data(self.low_fi_output_path) if self.low_fi_output_path else None
        self.train_loader, self.test_loader = data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
        self.train_loader_low_fi, self.test_loader_low_fi = data_low_fi.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size) if data_low_fi else (None, None)
        print("Data loaded in dataloader.")

    def _create_model(self):

        if self.model_type == "vanilla":
            print("Using Vanilla DeepONet model.")
            self.model = VanillaDeepONet(self.coord_dim, self.param_dim, self.hidden_size, self.num_hidden_layers, self.output_dim)
        if self.model_type == "FusionDeepONet":
            print("Using Fusion DeepONet model.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim
            )
        if self.model_type == "low_fi_fusion":
            print("Using Low Fidelity Fusion DeepONet model.")
            print(f"coord_dim={self.coord_dim + self.distance_dim}")
            self.model = Low_Fidelity_FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                npz_path=self.low_fi_output_path
            )

    def _train_model(self):
        if self.model_type == "low_fi_fusion":
            trainer_low_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader_low_fi, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer_low_fi.train(self.train_loader_low_fi, self.test_loader_low_fi, self.num_epochs, print_every=self.print_every)
            trainer_low_fi.save_model(low_fi=True)
            self._plot_loss_history(low_fidelity=True)
            print("Training low_fidelity complete. Loss history and model saved.")
            print("Evaluating low_fidelity model on high_fidelity data...")
            
            residual_npz = os.path.join(self.project_root, "Outputs", self.project_name, "residual_ellipse_low2high.npz")
            self._make_residual_dataset(hf_npz_out=residual_npz)
            hf_data = Data(residual_npz)
            self.train_loader, self.test_loader = hf_data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
            print("Residual dataset created and loaded.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                aux_dim=self.output_dim  # <- pointwise u_LF(ξ) concat
            ).to(self.device)
            trainer_hi_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer_hi_fi.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
            trainer_hi_fi.save_model()
            self._plot_loss_history()
            print("Training high_fidelity complete. Loss history and model saved.")
        else:
            trainer = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
            trainer.save_model()
            self._plot_loss_history()
            print("Training complete. Loss history and model saved.")
    
    def _plot_loss_history(self, low_fidelity=False):
        plt.plot(self.loss_history, label='Training Loss')
        plt.plot(self.test_loss_history, label='Testing Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        if low_fidelity:
            self.loss_history_file_name = "loss_history_low_fidelity.png"
            plt.title("Low Fidelity Training and Testing Loss History")
        else:
            self.loss_history_file_name = "loss_history.png"
            plt.title("Training and Testing Loss History")
        plt.legend()
        plt.grid(True)
        fig_dir = os.path.join("Outputs",f"{self.project_name}")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, self.loss_history_file_name))
        plt.close()

    def _infer_and_validate(self, file, shape):
        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(
            self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            low_fi_stats_path = os.path.join(
                self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            inference = Inference(self.project_name, config_path=self.config_path, model_path=self.model_path, stats_path=stats_path, low_fi_stats_path=low_fi_stats_path, param_columns=self.param_columns, distance_columns=self.distance_columns, low_fi_model_path=self.low_fi_model_path if self.model_type=="low_fi_fusion" else None)
        else:
            stats_path = self.npz_path
            inference = Inference(self.project_name, config_path=self.config_path, model_path=self.model_path, stats_path=stats_path, param_columns=self.param_columns, distance_columns=self.distance_columns, low_fi_model_path=self.low_fi_model_path if self.model_type=="low_fi_fusion" else None)
        coords_np, params_np, sdf_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params, sdf_np)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        print(f"Inference complete. Output saved to {self.predicted_output_file}.")
        print("Beginning postprocessing...")
        postprocess = Postprocess(self.project_name, path_true=file, path_pred=self.predicted_output_file, param_columns=self.param_columns)
        postprocess.run(self.dimension, shape)
    
    def _inference(self, file):
        inference = Inference(self.project_name,config_path=self.config_path, model_path=self.model_path, stats_path=self.npz_path, param_columns=self.param_columns, distance_columns=self.distance_columns)
        coords_np, params_np, sdf_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        print(f"Inference complete. Output saved to {self.predicted_output_file}.")
        print("Beginning postprocessing...")
        postprocess = Postprocess(self.project_name, path_true=None, path_pred=self.predicted_output_file, param_columns=self.param_columns)
        postprocess._plot_predicted_only(params)

        
    def _get_data_files(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    
    def _get_low_fi_data_files(self):
        if self.low_fi_data_folder is None:
            return []
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.low_fi_data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    

    def _make_residual_dataset(self, hf_npz_out):
        # 1) Load HF data loader (already built in _load_data)
        hf_loader = self.train_loader  # or build a combined loader incl. test if you prefer

        # 2) Load LF model once
        lf_model = Low_Fidelity_FusionDeepONet(
            coord_dim=self.coord_dim + self.distance_dim,
            param_dim=self.param_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=self.output_dim,
            npz_path=self.low_fi_output_path  # stats for LF
        ).to(self.device)
        lf_model.load_state_dict(torch.load(self.low_fi_model_path, map_location=self.device))
        lf_model.eval()

        # 3) Accumulators
        all_coords, all_params, all_sdf = [], [], []
        all_uHF, all_uLF, all_residual = [], [], []

        with torch.no_grad():
            for batch in hf_loader:
                # Expect dataloader to yield: (coords, params, targets, sdf)
                if isinstance(batch, dict):
                    coords = batch["coords"].to(self.device)
                    params = batch["params"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    sdf    = batch["sdf"].to(self.device)
                   
                else:
                    coords, params, targets, sdf = [t.to(self.device) for t in batch]

                # LF prediction at HF coordinates
                # Low_Fidelity_FusionDeepONet expects (coords, params, sdf)
                u_lf = lf_model(coords, params, sdf)          # [B, N, n_fields] or [B*N, n_fields] depending on your model
                # Align shapes
                if u_lf.shape != targets.shape:
                    # flatten/reshape if needed
                    u_lf = u_lf.view_as(targets)

                # Residual
                r = targets - u_lf

                # Move to CPU numpy and store
                all_coords.append(coords.detach().cpu().numpy())
                all_params.append(params.detach().cpu().numpy())
                all_sdf.append(sdf.detach().cpu().numpy())
                all_uHF.append(targets.detach().cpu().numpy())
                all_uLF.append(u_lf.detach().cpu().numpy())
                all_residual.append(r.detach().cpu().numpy())

        # 4) Concatenate over batches and save one NPZ
        def cat(lst): 
            arr = np.concatenate(lst, axis=0) if len(lst) > 1 else lst[0]
            return arr

        np.savez_compressed(
            hf_npz_out,
            coords=cat(all_coords),
            params=cat(all_params),
            sdf=cat(all_sdf),
            outputs=cat(all_residual),
            aux_lf_pointwise=cat(all_uLF),
            targets_highfi=cat(all_uHF),      # optional: keep HF too
        )
        print(f"Residual dataset written to: {hf_npz_out}")
