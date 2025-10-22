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
            
            residual_npz = os.path.join(self.project_root, "Outputs", self.project_name, "residual.npz")
            self._make_residual_dataset(hf_npz_out=residual_npz,  low_fi_stats_path=self.low_fi_output_path, high_fi_stats_path=self.npz_path)
            residual = Data(residual_npz)
            res_train_loader, res_test_loader = residual.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
            print("Residual dataset created and loaded.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                aux_dim=self.output_dim  # <- pointwise u_LF(ξ) concat
            ).to(self.device)
            trainer_hi_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=res_train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
            self.loss_history, self.test_loss_history = trainer_hi_fi.train(res_train_loader, res_test_loader, self.num_epochs, print_every=self.print_every)
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
    

    def _make_residual_dataset(self, hf_npz_out, low_fi_stats_path=None, high_fi_stats_path=None):
        hf_train_loader = self.train_loader
        hf_test_loader  = self.test_loader

        # --- Load stats as tensors ---
        lf_stats = self._load_stats(low_fi_stats_path)
        hf_stats = self._load_stats(high_fi_stats_path)
        mu_lf, std_lf = lf_stats["outputs_mean"].to(self.device), lf_stats["outputs_std"].to(self.device)
        mu_hf, std_hf = hf_stats["outputs_mean"].to(self.device), hf_stats["outputs_std"].to(self.device)

        # --- Load LF model ---
        lf_model = Low_Fidelity_FusionDeepONet(
            coord_dim=self.coord_dim + self.distance_dim,
            param_dim=self.param_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=self.output_dim,
            npz_path=self.low_fi_output_path
        ).to(self.device)
        lf_model.load_state_dict(torch.load(self.low_fi_model_path, map_location=self.device))
        lf_model.eval()

        # --- Accumulators ---
        all_coords, all_params, all_sdf = [], [], []
        all_uHF, all_uLF, all_residual = [], [], []

        with torch.no_grad():
            for loader in [hf_train_loader, hf_test_loader]:
                for batch in loader:
                    if isinstance(batch, dict):
                        coords = batch["coords"].to(self.device)
                        params = batch["params"].to(self.device)
                        outputs = batch["outputs"].to(self.device)
                        sdf     = batch["sdf"].to(self.device)
                    else:
                        coords, params, outputs, sdf = [t.to(self.device) for t in batch]
                    targets = outputs  # HF normalized outputs

                    # ---- Denormalize both HF and LF ----
                    u_hf_denorm = targets * std_hf + mu_hf
                    u_lf = lf_model(coords, params, sdf)
                    if u_lf.shape != targets.shape:
                        u_lf = u_lf.view_as(targets)
                    u_lf_denorm = u_lf * std_lf + mu_lf

                    # ---- Residuals in denormalized space ----
                    r_denorm = u_hf_denorm - u_lf_denorm

                    # ---- Store (CPU numpy for saving) ----
                    all_coords.append(coords.cpu().numpy())
                    all_params.append(params.cpu().numpy())
                    all_sdf.append(sdf.cpu().numpy())
                    all_uHF.append(u_hf_denorm.cpu().numpy())
                    all_uLF.append(u_lf_denorm.cpu().numpy())
                    all_residual.append(r_denorm.cpu().numpy())

        # --- Concatenate and compute residual stats ---
        def cat(lst): return np.concatenate(lst, axis=0) if len(lst) > 1 else lst[0]

        residuals = cat(all_residual)       # shape → (num_samples, npts_max, output_dim)
        uLF = cat(all_uLF)                  # same
        uHF = cat(all_uHF)
        coords = cat(all_coords)
        params = cat(all_params)
        sdf = cat(all_sdf)

        # --- Compute mean/std over all spatial points (flatten sample/point dims) ---
        residuals_flat = residuals.reshape(-1, residuals.shape[-1])
        mu_r = torch.tensor(np.mean(residuals_flat, axis=0), dtype=torch.float32)
        std_r = torch.tensor(np.std(residuals_flat, axis=0) + 1e-8, dtype=torch.float32)

        # --- Normalize back in the same padded shape ---
        r_norm = (residuals - mu_r.numpy()) / std_r.numpy()

        # --- Save residual dataset in padded format ---
        np.savez_compressed(
            hf_npz_out,
            coords=coords,
            params=params,
            sdf=sdf,
            outputs=r_norm,
            aux_lf_pointwise=uLF,
            targets_highfi=uHF,
            outputs_mean=mu_r.numpy(),   # keep naming consistent with _load_stats()
            outputs_std=std_r.numpy()
        )
        print(f"Residual dataset written to: {hf_npz_out}")

    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

