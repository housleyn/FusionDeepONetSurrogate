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
from src.utils.distributed import is_main_process 
import torch.distributed as dist 


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
        if is_main_process():
            print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        self.train_loader, self.test_loader, self.train_sampler = data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size, ddp=dist.is_initialized(), world_size=self.ddp_info.get("world_size",1), rank=self.ddp_info.get("rank",0))
        
        if self.model_type == "low_fi_fusion" and self.low_fi_output_path:
            data_low_fi = Data(self.low_fi_output_path)
            self.train_loader_low_fi, self.test_loader_low_fi, self.train_sampler_low_fi = data_low_fi.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size, ddp=dist.is_initialized(), world_size=self.ddp_info.get("world_size",1), rank=self.ddp_info.get("rank",0))
        else:
            self.train_loader_low_fi, self.test_loader_low_fi, self.train_sampler_low_fi = None, None, None
        if is_main_process():    
            print("Data loaded in dataloader.")
            print(f"train dataset size: {len(self.train_loader.dataset)}")
            print(f"steps per epoch (len(train_loader)): {len(self.train_loader)}")

    def _create_model(self):

        if self.model_type == "vanilla":
            if is_main_process():
                print("Using Vanilla DeepONet model.")
            self.model = VanillaDeepONet(self.coord_dim, self.param_dim, self.hidden_size, self.num_hidden_layers, self.output_dim)
        if self.model_type == "FusionDeepONet":
            if is_main_process():
                print("Using Fusion DeepONet model.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim
            )
        if self.model_type == "low_fi_fusion":
            if is_main_process():
                print("Using Low Fidelity Fusion DeepONet model.")
            self.model = Low_Fidelity_FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                npz_path=self.low_fi_output_path,
                dropout=self.low_fi_dropout
            )
        self.model = self.model.to(self.device)

        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = self.ddp_info["local_rank"]  # no fallback to 0
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        elif dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)


    def _train_model(self):
        if self.model_type == "low_fi_fusion":
            trainer_low_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader_low_fi, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type, train_sampler=self.train_sampler_low_fi)
            self.loss_history, self.test_loss_history = trainer_low_fi.train(self.train_loader_low_fi, self.test_loader_low_fi, self.num_epochs, print_every=self.print_every)
            trainer_low_fi.save_model(low_fi=True)
            if is_main_process():
                self._plot_loss_history(low_fidelity=True)
                print("Training low_fidelity complete. Loss history and model saved.")
                print("Evaluating low_fidelity model on high_fidelity data...")
            
            residual_npz = os.path.join(self.project_root, "Outputs", self.project_name, "residual.npz")
            if is_main_process():
                self._make_residual_dataset(hf_npz_out=residual_npz,  low_fi_stats_path=self.low_fi_output_path, high_fi_stats_path=self.npz_path)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            residual = Data(residual_npz)
            res_train_loader, res_test_loader, res_train_sampler = residual.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size, ddp=dist.is_initialized(), world_size=self.ddp_info.get("world_size",1), rank=self.ddp_info.get("rank",0))
            if is_main_process():
                print("Residual dataset created and loaded.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                aux_dim=self.output_dim,
                dropout=self.dropout
            ).to(self.device)
            if dist.is_initialized():
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.ddp_info.get("local_rank", 0)] if torch.cuda.is_available() else None,
                    output_device=self.ddp_info.get("local_rank", 0) if torch.cuda.is_available() else None,
                )
            trainer_hi_fi = Trainer(project_name=self.project_name, model=self.model, dataloader=res_train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type, train_sampler=res_train_sampler)
            self.loss_history, self.test_loss_history = trainer_hi_fi.train(res_train_loader, res_test_loader, self.num_epochs, print_every=self.print_every)
            trainer_hi_fi.save_model()
            if is_main_process():
                self._plot_loss_history()
                print("Training high_fidelity complete. Loss history and model saved.")
        else:
            trainer = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type, train_sampler=self.train_sampler)
            self.loss_history, self.test_loss_history = trainer.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
            trainer.save_model()
            if is_main_process():
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

    def _infer_and_validate(self, file):
        if dist.is_available() and dist.is_initialized() and not is_main_process():
            return
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
        if is_main_process():
            print(f"Inference complete. Output saved to {self.predicted_output_file}.")
            print("Beginning postprocessing...")
        postprocess = Postprocess(config_path=self.config_path, path_true=file, path_pred=self.predicted_output_file)
        postprocess.run(self.dimension)
    
    def _inference(self, file):
        if dist.is_available() and dist.is_initialized() and not is_main_process():
            return
        inference = Inference(self.project_name,config_path=self.config_path, model_path=self.model_path, stats_path=self.npz_path, param_columns=self.param_columns, distance_columns=self.distance_columns)
        coords_np, params_np, sdf_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        if is_main_process():
            print(f"Inference complete. Output saved to {self.predicted_output_file}.")
            print("Beginning postprocessing...")
        postprocess = Postprocess(self.project_name, path_true=None, path_pred=self.predicted_output_file, param_columns=self.param_columns)
        postprocess._plot_predicted_only(params)
    
    def _infer_all_unseen(self, folder):
        if dist.is_available() and dist.is_initialized() and not is_main_process():
            return
        errors = []
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
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
            coords_np, params_np, sdf_np = inference.load_csv_input(file_path)
            params = params_np[1]
            output = inference.predict(coords_np, params, sdf_np)
            predicted_output = os.path.join(self.project_root, "Outputs", self.project_name,"all_inferences", "predicted_"+filename)
            inference.save_to_csv(coords_np, output, out_path=predicted_output)
            if is_main_process():
                print(f"Inference complete. Output saved to {predicted_output}.")
            postprocess = Postprocess(config_path=self.config_path, path_true=file_path, path_pred=predicted_output)
            file_errors = dict(postprocess.get_errors())
            errors.append(file_errors)

        # Aggregate per-field error values across all files
        field_aggregates = {}
        for fe in errors:
            if not isinstance(fe, dict):
                continue
            for field, vals in fe.items():
                arr = np.asarray(vals).ravel()
                field_aggregates.setdefault(field, []).append(arr)

        # Prepare output directory
        plots_dir = os.path.join(self.project_root, "Outputs", self.project_name, "all_inference_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # --- Plot scatter for each field and save ---
        averages = {}  # store averages for later in all-in-one plot
        colors = plt.cm.tab10.colors

        for idx, (field, list_of_errors) in enumerate(field_aggregates.items()):
            # Each list_of_errors is a list of scalar L2 errors (floats)
            y = [float(e) for e in list_of_errors]
            x = np.arange(len(y))
            avg = np.mean(y)
            averages[field] = avg

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, s=80, color=colors[idx % len(colors)], edgecolors='black', alpha=0.8,
                        label=f"avg = {avg:.2f}%")
            plt.xlabel("Inferences")
            plt.ylabel("Relative L2 Error (%)")
            
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend( fontsize=9, title_fontsize=10)

            safe_field = "".join(c if (c.isalnum() or c in "._-") else "_" for c in field)
            out_path = os.path.join(plots_dir, f"{safe_field}_l2_errors.png")
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()

            print(f"Saved scatter for '{field}' -> {out_path}")

        # --- All-in-one summary plot with field averages ---
        plt.figure(figsize=(10, 6))

        for idx, (field, list_of_errors) in enumerate(field_aggregates.items()):
            y = [float(e) for e in list_of_errors]
            x = np.arange(len(y))
            avg = averages[field]
            plt.scatter(x, y, s=70, color=colors[idx % len(colors)], edgecolors='black',
                        alpha=0.8, label=f"{field} (avg = {avg:.2f}%)")

        plt.xticks(x, x+1)
        plt.xlabel("Inferences")
        plt.ylabel("Relative L2 Error (%)")
        plt.title("All Fields")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Fields and Averages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(plots_dir, "all_fields_l2_comparison.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved combined comparison plot with averages -> {out_path}")


        
    def _get_data_files(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    
    def _get_low_fi_data_files(self):
        if self.low_fi_data_folder is None:
            return []
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.low_fi_data_folder))
        return sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    

    def _make_residual_dataset(self, hf_npz_out, low_fi_stats_path, high_fi_stats_path):
        data_hf = Data(high_fi_stats_path)
        hf_train_loader, hf_test_loader, _ = data_hf.get_dataloader(self.batch_size, shuffle=False, test_size=self.test_size, ddp=False, world_size=1, rank=0)
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
        state = torch.load(self.low_fi_model_path, map_location=self.device)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        lf_model.load_state_dict(state, map_location=self.device)
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
                    assert u_lf.shape == targets.shape, \
                        f"LF/HF shape mismatch: lf {u_lf.shape}, hf {targets.shape}"
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
        uLF_norm = (uLF - mu_r.numpy()) / std_r.numpy()


        # --- Save residual dataset in padded format ---
        np.savez_compressed(
            hf_npz_out,
            coords=coords,
            params=params,
            sdf=sdf,
            outputs=r_norm,
            aux_lf_pointwise=uLF_norm,
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

