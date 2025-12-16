import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from src.dataloader import Data
from src.inference import Inference
from src.models.fusion_model import FusionDeepONet
from src.models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet
from src.models.vanilla_model import VanillaDeepONet
from src.postprocess import Postprocess
from src.preprocess import Preprocess
from src.trainer import Trainer
from src.utils.distributed import barrier, is_main_process


class MethodsSurrogate:
    def _train(self):
        self._preprocess_data()
        self._load_data()
        self._create_model()
        self._train_model()

    def _preprocess_data(self):
        preprocess = Preprocess(
            files=self.files,
            dimension=self.dimension,
            output_path=self.output_path,
            param_columns=self.param_columns,
            distance_columns=self.distance_columns,
            lhs_sample=self.lhs_sample,
        )
        preprocess.run_all()
        if self.model_type == "low_fi_fusion":
            preprocess_low_fi = Preprocess(
                files=self.low_fi_files,
                dimension=self.dimension,
                output_path=self.low_fi_output_path,
                param_columns=self.param_columns,
                distance_columns=self.distance_columns,
                lhs_sample=self.lhs_sample,
            )
            preprocess_low_fi.run_all()
        if is_main_process():
            print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        self.train_loader, self.test_loader, self.train_sampler = data.get_dataloader(
            self.batch_size,
            shuffle=self.shuffle,
            test_size=self.test_size,
            ddp=dist.is_initialized(),
            world_size=self.ddp_info.get("world_size", 1),
            rank=self.ddp_info.get("rank", 0),
        )

        if self.model_type == "low_fi_fusion" and self.low_fi_output_path:
            data_low_fi = Data(self.low_fi_output_path)
            self.train_loader_low_fi, self.test_loader_low_fi, self.train_sampler_low_fi = data_low_fi.get_dataloader(
                self.batch_size,
                shuffle=self.shuffle,
                test_size=self.test_size,
                ddp=dist.is_initialized(),
                world_size=self.ddp_info.get("world_size", 1),
                rank=self.ddp_info.get("rank", 0),
            )
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
        elif self.model_type == "FusionDeepONet":
            if is_main_process():
                print("Using Fusion DeepONet model.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
            )
        elif self.model_type == "low_fi_fusion":
            if is_main_process():
                print("Using Low Fidelity Fusion DeepONet model.")
            self.model = Low_Fidelity_FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                npz_path=self.low_fi_output_path,
                dropout=self.low_fi_dropout,
            )
        else:
            raise ValueError(f"Unknown model_type={self.model_type}")

        self.model = self.model.to(self.device)

        if dist.is_initialized():
            ddp_kwargs = {
                "device_ids": [self.ddp_info.get("local_rank", 0)] if torch.cuda.is_available() else None,
                "output_device": self.ddp_info.get("local_rank", 0) if torch.cuda.is_available() else None,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
            }
            if not torch.cuda.is_available():
                ddp_kwargs.pop("device_ids")
                ddp_kwargs.pop("output_device")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)

    def _train_model(self):
        if self.model_type == "low_fi_fusion":
            trainer_low_fi = Trainer(
                project_name=self.project_name,
                model=self.model,
                dataloader=self.train_loader_low_fi,
                device=self.device,
                lr=self.lr,
                lr_gamma=self.lr_gamma,
                loss_type=self.loss_type,
                train_sampler=self.train_sampler_low_fi,
            )
            self.loss_history, self.test_loss_history = trainer_low_fi.train(
                self.train_loader_low_fi, self.test_loader_low_fi, self.num_epochs, print_every=self.print_every
            )
            trainer_low_fi.save_model(low_fi=True)
            if is_main_process():
                self._plot_loss_history(low_fidelity=True)
                print("Training low_fidelity complete. Loss history and model saved.")
                print("Evaluating low_fidelity model on high_fidelity data...")

            residual_npz = os.path.join(self.project_root, "Outputs", self.project_name, "residual.npz")
            if is_main_process():
                if os.path.exists(residual_npz):
                    os.remove(residual_npz)
                tmp = residual_npz + ".tmp.npz"
                if os.path.exists(tmp):
                    os.remove(tmp)

                self._make_residual_dataset(
                    hf_npz_out=residual_npz,
                    low_fi_stats_path=self.low_fi_output_path,
                    high_fi_stats_path=self.npz_path,
                )
            barrier()
            residual = Data(residual_npz)
            res_train_loader, res_test_loader, res_train_sampler = residual.get_dataloader(
                self.batch_size,
                shuffle=self.shuffle,
                test_size=self.test_size,
                ddp=dist.is_initialized(),
                world_size=self.ddp_info.get("world_size", 1),
                rank=self.ddp_info.get("rank", 0),
            )
            if is_main_process():
                print("Residual dataset created and loaded.")
            self.model = FusionDeepONet(
                coord_dim=self.coord_dim + self.distance_dim,
                param_dim=self.param_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                out_dim=self.output_dim,
                aux_dim=self.output_dim,
                dropout=self.dropout,
            ).to(self.device)
            if dist.is_initialized():
                ddp_kwargs = {
                    "device_ids": [self.ddp_info.get("local_rank", 0)] if torch.cuda.is_available() else None,
                    "output_device": self.ddp_info.get("local_rank", 0) if torch.cuda.is_available() else None,
                    "find_unused_parameters": False,
                    "gradient_as_bucket_view": True,
                }
                if not torch.cuda.is_available():
                    ddp_kwargs.pop("device_ids")
                    ddp_kwargs.pop("output_device")
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
            trainer_hi_fi = Trainer(
                project_name=self.project_name,
                model=self.model,
                dataloader=res_train_loader,
                device=self.device,
                lr=self.lr,
                lr_gamma=self.lr_gamma,
                loss_type=self.loss_type,
                train_sampler=res_train_sampler,
            )
            self.loss_history, self.test_loss_history = trainer_hi_fi.train(
                res_train_loader, res_test_loader, self.num_epochs, print_every=self.print_every
            )
            trainer_hi_fi.save_model()
            if is_main_process():
                self._plot_loss_history()
                print("Training high_fidelity complete. Loss history and model saved.")
        else:
            trainer = Trainer(
                project_name=self.project_name,
                model=self.model,
                dataloader=self.train_loader,
                device=self.device,
                lr=self.lr,
                lr_gamma=self.lr_gamma,
                loss_type=self.loss_type,
                train_sampler=self.train_sampler,
            )
            self.loss_history, self.test_loss_history = trainer.train(
                self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every
            )
            trainer.save_model()
            if is_main_process():
                self._plot_loss_history()
                print("Training complete. Loss history and model saved.")

    def _plot_loss_history(self, low_fidelity=False):
        plt.plot(self.loss_history, label="Training Loss")
        plt.plot(self.test_loss_history, label="Testing Loss")
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
        fig_dir = os.path.join("Outputs", f"{self.project_name}")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, self.loss_history_file_name))
        plt.close()

    def _infer_and_validate(self, file):
        if dist.is_available() and dist.is_initialized() and not is_main_process():
            return
        if self.model_type == "low_fi_fusion":
            stats_path = os.path.join(self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz")
            low_fi_stats_path = os.path.join(
                self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz"
            )
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                low_fi_stats_path=low_fi_stats_path,
                param_columns=self.param_columns,
                distance_columns=self.distance_columns,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None,
            )
        else:
            stats_path = self.npz_path
            inference = Inference(
                self.project_name,
                config_path=self.config_path,
                model_path=self.model_path,
                stats_path=stats_path,
                param_columns=self.param_columns,
                distance_columns=self.distance_columns,
                low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None,
            )
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
        inference = Inference(
            self.project_name,
            config_path=self.config_path,
            model_path=self.model_path,
            stats_path=self.npz_path,
            param_columns=self.param_columns,
            distance_columns=self.distance_columns,
        )
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
        surface_metrics = []

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if self.model_type == "low_fi_fusion":
                stats_path = os.path.join(self.project_root, "Outputs", self.project_name, "processed_low_fi_data.npz")
                low_fi_stats_path = os.path.join(
                    self.project_root,
                    "Outputs",
                    self.project_name,
                    "processed_low_fi_data.npz",
                )
                inference = Inference(
                    self.project_name,
                    config_path=self.config_path,
                    model_path=self.model_path,
                    stats_path=stats_path,
                    low_fi_stats_path=low_fi_stats_path,
                    param_columns=self.param_columns,
                    distance_columns=self.distance_columns,
                    low_fi_model_path=self.low_fi_model_path if self.model_type == "low_fi_fusion" else None,
                )
            else:
                stats_path = self.npz_path
                inference = Inference(
                    self.project_name,
                    config_path=self.config_path,
                    model_path=self.model_path,
                    stats_path=stats_path,
                    param_columns=self.param_columns,
                    distance_columns=self.distance_columns,
                )
            coords_np, params_np, sdf_np = inference.load_csv_input(file_path)
            params = params_np[1]
            output = inference.predict(coords_np, params, sdf_np) if self.model_type == "low_fi_fusion" else inference.predict(coords_np, params)
            inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
            if is_main_process():
                print(f"Inference complete for {filename}. Output saved to {self.predicted_output_file}.")
                print("Beginning postprocessing...")
            postprocess = Postprocess(self.project_name, path_true=file_path, path_pred=self.predicted_output_file, param_columns=self.param_columns)
            err, surface_metric = postprocess.run(self.dimension)
            errors.append(err)
            surface_metrics.append(surface_metric)
        return errors, surface_metrics
