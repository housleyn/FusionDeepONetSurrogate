from src.models.fusion_model import FusionDeepONet
from src.models.vanilla_model import VanillaDeepONet
from src.dataloader import Data
from src.trainer import Trainer
from src.preprocess import Preprocess
from src.inference import Inference
from src.postprocess import Postprocess
import matplotlib.pyplot as plt
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
        print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        self.train_loader, self.test_loader = data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
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

    def _train_model(self):
        trainer = Trainer(project_name=self.project_name, model=self.model, dataloader=self.train_loader, device=self.device, lr=self.lr, lr_gamma=self.lr_gamma, loss_type=self.loss_type)
        self.loss_history, self.test_loss_history = trainer.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
        trainer.save_model()
        self._plot_loss_history()
        print("Training complete. Loss history and model saved.")
    
    def _plot_loss_history(self):
        plt.semilogy(self.loss_history, label='Training Loss')
        plt.semilogy(self.test_loss_history, label='Testing Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Testing Loss History")
        plt.legend()
        plt.grid(True)
        fig_dir = os.path.join("Outputs",f"{self.project_name}")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, self.loss_history_file_name))
        plt.close()

    def _infer_and_validate(self, file, shape):
        inference = Inference(self.project_name, config_path=self.config_path, model_path=self.model_path, stats_path=self.npz_path, param_columns=self.param_columns, distance_columns=self.distance_columns)
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