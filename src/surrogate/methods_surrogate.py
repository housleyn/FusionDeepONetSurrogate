from src.model import FusionDeepONet
from src.dataloader import Data
from src.trainer import Trainer
from src.preprocess import Preprocess
from src.inference import Inference
from src.postprocess import Postprocess
import matplotlib.pyplot as plt
import os

class MethodsSurrogate:
    
    def _train(self):
        self._preprocess_data()
        self._load_data()
        self._create_model()
        self._train_model()
        
    
    def _preprocess_data(self):
        preprocess = Preprocess(files=self.files ,dimension=self.dimension, output_path=self.output_path, param_columns=self.param_columns, lhs_sample=self.lhs_sample)
        preprocess.run_all()
        print("Data preprocessing complete.")

    def _load_data(self):
        data = Data(self.npz_path)
        self.train_loader, self.test_loader = data.get_dataloader(self.batch_size, shuffle=self.shuffle, test_size=self.test_size)
        print("Data loaded in dataloader.")

    def _create_model(self):
        self.model = FusionDeepONet(
            coord_dim=self.coord_dim,
            param_dim=self.param_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=self.output_dim
        )
        print("Model created with specified architecture.")

    def _train_model(self):
        trainer = Trainer(self.model, self.train_loader, device=self.device)
        self.loss_history, self.test_loss_history = trainer.train(self.train_loader, self.test_loader, self.num_epochs, print_every=self.print_every)
        trainer.save_model(self.model_path)
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
        fig_dir = os.path.join("Tables_and_Figures", "figures", "loss_history")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, self.loss_history_file_name))
        plt.show()

    def _infer_and_validate(self, file):
        inference = Inference(model_path=self.model_path, stats_path=self.npz_path, param_columns=self.param_columns)
        coords_np, params_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        print(f"Inference complete. Output saved to {self.predicted_output_file}.")
        print("Beginning postprocessing...")
        postprocess = Postprocess(path_true=file, path_pred=self.predicted_output_file, param_columns=self.param_columns)
        postprocess.run()
    
    def _inference(self, file):
        inference = Inference(model_path=self.model_path, stats_path=self.npz_path, param_columns=self.param_columns)
        coords_np, params_np = inference.load_csv_input(file)
        params = params_np[1]
        output = inference.predict(coords_np, params)
        inference.save_to_csv(coords_np, output, out_path=self.predicted_output_file)
        print(f"Inference complete. Output saved to {self.predicted_output_file}.")
        print("Beginning postprocessing...")
        postprocess = Postprocess(path_true=None, path_pred=self.predicted_output_file, param_columns=self.param_columns)
        postprocess._plot_predicted_only()

        
