import torch
from model import FusionDeepONet
from dataloader import get_dataloader
from trainer import Trainer
import matplotlib.pyplot as plt
from preprocess import Preprocess
import os

def main():
    # === Configuration ===
    npz_path = "processed_data.npz"
    batch_size = 3
    num_epochs = 5
    output_dim = 5  # u,v,w,rho, and p (not in that order)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load Data ===
    dataloader = get_dataloader(npz_path, batch_size=batch_size, shuffle=True)

    # === Create Model ===
    model = FusionDeepONet(
        coord_dim=3,
        param_dim=1,
        hidden_size=64,
        num_hidden_layers=3,
        out_dim=output_dim
    )

    # === Train Model ===
    trainer = Trainer(model, dataloader, device=device, lr=1e-3)
    loss_history = trainer.train(num_epochs=num_epochs, print_every=1)

    # === Save Model ===
    trainer.save_model("fusion_deeponet.pt")

    print("✅ Training complete.")
    return loss_history

def radius_file_dict():
        base_dir = os.path.dirname(__file__)
        return {
            0.2: os.path.join(base_dir, "sphere_data_02.csv"),
            0.6: os.path.join(base_dir, "sphere_data_06.csv"),
            1.0: os.path.join(base_dir, "sphere_data_1.csv")
        }

if __name__ == "__main__":
    preprocess = Preprocess(radius_files=radius_file_dict(), output_path="processed_data.npz")
    preprocess.run_all()
    loss_history = main()

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.show()
