import torch
from model import FusionDeepONet
from dataloader import get_dataloader
from trainer import Trainer

def main():
    # === Configuration ===
    npz_path = "test_processed_data.npz"
    batch_size = 3
    num_epochs = 5
    output_dim = 5  # or 5 if including pressure
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
    loss_history = trainer.train(num_epochs=num_epochs, print_every=100)

    # === Save Model ===
    trainer.save_model("fusion_deeponet.pt")

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
