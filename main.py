import torch
from model import FusionDeepONet
from dataloader import get_dataloader
from trainer import Trainer
import matplotlib.pyplot as plt
from matplotlib import cm
from preprocess import Preprocess
import os
from inference import load_model, load_stats, predict, load_csv_input, save_to_csv, save_to_vtk
import pyvista as pv

def main():
    # === Configuration ===
    npz_path = "processed_data.npz"
    batch_size = 1
    num_epochs = 50000
    output_dim = 5  # u,v,w,rho, and p (not in that order)
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # === Load Data ===
    train_loader, test_loader = get_dataloader(npz_path, batch_size=batch_size, test_size=0.25)

    # === Create Model ===
    model = FusionDeepONet(
        coord_dim=3,
        param_dim=1,
        hidden_size=32,
        num_hidden_layers=3,
        out_dim=output_dim
    )

    # === Train Model ===
 
    trainer = Trainer(model, train_loader, device=device)
    loss_history, test_loss_history = trainer.train(train_loader, test_loader, num_epochs, print_every=1)

    # === Save Model ===
    trainer.save_model("fusion_deeponet.pt")

    print("✅ Training complete.")
    return loss_history, test_loss_history

def radius_file_dict():
        base_dir = os.path.dirname(__file__)
        return {
            0.2: os.path.join(base_dir, "sphere_data_02.csv"),
            0.3: os.path.join(base_dir, "sphere_data_03.csv"),
            0.4: os.path.join(base_dir, "sphere_data_04.csv"),
            0.5: os.path.join(base_dir, "sphere_data_05.csv"),
            0.6: os.path.join(base_dir, "sphere_data_06.csv"),
            0.7: os.path.join(base_dir, "sphere_data_07.csv"),
            0.8: os.path.join(base_dir, "sphere_data_08.csv"),
            0.9: os.path.join(base_dir, "sphere_data_09.csv"),
            1.0: os.path.join(base_dir, "sphere_data_1.csv"),
        }

if __name__ == "__main__":
    preprocess = Preprocess(radius_files=radius_file_dict(), output_path="processed_data.npz")
    preprocess.run_all()
    loss_history, test_loss_history = main()

    plt.semilogy(loss_history, label='Training Loss')
    plt.semilogy(test_loss_history, label='Testing Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Testing Loss History")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_history.png")
    # plt.show()


    # # Load model and stats
    # device = "cpu"
    # model = load_model("fusion_deeponet.pt", device)
    # stats = load_stats("processed_data.npz")

    # # Prepare input
    # coords_np, radius_val = load_csv_input("sphere_data_1.csv")

    # # Predict
    # output = predict(coords_np, radius_val, model, stats, device)

    # # output shape = (n_pts, 5)

    # # Save to CSV
    # save_to_csv(coords_np, output, radius_val, out_path="predicted_output.csv")

    # # Save to VTK for visualization
    # save_to_vtk(coords_np, output, out_path="predicted_output.vtk")

    # plotter = pv.Plotter()
    # plotter.add_mesh(pv.read("predicted_output.vtk"), scalars="density",cmap=cm.get_cmap("jet"), point_size=5, render_points_as_spheres=True)
    # plotter.show()