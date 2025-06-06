import torch
from model import FusionDeepONet
from dataloader import Data
from trainer import Trainer
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from preprocess import Preprocess
import os
from inference import Inference
import pyvista as pv
import pandas as pd

def main():
    # === Configuration ===
    npz_path = "processed_data.npz"
    batch_size = 1
    num_epochs = 1
    output_dim = 5  # u,v,w,rho, and p (not in that order)
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # === Load Data ===
    data = Data(npz_path)
    train_loader, test_loader = data.get_dataloader(batch_size, shuffle=True, test_size=0.25)

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
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "3D_data"))
        
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

def csv_to_vtk(csv_path, vtk_path):
    df = pd.read_csv(csv_path)

    # Coordinates
    points = df[["X (m)", "Y (m)", "Z (m)"]].values
    cloud = pv.PolyData(points)

    # Vector field: Velocity
    velocity = df[["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"]].values
    cloud["velocity"] = velocity

    # Scalar fields: rename columns while adding
    rename_map = {
        "Absolute Pressure (Pa)": "pressure",
        "Density (kg/m^3)": "density",
    }

    for original, simplified in rename_map.items():
        cloud[simplified] = df[original].values

    # Save VTK
    cloud.save(vtk_path)

if __name__ == "__main__":
    preprocess = Preprocess(radius_files=radius_file_dict(), output_path="processed_data.npz")
    print("began training")
    preprocess.run_all()
    loss_history, test_loss_history = main()

    plt.semilogy(loss_history, label='Training Loss')
    plt.semilogy(test_loss_history, label='Testing Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Testing Loss History")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/loss_history.png")
    # plt.show()


    # # Load model and stats
    # file_to_load = "3D_data/sphere_data_075.csv"
    # file_to_make = "3D_data/sphere_data_075.vtk"
    # device = "cpu"
    # from inference import Inference
    # inference = Inference(model_path="fusion_deeponet.pt", stats_path="processed_data.npz", device=device)

    # # Prepare input
    # coords_np, radius_val = inference.load_csv_input(file_to_load)

    # # Predict
    # output = inference.predict(coords_np, radius_val)

    # # output shape = (n_pts, 5)

    # # Save to CSV
    # inference.save_to_csv(coords_np, output, radius_val, out_path="predicted_output.csv")

    # # Save to VTK for visualization
    # inference.save_to_vtk(coords_np, output, out_path="predicted_output.vtk")
    # csv_to_vtk(file_to_load, file_to_make)

    
    # mesh = pv.read("predicted_output.vtk")
    
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, scalars="density",style="points_gaussian", emissive=False , render_points_as_spheres=False, cmap=cm.get_cmap("jet"), clim=[0, 16.5])
    # plotter.show()


    
    # mesh = pv.read(file_to_make)
    # scalar_name = "density"  # change to "density" if needed

    # # Check if scalar exists
    # if scalar_name in mesh.point_data:
    #     data = mesh.point_data[scalar_name]
    #     print(f"{scalar_name} min: {data.min()}")
    #     print(f"{scalar_name} max: {data.max()}")
    # else:
    #     print(f"Scalar '{scalar_name}' not found in the VTK file.")
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, cmap=cm.get_cmap("jet"),clim=[0, 16.5])
    # plotter.show()


    #ADD CODE FOR ERROR CALCULATIONS IN INFERENCE CLASS