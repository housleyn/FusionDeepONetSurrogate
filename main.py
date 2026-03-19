from src.surrogate import Surrogate
from src.explorer import Explorer



if __name__ == "__main__":


   
    # orion = Surrogate(config_path="configs/orion_sequential.yaml")
    # orion._train()
    # orion._infer_and_validate(file="Data/orion_data/orion_data_AoA1.62290921_Mach13.06321187.csv")
    # orion._infer_all_unseen(folder="Data/orion_unseen_32")


    # x43 = Surrogate(config_path="configs/x_43_vanilla.yaml")
    # x43._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")
    # x432 = Surrogate(config_path="configs/x_43_multi.yaml")
    # x432._infer_and_validate(file="Data/x_43_data/x_43_a21.402224565_a311.43154317_a412.4105254.csv")
    # x433 = Surrogate(config_path="configs/x_43_sequential.yaml")
    # x433._infer_and_validate(file="Data/x_43_data/x_43_a21.402224565_a311.43154317_a412.4105254.csv")

    # spheres = Surrogate(config_path="configs/spheres_fusion.yaml")
    # spheres._train()
    # spheres._infer_and_validate(file="Data/spheres_unseen_20/sf_x1.0662246796756665_y4.335984435513547.csv")

    # waverider1 = Surrogate(config_path="configs/waverider_fusion.yaml")
    # waverider._train()
    # waverider1._infer_and_validate(file="Data/waverider_hf_data/3D_slice_Mach_7.2000_z_-3.085714.csv")
    # waverider1._infer_all_unseen(folder="Data/waverider_unseen")

    # waverider2 = Surrogate(config_path="configs/waverider_multi.yaml")
    # waverider2._infer_all_unseen(folder="Data/waverider_unseen")
    
    
    

    

    

    # 1. Create explorer
    explorer = Explorer(
        output_dir="Explorer_outputs",
        total_points=30000,
        n_surface_each=180,
        seed=42,
    )

    # 2. Initialize model (ONLY YAML needed now)
    explorer.initialize_model("configs/spheres_fusion.yaml")

    # 3. Define motion path
    path1 = explorer.build_linear_path(
        start=(2.75, 0.85),
        end=(8.0, 6.0),
        n_frames=40,
    )

    path2 = explorer.build_linear_path(
        start=(8.0, 6.0),
        end=(0.0, 3.0),
        n_frames=40,
    )
    positions = path1 + path2
    # 4. Generate GIF of predicted pressure field
    gif_path = explorer.make_prediction_gif(
        positions=positions,
        field_name="Absolute Pressure (Pa)",   # <-- change this to any field
        gif_name="pressure_field.gif",
        csv_subdir="csv_frames",
        pred_csv_subdir="predicted_csv_frames",
        plot_subdir="pressure_plot_frames",
        fps=10,
        levels=100,
        cmap="inferno",
    )

    print("Saved GIF to:", gif_path)
