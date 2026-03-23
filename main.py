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
    
    
    

    

    

    # # 1. Create explorer
    # explorer = Explorer(
    #     output_dir="Explorer_outputs",
    #     total_points=30000,
    #     n_surface_each=180,
    #     seed=42,
    # )

    # # 2. Initialize model (ONLY YAML needed now)
    # explorer.initialize_model("configs/spheres_fusion.yaml")

    # # 3. Define motion path
    # path1 = explorer.build_linear_path(
    #     start=(2.75, 0.85),
    #     end=(8.0, 6.0),
    #     n_frames=40,
    # )

    # path2 = explorer.build_linear_path(
    #     start=(8.0, 6.0),
    #     end=(0.0, 3.0),
    #     n_frames=40,
    # )
    # positions = path1 + path2
    # # 4. Generate GIF of predicted pressure field
    # gif_path = explorer.make_prediction_gif(
    #     positions=positions,
    #     field_name="Absolute Pressure (Pa)",   # <-- change this to any field
    #     gif_name="pressure_field.gif",
    #     csv_subdir="csv_frames",
    #     pred_csv_subdir="predicted_csv_frames",
    #     plot_subdir="pressure_plot_frames",
    #     fps=10,
    #     levels=100,
    #     cmap="inferno",
    # )

    # print("Saved GIF to:", gif_path)


    
    from src.explorer.calculations_explorer import compute_cd_from_dataframe

    explorer = Explorer(
        project_name="spheres_case",
        output_dir="Explorer_outputs",
        total_points=30000,
        n_surface_each=180,
        seed=42,
    )

    # 1) Create one case
    explorer.create_single_case(
        x_secondary=11.0,
        y_secondary=10.0,
        csv_path="Explorer_outputs/single_case.csv",
    )

    # 2) Initialize model
    explorer.initialize_model(
        config_path="configs/spheres_fusion.yaml",
    )

    # 3) Predict
    pred_df, input_df = explorer.predict_from_csv(
        input_csv_path="Explorer_outputs/single_case.csv",
        output_csv_path="Explorer_outputs/predicted_single_case.csv",
    )

    # 4) Merge area/surface info from input into prediction df
    pred_df["is_on_surface"] = input_df["is_on_surface"].values
    pred_df["Area[i] (m^2)"] = input_df["Area[i] (m^2)"].values
    pred_df["Area[j] (m^2)"] = input_df["Area[j] (m^2)"].values
    pred_df["Area[k] (m^2)"] = input_df["Area[k] (m^2)"].values

    # 5) Keep only the second sphere surface rows
    n = explorer.n_surface_each
    secondary_df = pred_df.iloc[n:2*n].copy()

    # 6) Compute Cd for the second sphere
    cd, force_vector, surface_mask = compute_cd_from_dataframe(secondary_df)

    print("Cd on secondary sphere:", cd)

    # --- Plot single case for debugging ---

    # 1) Plot predicted pressure
    explorer.plot_predicted_field(
        pred_df=pred_df,
        input_df=input_df,
        field_name="Absolute Pressure (Pa)",
        output_plot="Explorer_outputs/debug_pressure.png",
        levels=100,
    )

    # 2) Plot density (optional sanity check)
    explorer.plot_predicted_field(
        pred_df=pred_df,
        input_df=input_df,
        field_name="Density (kg/m^3)",
        output_plot="Explorer_outputs/debug_density.png",
        levels=100,
    )

    # 3) Plot distance field (should always look correct)
    explorer.df = input_df
    explorer.is_surface = input_df["is_on_surface"].values
    explorer.surface_points = input_df[["X (m)", "Y (m)"]].values[:2*explorer.n_surface_each]

    explorer.plot_single_case_distance(
        plot_name="debug_distance.png"
    )

    print("Debug plots saved.")
    
