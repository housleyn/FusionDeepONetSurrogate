from src.surrogate import Surrogate




if __name__ == "__main__":


    # ellipse_fusion = Surrogate(config_path="configs/ellipse_Fusion.yaml")
    # ellipse_fusion._train()
    # ellipse_fusion._infer_and_validate(file="Data/ellipse_data/ellipse_data_test2.csv")
    # ellipse_fusion._infer_all_unseen(folder="Data/unseen_ellipse_data")

    # ellipse_sequential = Surrogate(config_path="configs/ellipse_Multi.yaml")
    # ellipse_sequential._train()
    # ellipse_sequential._infer_and_validate(file="Data/ellipse_data/ellipse_data_test2.csv")
    # ellipse_sequential._infer_all_unseen(folder="Data/unseen_ellipse_data")

    # orion = Surrogate(config_path="configs/orion_sweep_5.yaml")
    # orion._train()
    # orion._infer_and_validate(file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv")
    # orion._infer_all_unseen(folder="Data/orion_unseen_32")


    x43 = Surrogate(config_path="configs/x_43_sweep_23.yaml")
    x43._train()
    x43._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")
    x43._infer_all_unseen(folder="Data/x_43_unseen_20")

    # x43_vanilla = Surrogate(config_path="configs/x_43_vanilla.yaml")
    # x43_vanilla._train()
    # x43_vanilla._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")
    # x43_vanilla._infer_all_unseen(folder="Data/x_43_unseen_20")


    # x43_fusion = Surrogate(config_path="configs/x_43_fusion.yaml")
    # x43_fusion._train()
    # x43_fusion._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")
    # x43_fusion._infer_all_unseen(folder="Data/x_43_unseen_20")

    # spheres = Surrogate(config_path="configs/spheres_sweep_10.yaml")
    # spheres._train()
    # spheres._infer_and_validate(file="Data/spheres_data/sf_x-0.015747631_y-2.700769227Mach15.81325245.csv")
