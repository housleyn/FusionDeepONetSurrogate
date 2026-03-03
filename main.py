from src.surrogate import Surrogate




if __name__ == "__main__":


   
    # orion = Surrogate(config_path="configs/orion_sequential.yaml")
    # orion._train()
    # orion._infer_and_validate(file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv")
    # orion._infer_all_unseen(folder="Data/orion_unseen_32")


    # x43 = Surrogate(config_path="configs/x_43_sequential.yaml")
    # x43._train()
    # x43._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")
    # x43._infer_all_unseen(folder="Data/x_43_unseen_20")

    spheres = Surrogate(config_path="configs/spheres_fusion.yaml")
    # spheres._train()
    # spheres._infer_and_validate(file="Data/spheres_data/sf_x9.778213128263568_y4.983763791274738.csv")
    spheres._infer_all_unseen(folder="Data/spheres_unseen_20")

    # waverider = Surrogate(config_path="configs/waverider.yaml")
    # waverider._train()
    # waverider._infer_and_validate(file="Data/waverider_hf_data/3D_slice_Mach_11.7700_z_-4.500000.csv")
    # waverider._infer_all_unseen(folder="Data/waverider_unseen")
    
