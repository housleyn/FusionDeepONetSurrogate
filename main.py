from src.surrogate import Surrogate




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

    spheres = Surrogate(config_path="configs/spheres_fusion.yaml")
    # spheres._train()
    spheres._infer_and_validate(file="Data/spheres_unseen_20/sf_x1.0662246796756665_y4.335984435513547.csv")

    # waverider1 = Surrogate(config_path="configs/waverider_fusion.yaml")
    # waverider._train()
    # waverider1._infer_and_validate(file="Data/waverider_hf_data/3D_slice_Mach_7.2000_z_-3.085714.csv")
    # waverider1._infer_all_unseen(folder="Data/waverider_unseen")

    # waverider2 = Surrogate(config_path="configs/waverider_multi.yaml")
    # waverider2._infer_all_unseen(folder="Data/waverider_unseen")
    
