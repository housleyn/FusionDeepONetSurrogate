from src.surrogate import Surrogate
from src.utils.distributed import cleanup_ddp, setup_ddp



if __name__ == "__main__":


    ddp_info = setup_ddp()
    if ddp_info["is_ddp"]:
        print(f"DDP Initialized | world_size={ddp_info['world_size']} | "
            f"rank={ddp_info['rank']} | local_rank={ddp_info['local_rank']}",
            flush=True)
    try:
        
        parallel_test = Surrogate(config_path="configs/parallel_test.yaml", ddp_info=ddp_info)
        parallel_test._train()

        # orion = Surrogate(config_path="configs/orion_sweep_3.yaml")
        # orion._train()
	    


        # x43 = Surrogate(config_path="configs/x_43_sweep_23.yaml")
        # x43._infer_and_validate(file="Data/x_43_data/x_43_a21.037862747_a320.23181912_a414.13846579.csv")

        # spheres = Surrogate(config_path="configs/spheres_sweep_10.yaml")
        # spheres._infer_and_validate(file="Data/spheres_data/sf_x-0.015747631_y-2.700769227Mach15.81325245.csv")
        
        

    finally:
        cleanup_ddp()

    # x43 = Surrogate(config_path="configs/x_43_sweep_23.yaml")
    # x43._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")

    # spheres = Surrogate(config_path="configs/spheres_sweep_10.yaml")
    # spheres._infer_and_validate(file="Data/spheres_data/sf_x-0.015747631_y-2.700769227Mach15.81325245.csv")
    # orion = Surrogate(config_path="configs/orion_sweep_3.yaml")




