from src.surrogate import Surrogate
from src.utils.distributed import setup_ddp, cleanup_ddp

ddp_info = setup_ddp()
if ddp_info["is_ddp"]:
    print(f"DDP Initialized | world_size={ddp_info['world_size']} | "
        f"rank={ddp_info['rank']} | local_rank={ddp_info['local_rank']}",
        flush=True)
try:

    test = Surrogate(config_path="configs/parallel_test.yaml", ddp_info=ddp_info)
    test._train()
    test._infer_and_validate(file="Data/orion_data_32/orion_data_AoA1.253577597_Mach11.88065991.csv")
    test._infer_all_unseen(folder="Data/orion_unseen_32")

finally:
    cleanup_ddp()
