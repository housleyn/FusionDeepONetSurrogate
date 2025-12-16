from src.surrogate import Surrogate

test = Surrogate(config_path="configs/parallel_test.yaml")
test._train()
test._infer_and_validate(file="Data/orion_data_32/orion_data_AoA1.253577597_Mach11.88065991.csv")
test._infer_all_unseen(folder="Data/orion_unseen_32")