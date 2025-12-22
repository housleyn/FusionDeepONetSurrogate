from src.surrogate import Surrogate
import os
import yaml
import itertools
import numpy as np



if __name__ == "__main__":



    orion = Surrogate(config_path="configs/test.yaml")
    # orion._train()
    orion._infer_and_validate(file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv")

    # x43 = Surrogate(config_path="configs/x_43_sweep_23.yaml")
    # x43._infer_and_validate(file="Data/x_43_data/x_43_a21.150067306_a322.39658073_a44.094249377.csv")

    # spheres = Surrogate(config_path="configs/spheres_sweep_10.yaml")
    # spheres._infer_and_validate(file="Data/spheres_data/sf_x-0.015747631_y-2.700769227Mach15.81325245.csv")
