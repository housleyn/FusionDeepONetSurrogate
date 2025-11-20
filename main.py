from src.surrogate import Surrogate
import os
import yaml
import itertools
import numpy as np



if __name__ == "__main__":



    surrogate = Surrogate(config_path="configs/orion_sweep_0.yaml")

    surrogate._infer_and_validate(file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv")
