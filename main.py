from src.surrogate import Surrogate
import os
import yaml
import itertools
import numpy as np



if __name__ == "__main__":


    surrogate = Surrogate(config_path="configs/config_ellipse_4.yaml")
    # surrogate._train()
    surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_test1.csv", shape='ellipse')
    # surrogate._infer_and_validate(file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv", shape=None)

    # surrogate = Surrogate(config_path="configs/low_fi_test.yaml")
    # surrogate._train()
    # surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_test2.csv", shape='ellipse')
    # surrogate._infer_and_validate(file="Data/x_43_unseen/x_43_a21.101756927_a324.59262374_a417.89264438.csv", shape=None)
    # surrogate._infer_and_validate(file="Data/x_43_data_36/x_43_a21.037862747_a320.23181912_a414.13846579.csv", shape=None)
    # surrogate._infer_and_validate(file="Data/x_43_low_fi_72/x_43_a21.8550107_a316.33293864_a410.25707196_low_fi.csv", shape=None)