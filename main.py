
from src.surrogate import Surrogate

if __name__ == "__main__":
    


    surrogate = Surrogate(config_path="configs/config_spheres_sdf.yaml")
    surrogate._infer_and_validate(file="Data/sphere_formation_data/sf0.csv", shape='spheres')


