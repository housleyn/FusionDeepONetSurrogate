
from src.surrogate import Surrogate

if __name__ == "__main__":
    


    surrogate = Surrogate(config_path="configs/config_spheres_sdf.yaml")
    surrogate._train()


