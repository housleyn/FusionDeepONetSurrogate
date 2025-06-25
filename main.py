
from src.surrogate import Surrogate

if __name__ == "__main__":
    
    
    
    surrogate = Surrogate(config_path="configs/config_spheres_big_batch.yaml")
    surrogate._train()


