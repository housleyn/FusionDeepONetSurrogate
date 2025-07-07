
from src.surrogate import Surrogate

if __name__ == "__main__":
    
    
    
    surrogate = Surrogate(config_path="configs/config_sphere_formation.yaml")
    surrogate._infer_and_validate('Data/sphere_formation_data/sf0.csv', 'spheres')


