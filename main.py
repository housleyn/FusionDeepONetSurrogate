
from src.surrogate import Surrogate

if __name__ == "__main__":
    
    
    
    surrogate = Surrogate(config_path="configs/config_ellipse_super_computer.yaml")
    surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_unseen2.csv", shape="ellipse")
    



