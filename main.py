
from src.surrogate import Surrogate

if __name__ == "__main__":
    


    surrogate = Surrogate(config_path="configs/config_ellipse.yaml")
    surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_train5.csv", shape="ellipse")


