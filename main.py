
from src.surrogate import Surrogate

if __name__ == "__main__":
    
    
    ellipse_model = Surrogate(config_path="configs/config_ellipse.yaml")
    # ellipse_model._train()
    ellipse_model._infer_and_validate(file="Data/ellipse_data/ellipse_data_unseen2.csv")
    ellipse_model._inference(file="Data/ellipse_data/ellipse_data_unseen2.csv")


