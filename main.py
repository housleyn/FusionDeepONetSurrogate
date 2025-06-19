import os
import glob
from src.surrogate import Surrogate

def sphere_file_list():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Data","sphere_data"))
    file_paths = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    return file_paths

def ellipse_file_list():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Data","ellipse_data"))
    file_paths = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    return file_paths

if __name__ == "__main__":
    
    
    ellipse_model = Surrogate(files=ellipse_file_list())
    # ellipse_model._train()
    ellipse_model._inference(file="Data/ellipse_data/ellipse_data_unseen2.csv")


    
    


    



    



    