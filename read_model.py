import torch
import os

def inspect_model_state_dict(model_path):
    print(f"\nInspecting: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    for k, v in state_dict.items():
        print(f"{k}: {tuple(v.shape)}")

def find_model_files(output_dir):
    model_dir = os.path.join(output_dir, "model")
    if not os.path.isdir(model_dir):
        print(f"No model directory found in {output_dir}")
        return []
    return [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]

if __name__ == "__main__":
    # Change this to your desired output folder
    output_folder = "Outputs/low_fi_test2"
    model_files = find_model_files(output_folder)
    if not model_files:
        print("No model files found.")
    for model_path in model_files:
        inspect_model_state_dict(model_path)