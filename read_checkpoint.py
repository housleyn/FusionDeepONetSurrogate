import torch
from collections import OrderedDict

def get_state_dict(obj):
    # Case 1: full model
    if hasattr(obj, "parameters") and callable(obj.parameters):
        return obj.state_dict()
    
    # Case 2: pure state_dict
    if isinstance(obj, (dict, OrderedDict)) and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    
    # Case 3: checkpoint dict with model state
    for key in ["state_dict", "model_state_dict", "model"]:
        if isinstance(obj, dict) and key in obj:
            candidate = obj[key]
            if isinstance(candidate, (dict, OrderedDict)):
                return candidate
    
    raise ValueError("Could not find a state_dict in this checkpoint.")

def count_trainable_params(checkpoint_path):
    obj = torch.load(checkpoint_path, map_location="cpu")
    sd = get_state_dict(obj)

    # Assume all parameters in state_dict are trainable (common case)
    trainable_params = sum(t.numel() for t in sd.values())
    return trainable_params

# ===== USAGE =====
path = "Outputs/single_waverider_multi/model/low_fi_fusion_deeponet.pt"
print("Trainable parameters:", f"{count_trainable_params(path):,}")
