import torch 


def gradient_weighted_value_loss(pred, target, coords, pressure_idx=-1, eps=1e-8, max_weight=1.75):

    pressure = pred[..., pressure_idx]

    grad_outputs = torch.ones_like(pressure)
    grads = torch.autograd.grad(
        pressure,
        coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    grad_xy = grads[..., :2]
    grad_magnitude = torch.sqrt(torch.sum(grad_xy**2, dim=-1) + eps)

    # Normalize and clamp to emphasize shocks
    weight = grad_magnitude / (grad_magnitude.mean(dim=1, keepdim=True) + eps)
    weight = torch.clamp(weight, min=1.0, max=max_weight)
    # print(f"Weight stats: min={weight.min().item()}, max={weight.max().item()}, mean={weight.mean().item()}")
    

    # Apply weight to squared error
    squared_error = (pred - target)**2  # shape (B, N, O)
    weighted_mse = weight.unsqueeze(-1) * squared_error  # shape (B, N, O)

    loss = weighted_mse.mean()
    return loss, weight.detach()