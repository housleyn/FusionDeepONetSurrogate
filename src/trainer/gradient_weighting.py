import torch 


def gradient_weighted_value_loss(pred, target, coords, pressure_idx=-1, eps=1e-8):
    """Compute value loss weighted by pressure gradient magnitude.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions of shape (B, N, O).
    target : torch.Tensor
        Ground truth of shape (B, N, O).
    coords : torch.Tensor
        Input coordinates with ``requires_grad=True`` of shape (B, N, C).
    pressure_idx : int, optional
        Index of the pressure component in ``pred``/``target``. Defaults to -1
        which corresponds to the last channel.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Scalar weighted loss value.
    torch.Tensor
        Weight map used for the loss with shape (B, N).
    """
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

    if grad_magnitude.max() > eps:
        weight = torch.ones_like(grad_magnitude)
    else:
        weight = grad_magnitude / (grad_magnitude.mean(dim=1, keepdim=True) + eps)

    weighted_mse = weight.unsqueeze(-1)* (pred - target)**2
    loss = weighted_mse.mean()
    return loss, weight.detach()