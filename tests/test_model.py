import pytest 
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import FusionDeepONet, MLP 
import torch


@pytest.fixture
def model():
    coord_dim = 3
    param_dim = 1
    hidden_size = 64
    num_hidden_layers = 3
    out_dim = 5
    return FusionDeepONet(coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim)

def test_forward_shape(model):
    batch_size = 2
    n_pts = 100
    coord_dim = 3
    param_dim = 1
    out_dim = 5

    coords = torch.rand(batch_size, n_pts, coord_dim)
    params = torch.rand(batch_size, param_dim)

    out = model(coords, params)

    assert out.shape == (batch_size, n_pts, out_dim), "Output shape mismatch"


def test_invalid_input_shape_raises(model):
    coords = torch.rand(2, 50, 2)  # Wrong coord_dim
    params = torch.rand(2, 1)
    with pytest.raises(RuntimeError):
        model(coords, params)


def test_branch_outputs_with_activations():
    branch = MLP(1, 320, 64, 3)
    x = torch.rand(4, 1)
    out, activations = branch.forward_with_activations(x)
    
    assert out.shape == (4, 320)
    assert len(activations) == 4
    for a in activations:
        assert a.shape == (4, 64)


def test_output_is_finite(model):
    coords = torch.rand(2, 50, 3)
    params = torch.rand(2, 1)
    out = model(coords, params)
    assert torch.isfinite(out).all(), "Model output contains NaNs or Infs"


def test_repeatable_output(model):
    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 1)
    out1 = model(coords, params)

    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 1)
    out2 = model(coords, params)

    assert torch.allclose(out1, out2, atol=1e-5), "Outputs not repeatable with fixed seed"


