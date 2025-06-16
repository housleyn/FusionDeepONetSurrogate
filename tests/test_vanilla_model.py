import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vanilla_model import VanillaDeepONet
import torch

@pytest.fixture
def model():
    return VanillaDeepONet(coord_dim=3, param_dim=1, hidden_size=64, num_hidden_layers=3, out_dim=5)

def test_forward_shape(model):
    coords = torch.rand(2, 100, 3)
    params = torch.rand(2, 1)
    out = model(coords, params)
    assert out.shape == (2, 100, 5)

def test_invalid_input_shape_raises(model):
    coords = torch.rand(2, 50, 2)
    params = torch.rand(2, 1)
    with pytest.raises(RuntimeError):
        model(coords, params)

def test_output_is_finite(model):
    coords = torch.rand(2, 50, 3)
    params = torch.rand(2, 1)
    out = model(coords, params)
    assert torch.isfinite(out).all()

def test_repeatable_output(model):
    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 1)
    out1 = model(coords, params)

    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 1)
    out2 = model(coords, params)

    assert torch.allclose(out1, out2, atol=1e-5)
