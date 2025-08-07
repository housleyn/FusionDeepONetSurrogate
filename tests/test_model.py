import pytest 
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import FusionDeepONet
from src.models.MLP import MLP
import torch


@pytest.fixture
def model():
    coord_dim = 3
    param_dim = 2
    hidden_size = 64
    num_hidden_layers = 3
    out_dim = 5
    return FusionDeepONet(coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim)

def test_forward_shape(model):
    batch_size = 2
    n_pts = 100
    coord_dim = 3
    param_dim = 2
    out_dim = 5

    coords = torch.rand(batch_size, n_pts, coord_dim)
    params = torch.rand(batch_size, param_dim)
    sdf = torch.rand(batch_size, n_pts, 1)

    out = model(coords, params, sdf)

    assert out.shape == (batch_size, n_pts, out_dim), "Output shape mismatch"


def test_invalid_input_shape_raises(model):
    coords = torch.rand(2, 50, 3)  # Wrong coord_dim
    params = torch.rand(2, 1)
    sdf = torch.rand(2, 50, 1)
    with pytest.raises(RuntimeError):
        model(coords, params, sdf)





def test_output_is_finite(model):
    coords = torch.rand(2, 50, 3)
    params = torch.rand(2, 2)
    sdf = torch.rand(2, 50, 1)
    out = model(coords, params, sdf)
    assert torch.isfinite(out).all(), "Model output contains NaNs or Infs"


def test_repeatable_output(model):
    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 2)
    sdf = torch.rand(1, 20, 1)
    out1 = model(coords, params, sdf)

    torch.manual_seed(42)
    coords = torch.rand(1, 20, 3)
    params = torch.rand(1, 2)
    sdf = torch.rand(1, 20, 1)
    out2 = model(coords, params, sdf)

    assert torch.allclose(out1, out2, atol=1e-5), "Outputs not repeatable with fixed seed"

def test_trunk_receives_distance(monkeypatch):
    model = FusionDeepONet(coord_dim=3 + 1, param_dim=2,
                           hidden_size=64, num_hidden_layers=2, out_dim=5)

    coords = torch.randn(2, 10, 3)
    params = torch.randn(2, 2)
    sdf = torch.randn(2, 10, 1)
    captured = {}

    def capture(x):
        captured["input"] = x.detach().clone()
        return torch.zeros(x.shape[0], x.shape[1], model.hidden_size)

    monkeypatch.setattr(model.trunk_layers[0], "forward", capture)
    model(coords, params, sdf)
    expected = torch.cat((coords, sdf), dim=-1)
    assert torch.allclose(captured["input"], expected)



