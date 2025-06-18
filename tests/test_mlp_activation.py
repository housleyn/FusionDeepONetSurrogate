import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.MLP import MLP
from src.model.activations import RowdyActivation


def test_rowdy_activation_forward():
    act = RowdyActivation(3, base_fn=torch.relu)
    act.alpha.data.fill_(2.0)
    x = torch.tensor([[-1.0, 0.5, 2.0]])
    out = act(x)
    expected = 2.0 * torch.relu(x)
    assert torch.allclose(out, expected)


def test_mlp_forward_and_hidden():
    mlp = MLP(2, 3, 4, 2)
    x = torch.randn(5, 2)
    out = mlp(x)
    assert out.shape == (5, 3)
    out2, hidden = mlp.forward_with_outputs(x)
    assert torch.allclose(out, out2)
    assert len(hidden) == 2
    assert hidden[0].shape[1] == 4
