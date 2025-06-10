import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainer import Trainer
from model import FusionDeepONet


def create_loaders():
    coords = torch.rand(4, 5, 3)
    params = torch.rand(4, 1)
    outputs = torch.rand(4, 5, 5)
    dataset = torch.utils.data.TensorDataset(coords, params, outputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    return loader


def test_trainer_single_epoch():
    loader = create_loaders()
    model = FusionDeepONet(3, 1, 8, 2, 5)
    trainer = Trainer(model, loader, device="cpu", lr=1e-3)
    train_hist, test_hist = trainer.train(loader, loader, num_epochs=1, print_every=1)
    assert len(train_hist) == len(test_hist) == 1


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coords, params):
        batch, n, _ = coords.shape
        return torch.zeros(batch, n, 1) + self.w


def create_simple_loader():
    coords = torch.zeros(1, 4, 3)
    params = torch.zeros(1, 1)
    targets = torch.ones(1, 4, 1)
    dataset = torch.utils.data.TensorDataset(coords, params, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=1)


def test_evaluate():
    loader = create_simple_loader()
    model = DummyModel()
    trainer = Trainer(model, loader, device="cpu", lr=0.0)
    loss = trainer.evaluate(loader)
    expected = torch.mean((torch.zeros(4, 1) - torch.ones(4, 1)) ** 2)
    assert torch.isclose(torch.tensor(loss), expected)


def test_train_no_update():
    loader = create_simple_loader()
    model = DummyModel()
    trainer = Trainer(model, loader, device="cpu", lr=0.0)
    train_hist, _ = trainer.train(loader, loader, num_epochs=1, print_every=1)
    assert len(train_hist) == 1
    assert abs(train_hist[0] - trainer.evaluate(loader)) < 1e-6


def test_save_and_load_model(tmp_path):
    loader = create_loaders()
    model = DummyModel()
    trainer = Trainer(model, loader, device="cpu", lr=0.0)
    trainer.save_model(path=str(tmp_path / "m.pt"))
    model.w.data.fill_(5.0)
    trainer.load_model(str(tmp_path / "m.pt"))
    assert torch.allclose(model.w, torch.zeros(1))
