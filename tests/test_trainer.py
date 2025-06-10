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
    mask = torch.ones(4, 5, dtype=torch.bool)
    dataset = torch.utils.data.TensorDataset(coords, params, outputs, mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    return loader


def test_trainer_single_epoch():
    loader = create_loaders()
    model = FusionDeepONet(3, 1, 8, 2, 5)
    trainer = Trainer(model, loader, device="cpu", lr=1e-3)
    train_hist, test_hist = trainer.train(loader, loader, num_epochs=1, print_every=1)
    assert len(train_hist) == len(test_hist) == 1
