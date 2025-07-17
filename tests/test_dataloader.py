import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import Data


def create_npz(tmp_path):
    coords = np.random.rand(4, 5, 3).astype(np.float32)
    outputs = np.random.rand(4, 5, 5).astype(np.float32)
    params = np.random.rand(4, 1).astype(np.float32)
    sdf = np.random.rand(4, 5, 1).astype(np.float32)
    path = tmp_path / "sample.npz"
    np.savez(path, coords=coords, outputs=outputs, params=params, sdf=sdf)
    return path


def test_initialization(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert data.coords.shape == (4, 5, 3)
    assert data.outputs.shape == (4, 5, 5)
    assert data.params.shape == (4, 1)
    assert data.sdf.shape == (4, 5, 1)


def test_initialization_dtypes(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert data.coords.dtype == torch.float32
    assert data.outputs.dtype == torch.float32
    assert data.params.dtype == torch.float32
    assert data.sdf.dtype == torch.float32

def test_get_dataloader(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    batch_size = 1
    train_loader, test_loader = data.get_dataloader(batch_size=batch_size)

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    assert len(train_loader.dataset) == 3  # 80% of 4 samples
    assert len(test_loader.dataset) == 1  # 20% of 4 samples

    for batch in train_loader:
        coords, params, outputs, sdf = batch
        assert coords.shape == (batch_size, 5, 3)
        assert params.shape == (batch_size, 1)
        assert outputs.shape == (batch_size, 5, 5)
        assert sdf.shape == (batch_size, 5, 1)


