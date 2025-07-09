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


def test_len_and_getitem(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert len(data) == 4
    c, p, o, s = data[1]
    assert c.shape == (5, 3)
    assert p.shape == (1,)
    assert o.shape == (5, 5)
    assert s.shape == (5, 1)


def test_get_dataloader(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    train, test = data.get_dataloader(batch_size=2, shuffle=False, test_size=0.5)
    total = len(train.dataset) + len(test.dataset)
    assert total == len(data)


def test_initialization_dtypes(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert data.coords.dtype == torch.float32
    assert data.outputs.dtype == torch.float32
    assert data.params.dtype == torch.float32
    assert data.sdf.dtype == torch.float32


def test_getitem_returns_correct_values(tmp_path):
    npz = create_npz(tmp_path)
    raw = np.load(npz)
    data = Data(str(npz))
    idx = 2
    c, p, o, s = data[idx]
    assert np.allclose(c.numpy(), raw['coords'][idx])
    assert np.allclose(o.numpy(), raw['outputs'][idx])
    assert np.allclose(p.numpy(), raw['params'][idx])
    assert np.allclose(s.numpy(), raw['sdf'][idx])


def test_get_dataloader_split_sizes(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    train, test = data.get_dataloader(batch_size=1, shuffle=False, test_size=0.25)
    assert len(train.dataset) + len(test.dataset) == len(data)
    assert len(test.dataset) == int(len(data) * 0.25)
    assert len(next(iter(train))[0]) == 1
