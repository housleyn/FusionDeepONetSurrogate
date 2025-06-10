import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import Data


def create_npz(tmp_path):
    coords = np.random.rand(4, 5, 3).astype(np.float32)
    outputs = np.random.rand(4, 5, 5).astype(np.float32)
    params = np.random.rand(4, 1).astype(np.float32)
    mask = np.ones((4, 5), dtype=bool)
    path = tmp_path / "sample.npz"
    np.savez(path, coords=coords, outputs=outputs, params=params, mask=mask)
    return path


def test_initialization(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert data.coords.shape == (4, 5, 3)
    assert data.outputs.shape == (4, 5, 5)
    assert data.params.shape == (4, 1)
    assert data.mask.shape == (4, 5)


def test_len_and_getitem(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    assert len(data) == 4
    c, p, o, m = data[1]
    assert c.shape == (5, 3)
    assert p.shape == (1,)
    assert o.shape == (5, 5)
    assert m.shape == (5,)


def test_get_dataloader(tmp_path):
    npz = create_npz(tmp_path)
    data = Data(str(npz))
    train, test = data.get_dataloader(batch_size=2, shuffle=False, test_size=0.5)
    total = len(train.dataset) + len(test.dataset)
    assert total == len(data)
