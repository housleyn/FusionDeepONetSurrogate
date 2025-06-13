import os
import numpy as np
import pandas as pd
import pytest

from preprocess import Preprocess
import preprocess.methods_preprocess as mp

# Use a small sample size for faster tests
TEST_SAMPLE_SIZE = 20


@pytest.fixture(autouse=True)
def patch_sampling(monkeypatch):
    def dummy_random(self, n):
        rng = np.random.default_rng(0)
        return rng.random((TEST_SAMPLE_SIZE, 3))

    class DummyNN:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, coords):
            self.n = coords.shape[0]
            return self
        def kneighbors(self, lhs):
            idx = np.arange(lhs.shape[0]) % self.n
            return None, idx.reshape(-1, 1)

    monkeypatch.setattr(mp.qmc.LatinHypercube, "random", dummy_random)
    monkeypatch.setattr(mp, "NearestNeighbors", DummyNN)


@pytest.fixture
def radius_file_dict(tmp_path):
    df = pd.DataFrame({
        "X (m)": np.linspace(0, 1, 10),
        "Y (m)": np.linspace(0, 1, 10),
        "Z (m)": np.linspace(0, 1, 10),
        "Density (kg/m^3)": np.linspace(1, 2, 10),
        "Velocity[i] (m/s)": np.linspace(0, 1, 10),
        "Velocity[j] (m/s)": np.linspace(0, 1, 10),
        "Velocity[k] (m/s)": np.linspace(0, 1, 10),
        "Absolute Pressure (Pa)": np.linspace(0, 1, 10),
    })
    mapping = {}
    for r in [0.2, 0.6, 1.0]:
        path = tmp_path / f"data_{r}.csv"
        df.to_csv(path, index=False)
        mapping[r] = str(path)
    return mapping


@pytest.fixture
def preprocess(radius_file_dict, tmp_path):
    return Preprocess(radius_files=radius_file_dict, dimension=3, output_path=str(tmp_path / "processed.npz"))

@pytest.fixture
def radius_file_dict_2d(tmp_path):
    df = pd.DataFrame({
        "X (m)": np.linspace(0, 1, 10),
        "Y (m)": np.linspace(0, 1, 10),
        "Z (m)": np.zeros(10),
        "Density (kg/m^3)": np.linspace(1, 2, 10),
        "Velocity[i] (m/s)": np.linspace(0, 1, 10),
        "Velocity[j] (m/s)": np.linspace(0, 1, 10),
        "Velocity[k] (m/s)": np.linspace(0, 1, 10),
        "Absolute Pressure (Pa)": np.linspace(0, 1, 10),
    })
    mapping = {}
    for r in [0.5]:
        path = tmp_path / f"data_{r}.csv"
        df.to_csv(path, index=False)
        mapping[r] = str(path)
    return mapping


def test_no_lhs_when_dimension_2(radius_file_dict_2d, tmp_path):
    p = Preprocess(radius_files=radius_file_dict_2d, dimension=2, output_path=str(tmp_path / "out.npz"))
    p.load_and_pad()
    coords = p.coords[0]
    z_column = coords[:, 2]
    assert p.lhs_applied is False

def test_initial_state(preprocess):
    assert preprocess.coords == []
    assert preprocess.radii == []
    assert preprocess.outputs == []
    assert preprocess.npts_max == 0


def test_load_and_pad(preprocess):
    preprocess.load_and_pad()
    assert len(preprocess.coords) == 3
    assert preprocess.npts_max == TEST_SAMPLE_SIZE
    for c, r, o in zip(preprocess.coords, preprocess.radii, preprocess.outputs):
        assert c.shape[0] == TEST_SAMPLE_SIZE
        assert r.shape[0] == TEST_SAMPLE_SIZE
        assert o.shape[0] == TEST_SAMPLE_SIZE


def test_to_numpy_and_save(preprocess):
    preprocess.run_all()
    assert os.path.exists(preprocess.output_path)
    data = np.load(preprocess.output_path)
    assert data["coords"].shape == (3, TEST_SAMPLE_SIZE, 3)
    assert data["outputs"].shape == (3, TEST_SAMPLE_SIZE, 5)
    assert data["params"].shape == (3, 1)

    # outputs should be normalized
    def check_norm(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        assert np.allclose(flat.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(flat.std(axis=0), 1, atol=1e-6)

    check_norm(data["outputs"])


def test_lhs_sampling_distribution(preprocess):
    preprocess.load_and_pad()
    coords = preprocess.coords[0]
    unpadded = coords[~np.all(coords == coords[0], axis=1)]
    assert unpadded.shape[0] >= 0.8 * TEST_SAMPLE_SIZE
    rng = unpadded.max(axis=0) - unpadded.min(axis=0)
    assert np.all(rng > 0.1)

def test_pad_functionality():
    p = Preprocess(radius_files={}, dimension=3, output_path="out.npz")
    p.npts_max = 5
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    padded = p._pad(arr)
    assert padded.shape == (5, 2)
    assert np.all(padded[3:] == arr[-1])


def test_normalize_function():
    p = Preprocess(radius_files={}, dimension=3, output_path="out.npz")
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm, mean, std = p._normalize(data)
    assert np.allclose(mean, [2.0, 3.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose(norm.mean(axis=0), 0.0)
    assert np.allclose(norm.std(axis=0), 1.0)


def test_lhs_applied_for_3d(preprocess):
    preprocess.load_and_pad()
    assert preprocess.lhs_applied is True
    for c in preprocess.coords:
        assert c.shape[0] == TEST_SAMPLE_SIZE


def test_to_numpy_shapes(preprocess):
    preprocess.load_and_pad()
    coords, outputs, params = preprocess.to_numpy()
    assert coords.shape == (3, TEST_SAMPLE_SIZE, 3)
    assert outputs.shape == (3, TEST_SAMPLE_SIZE, 5)
    assert params.shape == (3, 1)


def test_save_contains_statistics(preprocess, tmp_path):
    preprocess.load_and_pad()
    preprocess.output_path = str(tmp_path / "tmp.npz")
    preprocess.save()
    data = np.load(preprocess.output_path)
    for key in [
        "coords_mean",
        "coords_std",
        "outputs_mean",
        "outputs_std",
        "radii_mean",
        "radii_std",
    ]:
        assert key in data


def test_run_all_dimension_2(radius_file_dict_2d, tmp_path):
    p = Preprocess(radius_files=radius_file_dict_2d, dimension=2, output_path=str(tmp_path / "out2.npz"))
    p.run_all()
    data = np.load(p.output_path)
    assert data["coords"].shape[0] == 1
