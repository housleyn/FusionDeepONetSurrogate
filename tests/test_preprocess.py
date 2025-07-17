import os
import numpy as np
import pandas as pd
import pytest
import glob
import os


from src.preprocess import Preprocess
import src.preprocess.methods_preprocess as mp

def create_dummy_csv(tmp_dir, dim=2):
    cols = ["X (m)", "Y (m)", "Z (m)",
            "Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
            "Absolute Pressure (Pa)"]
    data = np.random.rand(10, len(cols))
    df = pd.DataFrame(data, columns=cols)
    if dim == 2:
        df["a"] = 1.0
        df["b"] = 2.0
    else:
        df["Sphere Radius"] = 0.5
    path = os.path.join(tmp_dir, f"dummy_{dim}d.csv")
    df.to_csv(path, index=False)
    return [path]

@pytest.fixture
def preprocess_instance2D(tmp_path):
    files = create_dummy_csv(tmp_path, dim=2)
    obj = Preprocess(files=files, dimension=2, output_path=str(tmp_path / "unittest_output.npz"), param_columns=["a", "b"], distance_columns=["distanceToEllipse"])
    return obj

@pytest.fixture
def preprocess_instance3D(tmp_path):
    files = create_dummy_csv(tmp_path, dim=3)
    obj = Preprocess(files=files, dimension=3, output_path=str(tmp_path / "unittest_output.npz"), param_columns=["Sphere Radius"], distance_columns=["distanceToSphere"])
    return obj


def test_preprocess_init_2D(preprocess_instance2D):
    assert isinstance(preprocess_instance2D, Preprocess)
    assert preprocess_instance2D.dimension == 2
    assert len(preprocess_instance2D.files) > 0
    assert preprocess_instance2D.output_path.endswith("unittest_output.npz")
    assert preprocess_instance2D.param_columns == ["a", "b"]
    assert len(preprocess_instance2D.coords) == 0
    assert len(preprocess_instance2D.params) == 0
    assert len(preprocess_instance2D.outputs) == 0
    assert preprocess_instance2D.npts_max == 0
    assert preprocess_instance2D.lhs_applied is False

def test_preprocess_init_3D(preprocess_instance3D):
    assert isinstance(preprocess_instance3D, Preprocess)
    assert preprocess_instance3D.dimension == 3
    assert len(preprocess_instance3D.files) > 0
    assert preprocess_instance3D.output_path.endswith("unittest_output.npz")
    assert preprocess_instance3D.param_columns == ["Sphere Radius"]
    assert len(preprocess_instance3D.coords) == 0
    assert len(preprocess_instance3D.params) == 0
    assert len(preprocess_instance3D.outputs) == 0
    assert preprocess_instance3D.npts_max == 0
    assert preprocess_instance3D.lhs_applied is False


def test_normalize_returns_correct_shape(preprocess_instance2D):
    data = np.random.rand(10, 5)
    normalized_data, mean, std = preprocess_instance2D._normalize(data)
    assert normalized_data.shape == data.shape
    assert mean.shape == (data.shape[1],)
    assert std.shape == (data.shape[1],)


def test_LHS_sampling_shape_and_bounds(preprocess_instance2D):
    pre = preprocess_instance2D
    pre.lhs_sample = 10

    coords = np.random.rand(100, 3) * 10
    outputs = np.random.rand(100, 5)
    sdf = np.random.rand(100, 1)

    sub_coords, sub_outputs, sub_sdf = pre.LHS(coords, outputs, sdf)

    assert sub_coords.shape == (10, 3)
    assert sub_outputs.shape == (10, 5)
    assert sub_sdf.shape == (10, 1)

def test_pad_correct_shape(preprocess_instance2D):
    pre = preprocess_instance2D
    pre.npts_max = 5
    arr = np.array([[1, 2], [3, 4]])

    padded = pre._pad(arr)
    assert padded.shape == (5, 2)
    for row in padded[2:]:
        np.testing.assert_array_equal(row, arr[-1])


def test_normalize_mean_std(preprocess_instance2D):
    pre = preprocess_instance2D
    data = np.random.rand(100, 3) * 10 + 5
    normed, mean, std = pre._normalize(data)

    np.testing.assert_allclose(np.mean(normed, axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(np.std(normed, axis=0), 1, atol=1e-6)


def test_to_numpy_shapes(preprocess_instance2D):
    pre = preprocess_instance2D
    pre.coords = [np.ones((4, 3)), np.ones((4, 3))]
    pre.outputs = [np.ones((4, 5)), np.ones((4, 5))]
    pre.params = [np.ones((4, 2)), np.ones((4, 2))]
    pre.sdf = [np.ones((4, 1)), np.ones((4, 1))]

    X, Y, G, S = pre.to_numpy()
    assert X.shape == (2, 4, 3)
    assert Y.shape == (2, 4, 5)
    assert G.shape == (2, 2)
    assert S.shape == (2, 4, 1)


def test_save_creates_file(tmp_path, preprocess_instance2D):
    pre = preprocess_instance2D
    pre.output_path = tmp_path / "test_output.npz"
    pre.coords = [np.ones((2, 3))]
    pre.outputs = [np.ones((2, 5))]
    pre.params = [np.ones((2, 2))]
    pre.sdf = [np.ones((2, 1))]
    pre.outputs_mean = np.zeros(5)
    pre.outputs_std = np.ones(5)

    pre.save()
    assert os.path.exists(pre.output_path)