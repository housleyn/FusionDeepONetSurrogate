import pytest
import os 
import sys 
from preprocess import Preprocess
import numpy as np

@pytest.fixture
def radius_file_dict():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '3D_data'))
    return {
        0.2: os.path.join(base_dir, "sphere_data_02.csv"),
        0.6: os.path.join(base_dir, "sphere_data_06.csv"),
        1.0: os.path.join(base_dir, "sphere_data_1.csv")
    }
@pytest.fixture
def preprocess(radius_file_dict):
    return Preprocess(radius_files=radius_file_dict, output_path="test_processed_data.npz")

@pytest.fixture
def small_radius_file_dict():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return {
        0.2: os.path.join(base_dir, "sphere_data_02.csv")
    }

def test_radius_files_initialized_correctly(preprocess):
    assert len(preprocess.radius_files) == 3
    assert all(isinstance(radius, float) for radius in preprocess.radius_files.keys())

def test_output_path_initialized_correctly(preprocess):
    assert preprocess.output_path == "test_processed_data.npz"
    assert os.path.splitext(preprocess.output_path)[1] == ".npz"

def test_coords_initialized_empty(preprocess):
    assert len(preprocess.coords) == 0
    assert isinstance(preprocess.coords, list)

def test_radii_initialized_empty(preprocess):
    assert len(preprocess.radii) == 0
    assert isinstance(preprocess.radii, list)

def test_npts_max_initialized_zero(preprocess):
    assert preprocess.npts_max == 0
    assert isinstance(preprocess.npts_max, int)


def test_outputs_initialized_empty(preprocess):
    assert len(preprocess.outputs) == 0
    assert isinstance(preprocess.outputs, list)


def test_in_load_and_pad_for_loop_appends_correctly(preprocess):
    preprocess.load_and_pad()
    assert len(preprocess.coords) == 3
    assert len(preprocess.radii) == 3
    assert len(preprocess.outputs) == 3
    assert all(isinstance(coord, np.ndarray) for coord in preprocess.coords)
    assert all(isinstance(radius, np.ndarray) for radius in preprocess.radii)
    assert all(isinstance(output, np.ndarray) for output in preprocess.outputs)

def test_npts_max_updated_correctly(preprocess):
    preprocess.load_and_pad()
    assert preprocess.npts_max == 500000

def test_pad_set_all_sizes_to_max_points(preprocess):
    preprocess.load_and_pad()
    for coord, radius, output in zip(preprocess.coords, preprocess.radii, preprocess.outputs):
        assert coord.shape[0] == preprocess.npts_max
        assert radius.shape[0] == preprocess.npts_max
        assert output.shape[0] == preprocess.npts_max


def test_to_numpy_shapes_correct(preprocess):
    preprocess.load_and_pad()
    batch_size = 3 #3 sets of data, 3 simulations
    X_coords, Y_outputs, G_params = preprocess.to_numpy()
    assert X_coords.shape == (3, preprocess.npts_max, 3)
    assert Y_outputs.shape == (3, preprocess.npts_max, 5)
    assert G_params.shape == (3, 1)


def test_save_creates_file(preprocess):
    preprocess.load_and_pad()
    preprocess.save()
    assert os.path.exists(preprocess.output_path)
    data = np.load(preprocess.output_path)
    assert "coords" in data
    assert "outputs" in data
    assert "params" in data
    assert data["coords"].shape == (3, preprocess.npts_max, 3)
    assert data["outputs"].shape == (3, preprocess.npts_max, 5)
    assert data["params"].shape == (3, 1)

def test_run_all_creates_file(preprocess):
    preprocess.run_all()
    assert os.path.exists(preprocess.output_path)
    data = np.load(preprocess.output_path)
    assert "coords" in data
    assert "outputs" in data
    assert "params" in data
    assert data["coords"].shape == (3, preprocess.npts_max, 3)
    assert data["outputs"].shape == (3, preprocess.npts_max, 5)
    assert data["params"].shape == (3, 1)

def test_saved_arrays_are_normalized(preprocess):
    preprocess.run_all()
    data = np.load(preprocess.output_path)

    def assert_normalized(arr, tol=1e-6):
        flat = arr.reshape(-1, arr.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        assert np.all(np.abs(mean) < tol), f"Mean not ~0: {mean}"
        assert np.all(np.abs(std - 1) < tol), f"Std not ~1: {std}"

    assert_normalized(data["coords"])
    assert_normalized(data["outputs"])
    assert_normalized(data["params"])
def test_lhs_sample_point_count_and_spread():
    import os
    from preprocess import Preprocess
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '3D_data'))
    radius_files = {
        0.2: os.path.join(base_dir, "sphere_data_02.csv")
    }
    preprocess = Preprocess(radius_files=radius_files, output_path="test_processed_data.npz")
    preprocess.load_and_pad()

    coords = preprocess.coords[0]
    unpadded = coords[~np.all(coords == coords[0], axis=1)]  # filter padding

    n_expected = 50000
    n_actual = unpadded.shape[0]

    # Assert reasonable number of unique sampled points
    assert n_actual >= 0.8 * n_expected, f"Expected ~{n_expected} points, got {n_actual}"

    # Check bounding box coverage (should span >10% of domain in each axis)
    min_vals, max_vals = unpadded.min(axis=0), unpadded.max(axis=0)
    bbox_range = max_vals - min_vals
    assert np.all(bbox_range > 0.1), "Sampled points are not well-distributed in space"

