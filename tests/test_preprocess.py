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
    obj = Preprocess(files=files, dimension=2, output_path=str(tmp_path / "unittest_output.npz"), param_columns=["a", "b"])
    return obj

@pytest.fixture
def preprocess_instance3D(tmp_path):
    files = create_dummy_csv(tmp_path, dim=3)
    obj = Preprocess(files=files, dimension=3, output_path=str(tmp_path / "unittest_output.npz"), param_columns=["Sphere Radius"])
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

def test_preprocess_load_and_pad_2D(preprocess_instance2D):
    preprocess_instance2D.load_and_pad()
    
    assert len(preprocess_instance2D.coords) > 0
    assert len(preprocess_instance2D.params) > 0
    assert len(preprocess_instance2D.outputs) > 0
    assert preprocess_instance2D.npts_max > 0
    assert preprocess_instance2D.lhs_applied is False
    assert preprocess_instance2D.outputs_mean is not None
    assert preprocess_instance2D.outputs_std is not None


    

def test_preprocess_load_and_pad_3D(preprocess_instance3D):
    preprocess_instance3D.load_and_pad()
    
    assert len(preprocess_instance3D.coords) > 0
    assert len(preprocess_instance3D.params) > 0
    assert len(preprocess_instance3D.outputs) > 0
    assert preprocess_instance3D.npts_max > 0
    assert preprocess_instance3D.lhs_applied is True
    assert preprocess_instance3D.outputs_mean is not None
    assert preprocess_instance3D.outputs_std is not None

