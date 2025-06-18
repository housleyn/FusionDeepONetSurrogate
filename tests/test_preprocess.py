import os
import numpy as np
import pandas as pd
import pytest
import glob
import os


from preprocess import Preprocess
import preprocess.methods_preprocess as mp

def ellipse_file_list():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","ellipse_data"))
    file_paths = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    return file_paths
def sphere_file_list():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","sphere_data"))
    file_paths = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    return file_paths

@pytest.fixture
def preprocess_instance2D():
    object = Preprocess(files=ellipse_file_list(), dimension=2, output_path="unittest_output.npz", param_columns=["a", "b"])
    return object
@pytest.fixture
def preprocess_instance3D():
    object = Preprocess(files=sphere_file_list(), dimension=3, output_path="unittest_output.npz", param_columns=["Sphere Radius"])
    return object


def test_preprocess_init_2D(preprocess_instance2D):
    assert isinstance(preprocess_instance2D, Preprocess)
    assert preprocess_instance2D.dimension == 2
    assert len(preprocess_instance2D.files) > 0
    assert preprocess_instance2D.output_path == "unittest_output.npz"
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
    assert preprocess_instance3D.output_path == "unittest_output.npz"
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

