import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sdf import SDF


def test_sphere_origin_radius():
    sdf = SDF()
    coords = np.array([[1.0, 0.0, 0.0], [0.0,0.0,0.0], [2.0,0.0,0.0]])
    params = np.ones((3,1))
    d = sdf.sphere_formation_sdf(coords, params)
    assert d.shape == (3,1)
    assert np.isclose(d[0,0], 0.0)
    assert np.isclose(d[1,0], -1.0)
    assert np.isclose(d[2,0], 1.0)


def test_variable_center():
    sdf = SDF()
    coords = np.array([[2.0,0.0,0.0]])
    params = np.array([[1.0,0.0]])  # sphere center at (1,0,0) with radius 1
    d = sdf.sphere_formation_sdf(coords, params)
    assert d.shape == (1,1)
    assert np.isclose(d[0,0], 0.0)

