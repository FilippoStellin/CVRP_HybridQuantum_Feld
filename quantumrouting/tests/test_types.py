import pytest

import numpy as np

from quantumrouting.src.quantumrouting.types import CVRPProblem


def test_cvrp_from_file():
    path = 'tests/sample_test.json'
    problem = CVRPProblem.from_file(path, 1)

    expected_coords = [
        [-15.657013854445248, -47.802664728268745],
        [-15.65879313293694, -47.7496622016347],
        [-15.651440380492554, -47.75887552060412],
        [-15.651207309372888, -47.755018806591394],
        [-15.648706444367969, -47.758785390289965],
        [-15.66047286919706, -47.75284167302011]
    ]

    expected_idx = [0, 1, 2, 3, 4, 5]

    assert problem.problem_identifier == "cvrp-0-df-0"
    assert (problem.location_idx == np.array(expected_idx)).all()
    assert (problem.coords == np.array(expected_coords)).all()
    assert (problem.demands == np.array([0, 10, 10, 7, 3, 10])).all()
    assert problem.max_deliveries == 5
    assert problem.depot_idx == 0
