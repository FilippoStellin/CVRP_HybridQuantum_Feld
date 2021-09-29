import pytest

import numpy as np

from src.quantumrouting.solvers import lk3

from src.quantumrouting.types import CVRPProblem


@pytest.fixture
def cvrp_problem():
    max_num_vehicles = 1

    coords = [
        [-15.6570138544452, -47.802664728268745],
        [-15.65879313293694, -47.7496622016347],
        [-15.651440380492554, -47.75887552060412],
        [-15.651207309372888, -47.755018806591394],
        [-15.648706444367969, -47.758785390289965],
        [-15.66047286919706, -47.75284167302011]
    ]
    return CVRPProblem(problem_identifier='bla',
                       location_idx=np.array([0, 1, 2, 3, 4, 5]),
                       coords=np.array(coords),
                       vehicle_capacity=100,
                       num_vehicles=max_num_vehicles,
                       max_deliveries=5,
                       demands=np.array([0, 10, 10, 7, 3, 10]),
                       depot_idx=0)


def test_vrp_lkh_solver(cvrp_problem):

    params = lk3.LKHParams()

    solver = lk3.solver_fn(params=params)

    result = solver(problem=cvrp_problem)

    assert result.problem_identifier == 'bla'
    assert (result.routes == np.array([[0, 4, 2, 3, 1, 5, 0]])).all()
    assert result.total_demands == 40
    assert int(result.cost) == 12262


