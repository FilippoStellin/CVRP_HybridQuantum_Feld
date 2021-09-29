
import pytest
import numpy as np

from src.quantumrouting.types import CVRPProblem
from src.quantumrouting.wrappers.qubo import vrp_objective_function


@pytest.fixture
def cvrp_problem():
    max_num_vehicles = 1

    coords = [
        [-15.6570138544452, -47.802664728268745],
        [-15.65879313293694, -47.7496622016347],
        [-15.651440380492554, -47.75887552060412],
    ]
    return CVRPProblem(problem_identifier='bla',
                       location_idx=np.array([0, 1, 2]),
                       coords=np.array(coords),
                       vehicle_capacity=100,
                       num_vehicles=max_num_vehicles,
                       max_deliveries=5,
                       demands=np.array([0, 10, 10]),
                       depot_idx=0)


@pytest.fixture
def cvrp_problem_two_vehicles():
    max_num_vehicles = 2

    coords = [
        [-15.6570138544452, -47.802664728268745],
        [-15.65879313293694, -47.7496622016347],
        [-15.651440380492554, -47.75887552060412],
    ]
    return CVRPProblem(problem_identifier='bla',
                       location_idx=np.array([0, 1, 2]),
                       coords=np.array(coords),
                       vehicle_capacity=100,
                       num_vehicles=max_num_vehicles,
                       max_deliveries=5,
                       demands=np.array([0, 10, 10]),
                       depot_idx=0)


def test_vrp_objective_function(cvrp_problem):
    # For for each destination: ((i1, j), (i2, j)) -> customer j in position i1 (i2)
    expected_qubo = {
        ((0, 0), (0, 0)): 0.0,
        ((0, 0), (1, 0)): 0.0,
        ((0, 0), (1, 1)): 5678.349666395941,
        ((0, 0), (1, 2)): 4729.312006109361,
        ((0, 1), (0, 1)): 5678.349666395941,
        ((0, 1), (1, 0)): 5678.349666395941,
        ((0, 1), (1, 1)): 0.0,
        ((0, 1), (1, 2)): 1281.239300025782,
        ((0, 2), (0, 2)): 4729.312006109361,
        ((0, 2), (1, 0)): 4729.312006109361,
        ((0, 2), (1, 1)): 1281.239300025782,
        ((0, 2), (1, 2)): 0.0,
        ((1, 0), (1, 0)): 0.0,
        ((1, 1), (1, 1)): 5678.349666395941,
        ((1, 2), (1, 2)): 4729.312006109361
    }

    result = vrp_objective_function(problem=cvrp_problem, cost_const=1)
    assert expected_qubo == result


def test_vrp_objective_function_two_vehicles(cvrp_problem_two_vehicles):
    # For for each destination: ((i1, j), (i2, j)) -> customer j in position i1 (i2)
    expected_qubo = {
        ((0, 0), (1, 0)): 0.0,
        ((0, 0), (1, 1)): 5678.349666395941,
        ((0, 0), (1, 2)): 4729.312006109361,
        ((0, 1), (1, 0)): 5678.349666395941,
        ((0, 1), (1, 1)): 0.0,
        ((0, 1), (1, 2)): 1281.239300025782,
        ((0, 2), (1, 0)): 4729.312006109361,
        ((0, 2), (1, 1)): 1281.239300025782,
        ((0, 2), (1, 2)): 0.0,
        ((0, 0), (0, 0)): 0.0,
        ((1, 0), (1, 0)): 0.0,
        ((0, 1), (0, 1)): 5678.349666395941,
        ((1, 1), (1, 1)): 5678.349666395941,
        ((0, 2), (0, 2)): 4729.312006109361,
        ((1, 2), (1, 2)): 4729.312006109361,
        ((2, 0), (3, 0)): 0.0,
        ((2, 0), (3, 1)): 5678.349666395941,
        ((2, 0), (3, 2)): 4729.312006109361,
        ((2, 1), (3, 0)): 5678.349666395941,
        ((2, 1), (3, 1)): 0.0,
        ((2, 1), (3, 2)): 1281.239300025782,
        ((2, 2), (3, 0)): 4729.312006109361,
        ((2, 2), (3, 1)): 1281.239300025782,
        ((2, 2), (3, 2)): 0.0,
        ((2, 0), (2, 0)): 0.0,
        ((3, 0), (3, 0)): 0.0,
        ((2, 1), (2, 1)): 5678.349666395941,
        ((3, 1), (3, 1)): 5678.349666395941,
        ((2, 2), (2, 2)): 4729.312006109361,
        ((3, 2), (3, 2)): 4729.312006109361
    }

    result = vrp_objective_function(problem=cvrp_problem_two_vehicles, cost_const=1)
    assert expected_qubo == result
