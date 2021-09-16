import tsplib95
from src.quantumrouting.solvers.lk3 import LKHParams
from src.quantumrouting.types import CVRPProblem


def wrapper_to_lk3(problem: CVRPProblem, params: LKHParams):

    demands_section = {
        idx + 1: demand for idx, demand in enumerate(problem.demands)
    }

    coords_section = {
        idx + 1: coord for idx, coord in enumerate(problem.coords)
    }

    problem = tsplib95.models.StandardProblem(
        type="CVRP",
        dimension=len(problem.coords),
        capacity=problem.vehicle_capacity,
        edge_weight_type="EUC_2D",
        node_coords=coords_section,
        demands=demands_section,
    )

    return problem

