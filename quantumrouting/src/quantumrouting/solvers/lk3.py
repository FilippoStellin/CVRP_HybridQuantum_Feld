import lkh
import os

from dataclasses import dataclass

from src.quantumrouting.types import CVRPProblem, CVRPSolution


@dataclass
class LKHParams:

    time_limit_s: int = 60
    """Time limit in seconds to step the solver."""

    num_runs: int = 1
    """Number of runs."""

    distance_scaling_factor: int = 10
    """Scaling factor for distance matrices. """


def solve(problem: CVRPProblem, params: LKHParams) -> CVRPSolution:
    from src.quantumrouting.wrappers.lk3 import wrapper_to_lk3
    lkh_params = dict(
        mtsp_objective="MINSUM",
        runs=params.num_runs,
        time_limit=params.time_limit_s,
        vehicles=problem.num_vehicles,
    )

    tsplib_problem = wrapper_to_lk3(problem, params)

    current_path = os.path.dirname(os.path.abspath(__file__))
    solution = lkh.solve(
        f"{current_path}/LKH", tsplib_problem, **lkh_params
    )
    print(solution)
    solution = _unwrap_lkh_solution(problem, solution)

    return solution


def _unwrap_lkh_solution(problem: CVRPProblem, solution) -> CVRPSolution:
    pass