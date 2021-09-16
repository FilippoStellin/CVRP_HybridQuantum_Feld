from typing import List, Callable

import lkh
import os

import numpy as np
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


def solve(params: LKHParams) -> Callable:
    from src.quantumrouting.wrappers.lk3 import wrapper_to_lk3

    def _solve(problem: CVRPProblem) -> CVRPSolution:

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

        return _unwrap_lkh_solution(problem, solution)

    return _solve


def _unwrap_lkh_solution(problem: CVRPProblem, solution: List[int]) -> CVRPSolution:
    """
    Decode result in a TSPLib format.
    Code based on
    https://github.com/loggi/loggibud/blob/master/loggibud/v1/baselines/task1/lkh_3.py#L70
    """

    sample_solution = solution[0]
    # Add depot at the end of solution
    sample_solution.append(sample_solution[0])
    num_deliveries = len(problem.demands) - 1

    # Reindexing
    delivery_indices = np.array(solution[0]) - 1

    # Now we split the sequence into vehicles using a simple generator.
    def route_gen(seq):
        route = [problem.depot_idx]
        for el in seq[1:]:
            if el <= num_deliveries:
                route.append(el)

            elif route:
                route.append(problem.depot_idx)
                yield np.array(route)
                route = [problem.depot_idx]

        # Output last route if any
        if route:
            yield np.array(route)

    all_vehicles_results = list(route_gen(delivery_indices))

    # Calculate Cost and total capacity occupied in each vehicle
    cost = 0
    total_demands_size = []
    for vehicle_route in all_vehicles_results:
        demands_size = 0
        if vehicle_route == []:
            continue
        prev = vehicle_route[0]
        for dest in vehicle_route[1:]:
            cost += problem.costs[prev][dest]
            demands_size += problem.demands[dest]
            prev = dest
        total_demands_size.append(demands_size)
        cost += problem.costs[prev][problem.depot_idx]

    return CVRPSolution(
        problem_identifier=problem.problem_identifier,
        routes=np.array(all_vehicles_results),
        cost=cost,
        total_demands=np.array(total_demands_size)
    )
