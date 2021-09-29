from typing import Callable, List

import neal
import numpy as np

from dataclasses import dataclass

from dimod import SampleSet, Sampler

from src.quantumrouting.solvers.utils import total_solution_cost, calculate_capacity_occupied
from src.quantumrouting.types import CVRPProblem, CVRPSolution

from src.quantumrouting.distances import compute_distances


@dataclass
class FullQuboParams:
    constraint_const: int = 10 ** 7
    """Constraint multiplier for qubo."""
    cost_const: int = 1
    """Cost Function multiplier for qubo."""


def solver_fn(
        backend_solver: Sampler,
        qubo_problem_fn: Callable) -> Callable:

    def _solve(problem: CVRPProblem) -> CVRPSolution:
        # Get qubo formulation problem
        vrp_qubo = qubo_problem_fn(problem=problem)

        # Solve qubo
        response = backend_solver.sample_qubo(vrp_qubo,
                                              solver=neal.SimulatedAnnealingSampler())

        return _unwrap_fullqubo_solution(problem=problem, result=response)

    return _solve


def _unwrap_fullqubo_solution(problem: CVRPProblem, result: SampleSet) -> CVRPSolution:

    distances = compute_distances(coords=problem.coords)

    sample = list(result)[0]

    all_vehicles_results = []
    vehicle_result = []

    vehicle = 0
    step = 0

    # Decoding solution from qubo sample.
    for (s, dest) in sample:
        if sample[(s, dest)] == 1:
            if dest != 0:
                vehicle_result.append(dest)
            step += 1
            if problem.max_deliveries == step:
                # Add depot at begginning
                vehicle_result.insert(0, problem.depot_idx)
                # Add depot at ending
                vehicle_result.append(problem.depot_idx)
                all_vehicles_results.append(vehicle_result)
                step = 0
                vehicle += 1
                vehicle_result = []
                if problem.num_vehicles <= vehicle:
                    break

    # Calculate Cost and total capacity occupied in each vehicle
    cost = total_solution_cost(
        depot_idx=problem.depot_idx,
        demands=problem.demands,
        routes=all_vehicles_results,
        cost_matrix=distances
    )

    occupied_capacity = calculate_capacity_occupied(
        demands=problem.demands,
        routes=all_vehicles_results
    )

    return CVRPSolution(
        problem_identifier=problem.problem_identifier,
        routes=np.array(all_vehicles_results),
        cost=cost,
        total_demands=np.array(occupied_capacity)
    )


