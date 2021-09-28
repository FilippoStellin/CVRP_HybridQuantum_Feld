import neal
import numpy as np

from dataclasses import dataclass
from typing import Optional, Callable
from dimod import Sampler
from sklearn.cluster import KMeans

from src.quantumrouting.solvers.fullqubo import _unwrap_fullqubo_solution
from src.quantumrouting.types import CVRPSolution, CVRPProblem


@dataclass
class KmeansPartitionFullQuboParams:
    fixed_num_clusters: Optional[int] = None
    constraint_const: int = 10 ** 7
    """Constraint multiplier for qubo."""
    cost_const: int = 1
    """Cost Function multiplier for qubo."""
    seed: int = 0


def solver_fn(params: KmeansPartitionFullQuboParams,
              qubo_problem_fn: Callable, backend_solver: Sampler) -> Callable:

    def _solve(problem: CVRPProblem) -> CVRPSolution:

        deliveries = problem.location_idx[1:]

        num_clusters = int(params.fixed_num_clusters)

        clustering = KMeans(num_clusters, random_state=params.seed)
        clusters = clustering.fit_predict(problem.coords[deliveries])

        # Group deliveries at same clusters and reindex considering depot at 0

        subproblems_idx = np.array([
            deliveries[clusters == i] for i in range(num_clusters)
        ])

        subproblems = [
            CVRPProblem(problem_identifier=problem.problem_identifier,
                        location_idx=np.array(range(len(idxs) + 1)),
                        coords=np.concatenate(([problem.coords[0]], problem.coords[idxs])),
                        vehicle_capacity=problem.vehicle_capacity,
                        num_vehicles=1,
                        max_deliveries=problem.max_deliveries,
                        demands=np.concatenate(([0], problem.demands[idxs])),
                        depot_idx=problem.depot_idx)
            for idxs in subproblems_idx
        ]

        all_routes, all_costs, all_demands = [], 0, []
        for problem in subproblems:
             # Get qubo formulation problem
            vrp_qubo = qubo_problem_fn(problem=problem)

            # Solve qubo
            response = backend_solver.sample_qubo(vrp_qubo, solver=neal.SimulatedAnnealingSampler())
            solution = _unwrap_fullqubo_solution(problem=problem, result=response)
            for route in solution.routes:
                all_routes.append(route)
            for demand in solution.total_demands:
                all_demands.append(demand)
            all_costs += solution.cost

        return CVRPSolution(
            problem_identifier=problem.problem_identifier,
            routes=np.array(all_routes),
            cost=all_costs,
            total_demands=np.array(all_demands)
        )

    return _solve
