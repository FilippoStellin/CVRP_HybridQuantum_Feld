import neal
import numpy as np

from dwave_qbsolv import QBSolv

from src.quantumrouting.qubo import get_qubo
from src.quantumrouting.types import CVRPProblem, CVRPSolution


def solve(
        problem: CVRPProblem,
        solver: QBSolv
) -> CVRPSolution:

    # Get qubo formulation problem
    vrp_qubo = get_qubo(problem=problem)

    # Solve qubo
    response = solver.sample_qubo(vrp_qubo, solver=neal.SimulatedAnnealingSampler())

    sample = list(response)[0]

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
            if problem.maximum_deliveries[vehicle] == step:
                # Add depot at begginning
                vehicle_result.insert(0, problem.depot_idx)
                # Add depot at ending
                vehicle_result.append(problem.depot_idx)
                all_vehicles_results.append(vehicle_result)
                step = 0
                vehicle += 1
                vehicle_result = []
                if len(problem.maximum_deliveries) <= vehicle:
                    break

    # Calculate Cost
    cost = 0
    for vehicle_route in all_vehicles_results:
        if vehicle_route == []:
            continue
        prev = vehicle_route[0]
        for dest in vehicle_route[1:]:
            cost += problem.costs[prev][dest]
            prev = dest
        cost += problem.costs[prev][problem.depot_idx]

    return CVRPSolution(
        route=np.array(all_vehicles_results),
        cost=cost
    )
