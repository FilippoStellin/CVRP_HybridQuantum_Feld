from itertools import product, combinations

import numpy as np

from collections import defaultdict
from typing import Dict, Tuple

from src.quantumrouting.types import CVRPProblem



"""
cost_example =
array([[0.  , 0.27, 0.86, 0.34],
       [0.27, 0.  , 0.81, 0.13],
       [0.86, 0.81, 0.  , 0.69],
       [0.34, 0.13, 0.69, 0.  ]], dtype=float32)
"""


def objective_function(problem: CVRPProblem, cost_const: int) -> Dict[Tuple, int]:
    variables = defaultdict(int)

    start = 0

    for vehicle in range(len(problem.capacities)):
        min_final = start - 1
        max_final = start + len(problem.location_idx) - 2

        # For for each destination: ((i1, j), (i2, j)) -> customer j in position i1 (i2)
        # 0 <= i1 <= num*vehicles * (number_destinations-1)
        # i2 = i1 + 1
        # 0 <= j <= number_destinations
        # ((0, 0), (1, 0)): 0.0, ((0, 0), (1, 1)): 0.27,
        # ((0, 0), (1, 2)): 0.86, ((0, 0), (1, 3)): 0.34, ...)
        for step in range(min_final + 1, max_final):
            for node1 in problem.location_idx:
                for node2 in problem.location_idx:
                    cost = problem.costs[node1][node2]
                    idx = ((step, node1), (step + 1, node2))
                    variables[idx] += cost * cost_const

        # First and Last destination to depot cost
        for destination in problem.location_idx:
            # Depot and first destination
            # ((0, 0), (0, 0)): 0.0, ((0, 1), (0, 1)): 0.27,
            # ((0, 2), (0, 2)): 0.86, ((0, 3), (0, 3)): 0.34,
            # ((3, 0), (3, 0)): 0.0, ((3, 1), (3, 1)): 0.27
            # ((3, 2), (3, 2)): 0.86, ((3, 3), (3, 3)): 0.34
            idx = ((start, destination), (start, destination))
            cost = problem.costs[problem.depot_idx][destination]
            variables[idx] += cost_const * cost

            # Last destination and depot
            # ((2, 0), (2, 0)): 0.0, ((2, 1), (2, 1)): 0.27
            # ((2, 2), (2, 2)): 0.86,((2, 3), (2, 3)): 0.34
            # ((5, 0), (5, 0)) 0.0, ((5, 1), (5, 1)): 0.27
            # ((5, 2), (5, 2)): 0.86, ((5, 3), (5, 3)): 0.34
            idx = ((max_final, destination), (max_final, destination))
            cost = problem.costs[destination][problem.depot_idx]
            variables[idx] += cost_const * cost

            # Capacity optimization
            capacity = problem.capacities[vehicle]
            for (d1, d2) in combinations(problem.location_idx[1:], 2):
                for (s1, s2) in combinations(range(start, max_final + 1), 2):
                    idx = ((s1, d1), (s2, d2))
                    idx2 = ((s1, d2), (s2, d1))
                    cost = problem.demands[d1] * problem.demands[d2] / capacity ** 2
                    variables[idx] += cost_const * cost
                    variables[idx2] += cost_const * cost

        start = max_final + 1

    return variables


def constraints(problem: CVRPProblem, constraint_const: int) -> Dict[Tuple, int]:
    constraints = defaultdict(int)

    steps = int(np.sum(problem.maximum_deliveries))

    # Only one step for one destination.
    for dest in problem.location_idx[1:]:
        variables = [(step, dest) for step in range(steps)]
        for var in variables:
            constraints[(var, var)] -= 2 * constraint_const
        for field in product(variables, variables):
            constraints[field] += constraint_const

    # Vehicles can wait in depot..
    for step in range(0, int(steps)):
        variables = [(step, dest) for dest in problem.location_idx]
        for var in variables:
            constraints[(var, var)] -= 2 * constraint_const
        for field in product(variables, variables):
            constraints[field] += constraint_const

    return constraints


def get_qubo(
        problem: CVRPProblem,
        constraint_const: int = 10 ** 7,
        cost_const: int = 1
) -> Dict[Tuple[int, int], int]:
    """
    Build the VRP problem in the qubo formulation.

    Objective function:
        Minimize the distance traveled by all vehicles.

    Constraints:
        Each delivery must occurs only one once in routes.
        Each position in route must be assigned to only one delivery.

    Constants:
        constraint_const - A: multiplier for constraints in qubo
        distance_const - B: multiplier for costs in qubo

    """

    objective_repr = objective_function(problem=problem, cost_const=cost_const)
    constraints_repr = constraints(problem=problem, constraint_const=constraint_const)
    return {k: objective_repr.get(k, 0) + constraints_repr.get(k, 0) for k in
            set(objective_repr) | set(constraints_repr)}
