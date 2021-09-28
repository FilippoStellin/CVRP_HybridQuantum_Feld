from typing import List

import numpy as np


def calculate_capacity_occupied(
        demands: np.ndarray,
        routes: List[List[int]],
) -> List[int]:
    total_capacity_occupied = []
    for vehicle_route in routes:
        demands_size = 0
        for dest in vehicle_route[1:]:
            demands_size += demands[dest]
        total_capacity_occupied.append(demands_size)

    return total_capacity_occupied


def total_solution_cost(
        depot_idx: int,
        demands: np.ndarray,
        routes: List[List[int]],
        cost_matrix: np.ndarray
) -> int:
    cost = 0
    for vehicle_route in routes:
        demands_size = 0
        if vehicle_route == []:
            continue
        prev = vehicle_route[0]
        for dest in vehicle_route[1:]:
            cost += cost_matrix[prev][dest]
            demands_size += demands[dest]
            prev = dest
        cost += cost_matrix[prev][depot_idx]

    return cost
