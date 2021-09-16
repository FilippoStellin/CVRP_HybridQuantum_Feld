from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from dataclasses import dataclass
from random import sample

from src.quantumrouting.utils import compute_euclidean_distances

INSTANCE_MAX_SIZE = 10
MAX_NUM_VEHICLES = 3


@dataclass
class CVRPProblem:
    problem_identifier: str
    """Identifier"""
    location_idx: np.ndarray
    """ Location idx identifiers"""
    coords: np.ndarray
    """Delivery coordinates"""
    costs: np.ndarray
    """Cost matrix, usually the distance between locations"""
    vehicle_capacity: int
    """Maximum vehicle capacity"""
    num_vehicles: int
    """"Maximum number of vehicles"""
    demands: np.ndarray
    """Each demand size (dimension occupied in vehicle)"""
    max_deliveries: int
    """Maximum number of deliveries for each vehicle"""
    depot_idx: int = 0
    """Depot idx identifier"""

    @classmethod
    def from_file(cls, path: Union[Path, str]) -> CVRPProblem:
        """Load dataclass instance from provided file path."""

        with open(path) as f:
            data = json.load(f)

            packages = data['deliveries']
            # We are not able to solve big instances with a exact approach.
            # For now, I'm sampling results
            sampled_packages = sample(packages, INSTANCE_MAX_SIZE)

            coords = []
            demands = []
            for package_info in sampled_packages:
                coords.append([package_info['point']['lat'], package_info['point']['lng']])
                demands.append(package_info['size'])

            coords = [[data['origin']['lat'], data['origin']['lng']]] + coords
            demands = [0] + demands

            distances = compute_euclidean_distances(coords=np.array(coords))
            return CVRPProblem(problem_identifier=data['name'],
                               location_idx=np.array(range(len(sampled_packages) + 1)),
                               coords=np.array(coords),
                               costs=distances,
                               vehicle_capacity=data['vehicle_capacity'],
                               num_vehicles=MAX_NUM_VEHICLES,
                               max_deliveries=len(sampled_packages),
                               demands=np.array(demands),
                               depot_idx=0)


@dataclass
class CVRPSolution:
    """Problem identifier"""
    problem_identifier: str
    """ Computed routes"""
    routes: np.ndarray
    """ Total cost"""
    cost: int
    """Total dimension occupied"""
    total_demands: np.ndarray
