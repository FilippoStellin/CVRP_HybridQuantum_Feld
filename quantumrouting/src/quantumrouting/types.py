from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from dataclasses import dataclass
from random import sample


from src.quantumrouting.utils import compute_euclidean_distances

INSTANCE_MAX_SIZE = 15
MAX_NUM_VEHICLES = 5


@dataclass
class CVRPProblem:
    problem_identifier: str

    location_idx: np.ndarray
    """ Location idx identifiers"""
    costs: np.ndarray
    """Cost matrix, usually the distance between locations"""
    capacities: np.ndarray
    """Maximum vehicle capacities allowed"""
    demands: np.ndarray
    """Each demand size (dimension occupied in vehicle)"""
    maximum_deliveries: np.ndarray
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

            number_packages = len(sampled_packages)

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
                               costs=distances,
                               capacities=np.ones((MAX_NUM_VEHICLES))*data['vehicle_capacity'],
                               demands=np.array(demands),
                               maximum_deliveries=np.ones((MAX_NUM_VEHICLES))*number_packages,
                               depot_idx=0
                               )


@dataclass
class CVRPSolution:
    route: np.ndarray
    cost: int
