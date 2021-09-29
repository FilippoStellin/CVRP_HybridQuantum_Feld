from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from dataclasses import dataclass
from random import sample, seed


MAX_NUM_VEHICLES = 1


@dataclass
class CVRPProblem:
    problem_identifier: str
    """Identifier"""
    location_idx: np.ndarray
    """ Location idx identifiers"""
    coords: np.ndarray
    """Delivery coordinates"""
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
    def from_file(cls, path: Union[Path, str], sample_frac: float = 0.015) -> CVRPProblem:
        """Load dataclass instance from provided file path."""

        with open(path) as f:
            data = json.load(f)

            packages = data['deliveries']
            # We are not able to solve big instances with a exact approach.
            # For now, I'm sampling results
            if sample_frac != 1:
                seed(a=sample_frac, version=2)
                total_packages = int(sample_frac*len(packages))
                packages = sample(packages, total_packages)

            coords = []
            demands = []
            for package_info in packages:
                coords.append([package_info['point']['lat'], package_info['point']['lng']])
                demands.append(package_info['size'])

            origin = data['origin']
            coords = [[origin['lat'], origin['lng']]] + coords
            demands = [0] + demands

            return CVRPProblem(problem_identifier=data['name'],
                               location_idx=np.array(range(len(packages) + 1)),
                               coords=np.array(coords),
                               vehicle_capacity=data['vehicle_capacity'],
                               num_vehicles=MAX_NUM_VEHICLES,
                               max_deliveries=len(packages),
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
