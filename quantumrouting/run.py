from pathlib import Path

import argparse

from dwave_qbsolv import QBSolv

from src.quantumrouting.analysis.plot import plot_route
from src.quantumrouting.types import CVRPProblem
from src.quantumrouting.solvers.qbsolv import solve, QBSolvParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--instances", type=str, required=True)

    args = parser.parse_args()

    instances_path = Path(args.instances)
    if instances_path.is_file():
       instances = {"": CVRPProblem.from_file(instances_path)}

    elif instances_path.is_dir():
        instances = {
            f.stem: CVRPProblem.from_file(f) for f in instances_path.iterdir()
        }

    else:
        raise ValueError("input files do not match, use files or directories.")

    stems = instances.keys()
    solver = QBSolv()
    params = QBSolvParams()
    for stem in stems:
        problem = instances[stem]
        result = solve(problem=problem, params=params)
        plot_route(problem=problem, solution=result)
