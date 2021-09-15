from pathlib import Path

import argparse

from dwave_qbsolv import QBSolv

from src.quantumrouting.qubo import get_qubo
from src.quantumrouting.types import CVRPProblem
from src.quantumrouting.solver import solve

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
    for stem in stems:
        result = solve(problem=instances[stem], solver=solver)
        print(result)
        #solution = solve(instances[stem])
