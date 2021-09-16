from pathlib import Path

import argparse

from src.quantumrouting.solvers import lk3
from src.quantumrouting.solvers import qbsolv
from src.quantumrouting.analysis.plot import plot_route
from src.quantumrouting.types import CVRPProblem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--solver", type=str)

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
    #params = QBSolvParams()
    #for stem in stems:
    #    problem = instances[stem]
    #    result = qbsolv_solve(problem=problem, params=params)
    #    print(result)
    #    plot_route(problem=problem, solution=result)

    params = lk3.LKHParams()
    for stem in stems:
        problem = instances[stem]
        result = lk3.solve(problem=problem, params=params)
        print(result)
        plot_route(problem=problem, solution=result)