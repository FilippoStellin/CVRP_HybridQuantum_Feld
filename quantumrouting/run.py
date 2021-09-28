from pathlib import Path

import argparse

import hybrid
from dwave_qbsolv import QBSolv

from src.quantumrouting.solvers import lk3
from src.quantumrouting.solvers import partitionqubo
from src.quantumrouting.solvers import fullqubo
from src.quantumrouting.analysis.plot import plot_route
from src.quantumrouting.types import CVRPProblem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--backend", type=str,  default='cpu')

    args = parser.parse_args()

    instances_path = Path(args.instances)
    if instances_path.is_file():
       instances = {"": CVRPProblem.from_file(instances_path)}

    elif instances_path.is_dir():
        instances = {
            f.stem: CVRPProblem.from_file(f) for f in instances_path.iterdir()
        }

    else:
        raise ValueError("Input files do not match, use files or directories.")

    if args.backend == 'cpu':
        backend_solver = QBSolv()

    elif args.backend == 'qpu':
        workflow = hybrid.Loop(
            hybrid.RacingBranches(
                hybrid.InterruptableTabuSampler(),
                hybrid.EnergyImpactDecomposer(size=30, rolling=True, rolling_history=0.75)
                | hybrid.QPUSubproblemAutoEmbeddingSampler()
                | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=1)

        backend_solver = hybrid.HybridSampler(workflow)
    else:
        raise ValueError("Backend solver not implemented.")

    if args.solver == 'lkh':
        params = lk3.LKHParams()
        solver = lk3.solver_fn(params=params)

    elif args.solver == 'fullqubo':
        params = fullqubo.FullQuboParams()
        solver = fullqubo.solver_fn(params=params, backend_solver=backend_solver)

    elif args.solver == 'partitionfullqubo':
        params = partitionqubo.KmeansPartitionFullQuboParams(fixed_num_clusters=3)
        solver = partitionqubo.solver_fn(params=params, backend_solver=backend_solver)

    else:
        raise ValueError("Solver not Implemented..")

    stems = instances.keys()

    for stem in stems:
        problem = instances[stem]
        result = solver(problem=problem)
        m = plot_route(problem=problem, solution=result)
        m.save(f'{problem.problem_identifier}_{args.solver}.html')
