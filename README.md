# Quantum CVRP

Implements the Capacited Vehicle Routing problem using a Quantum Annealer with the QUBO formulation.

Based on the the implementation described in https://arxiv.org/pdf/1811.07403.pdf

Using data from https://github.com/loggi/loggibud/blob/master/docs/quickstart.md



### VRP Resuts
![vrp](https://user-images.githubusercontent.com/14301789/134584442-946fbe43-1e1d-4477-b4d7-0ef149548d9a.jpeg)

#### FullQuboSolver

Size: 1
Vehicles: 1

![image](https://user-images.githubusercontent.com/14301789/135166925-1379f3f5-11f9-4e9b-9141-44dd104c6862.png)


### CVRP Results


## Run

`poetry run python3 run.py  --instances PATH-TO-DATA --solver partitionfullqubo`
