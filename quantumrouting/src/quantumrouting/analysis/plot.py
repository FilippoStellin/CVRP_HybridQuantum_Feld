
import folium

from src.quantumrouting.types import CVRPSolution, CVRPProblem

MAP_COLORS = (
    "black",
    "blue",
    "darkred",
    "purple",
    "red",
    "orange",
    "green",
    "pink",
    "darkblue",
    "beige",
    "gray",
    "lightgreen",
    "lightblue",
    "lightgray",
    "cadetblue",
)


def plot_route(problem: CVRPProblem, solution: CVRPSolution) -> None:

    routes = solution.routes
    m = folium.Map(
        location=(problem.coords[0][0], problem.coords[0][1]),
        zoom_start=12,
        tiles="cartodbpositron",
    )

    for i, route in enumerate(routes):
        origin = (problem.coords[route[problem.depot_idx]][0], problem.coords[route[problem.depot_idx]][1])
        folium.CircleMarker(origin, color="red", radius=3, weight=5).add_to(m)
        route_color = MAP_COLORS[i % len(MAP_COLORS)]
        route_coords = [(problem.coords[idx][0], problem.coords[idx][1]) for idx in route]
        folium.Polygon(
            route_coords,
            popup=f"Vehicle {i}",
            color=route_color,
            weight=1,
        ).add_to(m)

    m.save("route_example.html")

    return
