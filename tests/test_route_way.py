import random

import pandas as pd
import pytest

from conftest import load_module

rw = load_module("Generative-Algorithm/route_way.py")

CITIES = ["A", "B", "C", "D", "E"]


def make_distances():
    distances = {}
    value = 100
    for i, c1 in enumerate(CITIES):
        for c2 in CITIES[i + 1:]:
            distances[frozenset([c1, c2])] = value
            value += 10
    return distances


def test_compute_fitness_sums_leg_distances():
    distances = {
        frozenset(["A", "B"]): 5,
        frozenset(["B", "C"]): 7,
        frozenset(["A", "C"]): 100,
    }
    assert rw.compute_fitness(("A", "B", "C"), distances) == 12
    # the route does not loop back to the start
    assert rw.compute_fitness(("C", "B", "A"), distances) == 12


def test_mutate_agent_preserves_waypoints():
    random.seed(7)
    genome = tuple(CITIES)
    mutated = rw.mutate_agent(genome, max_mutations=3)
    assert sorted(mutated) == sorted(genome)
    assert len(mutated) == len(genome)


def test_shuffle_mutation_preserves_waypoints():
    random.seed(7)
    genome = tuple(CITIES)
    mutated = rw.shuffle_mutation(genome)
    assert sorted(mutated) == sorted(genome)


def test_generate_random_population_size_and_contents():
    random.seed(7)
    population = rw.generate_random_population(8, set(CITIES))
    assert len(population) == 8
    for agent in population:
        assert sorted(agent) == sorted(CITIES)


def test_load_waypoints_and_full_ga_run(tmp_path):
    rows = []
    distances = make_distances()
    for pair, dist in distances.items():
        w1, w2 = sorted(pair)
        rows.append({"waypoint1": w1, "waypoint2": w2,
                     "distance_m": dist, "duration_s": dist * 2})
    csv_file = tmp_path / "waypoints.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    loaded_dist, loaded_dur, waypoints = rw.load_waypoints(str(csv_file))
    assert waypoints == set(CITIES)
    assert loaded_dist == distances
    assert loaded_dur[frozenset(["A", "B"])] == distances[frozenset(["A", "B"])] * 2

    random.seed(7)
    routes = rw.run_genetic_algorithm(generations=50, population_size=10,
                                      csv_file=str(csv_file))
    assert len(routes) > 0
    best = routes[-1]
    assert sorted(best) == sorted(CITIES)
    # the GA should never end worse than it started
    assert rw.compute_fitness(best, distances) <= rw.compute_fitness(routes[0], distances)


def test_run_genetic_algorithm_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        rw.run_genetic_algorithm(generations=10, population_size=10,
                                 csv_file=str(tmp_path / "missing.csv"))
