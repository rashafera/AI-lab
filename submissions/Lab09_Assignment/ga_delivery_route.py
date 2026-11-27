"""
Genetic Algorithm for Delivery Route Optimisation
Depot: A. Delivery locations: B, C, D, E, F, G.
Goal: find the shortest route A -> (B..G in some order) -> A.
"""

import random

# ---------------------------------------------------------------
# Dataset: symmetric distance matrix in kilometres
# ---------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCE = [
    #  A  B  C  D  E  F  G
    [0, 4, 8, 6, 7, 9, 5],  # A
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
]
DEPOT = "A"
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]

# ---------------------------------------------------------------
# GA settings
# ---------------------------------------------------------------
RANDOM_SEED = 42
POPULATION_SIZE = 30
GENERATIONS = 100
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_COUNT = 2


def distance_between(a, b):
    """Distance in km between two named locations."""
    return DISTANCE[LOCATIONS.index(a)][LOCATIONS.index(b)]


def route_distance(chromosome):
    """Total route length: depot -> stops in chromosome order -> back to depot."""
    full_route = [DEPOT] + chromosome + [DEPOT]
    return sum(distance_between(full_route[i], full_route[i + 1])
               for i in range(len(full_route) - 1))


def fitness(chromosome):
    """Higher fitness = shorter route (we minimise distance)."""
    return 1.0 / route_distance(chromosome)


def create_initial_population():
    """Each chromosome is a random permutation of B-G (always valid)."""
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = DELIVERY_STOPS[:]
        random.shuffle(chromosome)
        population.append(chromosome)
    return population


def tournament_selection(population):
    """Pick TOURNAMENT_SIZE random routes and return the shortest one."""
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=route_distance)


def ordered_crossover(parent1, parent2):
    """
    Ordered Crossover (OX): copy a slice from parent1, then fill the
    remaining positions with parent2's stops in their original order,
    skipping any already used. The child never repeats or misses a stop.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    fill_stops = [stop for stop in parent2 if stop not in child]
    fill_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_stops[fill_index]
            fill_index += 1
    return child


def swap_mutation(chromosome):
    """Swap two random stops; the route stays a valid permutation."""
    mutated = chromosome[:]
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def format_route(chromosome):
    return " -> ".join([DEPOT] + chromosome + [DEPOT])


def run_ga():
    random.seed(RANDOM_SEED)
    population = create_initial_population()

    best_route = min(population, key=route_distance)
    print(f"Generation   0 | best distance = {route_distance(best_route)} km")

    for generation in range(1, GENERATIONS + 1):
        # Elitism: carry the best routes over unchanged
        population.sort(key=route_distance)
        next_population = [route[:] for route in population[:ELITE_COUNT]]

        # Fill the rest with children from selection + crossover + mutation
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1[:]
            child = swap_mutation(child)
            next_population.append(child)

        population = next_population
        generation_best = min(population, key=route_distance)
        if route_distance(generation_best) < route_distance(best_route):
            best_route = generation_best[:]

        if generation % 10 == 0:
            print(f"Generation {generation:3d} | best distance = "
                  f"{route_distance(best_route)} km")

    print("\nBest route found:", format_route(best_route))
    print("Total distance:", route_distance(best_route), "km")
    return best_route


if __name__ == "__main__":
    run_ga()
