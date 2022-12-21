from random import random

import pandas
from pandas import DataFrame
from pandas import read_csv


def get_data() -> DataFrame:
    data: DataFrame = read_csv("wdbc.csv")

    # Set column names
    data.columns = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
                    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
                    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst",
                    "fractal_dimension_worst"]
    data.drop(columns="id")
    data.replace("?", pandas.np.nan)

    data["diagnosis"].replace("M", True).replace("B", False)

    return data


def fitness_function(data: DataFrame, layers: tuple) -> float:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000)
    score = cross_val_score(mlp, data.drop(columns="diagnosis"), data["diagnosis"])
    return score.mean()


# Find the best layers with a genetic algorithm
def crossover(layers1: tuple, layers2: tuple) -> tuple:
    if len(layers1) != len(layers2):
        raise ValueError("layers1 and layers2 must have the same length")

    layers = []
    for i in range(len(layers1)):
        if random() > 0.5:
            layers.append(layers1[i])
        else:
            layers.append(layers2[i])

    return tuple(layers)


def mutate(layers: tuple) -> tuple:
    layers = list(layers)
    for i in range(len(layers)):
        if random() > 0.5:
            layers[i] = int(layers[i] * random())
    return tuple(layers)


def genetic_algorithm(data: DataFrame, population_size: int, generations: int) -> tuple:
    population = []
    for i in range(population_size):
        layers = []
        for j in range(3):
            layers.append(int(random() * 1000))
        population.append(tuple(layers))

    for i in range(generations):
        fitnesses = []
        for layers in population:
            fitnesses.append(fitness_function(data, layers))

        # Sort the population by fitness
        population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]

        # Crossover
        for i in range(population_size // 2):
            population.append(crossover(population[i], population[population_size - i - 1]))

        # Mutate
        for i in range(population_size // 2):
            population.append(mutate(population[i]))

    return population[0]


result: tuple = genetic_algorithm(get_data(), 10, 10)
print(result)
