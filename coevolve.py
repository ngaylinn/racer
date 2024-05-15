"""The co-evolutionary algorithm for this project.

The main entrypoint to this module is the coevolve() function, which runs the
main evolutionary loop for all trials of a single experiment, and returns a log
of fitness scores for all roles (controller, topography, and overall) across
all trials and generations. This module also provides the select() and
pair_select() functions, which use Stochastic Universal Selection to pick
population indices for breeding based on fitness scores.

The fitness functions themselves are provided by main.py and the work of
generating, simulating, and breeding populations is provided by the
PopulationManager class.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import trange

import constants as c


def select(fitness_scores):
    count = len(fitness_scores)
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return np.random.randint(
            0, len(fitness_scores), size=count, dtype=np.int32)

    sample_period = total_fitness / count
    sample_offset = np.random.random() * sample_period
    sample_points = [sample_offset + i * sample_period for i in range(count)]

    result = np.empty(count, dtype=np.int32)
    population_index = -1
    fitness_so_far = 0.0
    for sample_index, sample in enumerate(sample_points):
        while sample > fitness_so_far:
            population_index += 1
            fitness_so_far += fitness_scores[population_index]
        result[sample_index] = population_index
    return result


def shuffled(array):
    return np.random.choice(array, size=len(array), replace=False)


def pair_select(fitness_scores):
    return np.array([
        # In every breeding pair, the most fit is considered the "parent" and
        # the other is the "mate." This is used by the Neat algorithm to
        # introduce a small bias towards more fit individuals in crossover.
        [p, m] if fitness_scores[p] > fitness_scores[m] else [m, p]
        for p, m in zip(select(fitness_scores),
                        shuffled(select(fitness_scores)))
    ], dtype=np.int32)


@dataclass
class LogData:
    # Objective summaries of the simulation
    metrics: pd.DataFrame
    # Subjective fitness of individuals simulated
    scores: pd.DataFrame
    # Match-making history of individuals simulated
    genealogy: pd.DataFrame
    # TODO: Add population diversity metrics?


def coevolve(simulator, population_manager, fitness):
    # Randomly initialize the populations
    population_manager.randomize()

    # Tracking across generations
    progress = trange(c.NUM_GENERATIONS)
    overall_score = 0.0
    all_metrics = []
    all_scores = []
    all_genealogies = []

    # Evolve the random populations
    for generation in progress:
        progress.set_description(f'Score == {overall_score:4.2f}')

        population_manager.populate_simulator(simulator)
        metrics = population_manager.annotate_metrics(simulator.run())
        metrics['generation'] = generation
        all_metrics.append(metrics)

        # TODO: Optimize? This is actually the slowest part of the whole process,
        # since the next generation can't run until scores are returned to Python,
        # fitness is calculated, and matches are pushed back to Taichi on the GPU.
        # That back and forth can only be avoided by computing fitness scores and
        # doing selection on the GPU.
        scores = population_manager.get_scores(fitness, metrics)
        scores['generation'] = generation
        scores.to_csv('scores.csv', index=False)
        overall_score = scores[scores['role'] == 'overall']['fitness'].mean()
        all_scores.append(scores)

        if generation + 1 < c.NUM_GENERATIONS:
            genealogy = population_manager.propagate(scores)
            genealogy['generation'] = generation
            all_genealogies.append(genealogy)

    best_row = fitness['overall'](metrics).argmax()
    # Currently, the index values correspond with the values in the world
    # column, but look up the world just in case that ever changes.
    best_world_index = int(metrics.loc[best_row]['world'])

    logs = LogData(
        pd.concat(all_metrics),
        pd.concat(all_scores),
        pd.concat(all_genealogies)
    )
    return (best_world_index, logs)
