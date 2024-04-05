from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
import taichi as ti


def select(fitness_scores, count):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return np.random.randint(0, len(fitness_scores), size=count)

    sample_period = total_fitness / count
    sample_offset = np.random.random() * sample_period
    sample_points = [sample_offset + i * sample_period for i in range(count)]

    result = []
    index = -1
    fitness_so_far = 0.0
    for sample in sample_points:
        while sample > fitness_so_far:
            index += 1
            fitness_so_far += fitness_scores[index]
        result.append(index)
    return result


class FitnessEvaluator(ABC):
    @abstractmethod
    def score_one(self, simulator, world=0):
        ...

    def score_all(self, simulator):
        return np.array(
            [self.score_one(simulator, world=world)
             for world in range(simulator.num_worlds)])

    @abstractmethod
    def visualize(self):
        ...


@ti.data_oriented
class Speedy(FitnessEvaluator):
    def __init__(self, num_worlds):
        self.num_worlds = num_worlds
        self.scores = ti.field(float, num_worlds)

    def score_one(self, simulator, world=0):
        self.dist = simulator.objects.dist.to_numpy()[world].sum()
        self.hits = simulator.objects.hits.to_numpy()[world].sum()
        self.score = self.dist / (1 + self.hits)
        return self.score

    @ti.kernel
    def score_all_kernel(self, objects: ti.template()):
        hits = ti.Vector([0] * self.num_worlds)
        dist = ti.Vector([0.0] * self.num_worlds)
        for w, o in ti.ndrange(*objects.shape):
            hits[w] += objects[w, o].hits
            dist[w] += objects[w, o].dist
        for w in range(self.num_worlds):
            self.scores[w] = dist[w] / (1 + hits[w])

    def score_all(self, simulator):
        self.score_all_kernel(simulator.objects)
        return self.scores.to_numpy()

    def visualize(self, gui):
        gui.text(f'{self.dist:4.2} / {self.hits} = {self.score:4.2f}',
                 [0.05, 0.95], font_size=24, color=0xff00ff)
