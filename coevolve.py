from neatchi import Neat, NeatControllers, NeatRenderers
import numpy as np
import taichi as ti
from tqdm import trange

import agent
import constants as c

def select(fitness_scores, count=None):
    if count is None:
        count = len(fitness_scores)

    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return np.random.randint(0, len(fitness_scores), size=count)

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

def pair_select(fitness_scores, count=None):
    return np.array([
        [p, m] if fitness_scores[p] > fitness_scores[m] else [m, p]
        for p, m in zip(select(fitness_scores, count),
                        select(fitness_scores, count))
    ])

def fixed_world_assignments():
    return np.tile(np.arange(c.NUM_INDIVIDUALS), c.NUM_MATCH_UPS)

def random_world_assignments():
    return np.concatenate([
        np.random.permutation(c.NUM_INDIVIDUALS)
        for _ in range(c.NUM_MATCH_UPS)])

def reduce_fitness(fitness_scores, world_assignments):
    scores = np.zeros(c.NUM_INDIVIDUALS)
    for score, individual in zip(fitness_scores, world_assignments):
        scores[individual] += score
    return scores / c.NUM_MATCH_UPS


@ti.kernel
def render_fixed_topology(topo: ti.template()):
    center = ti.math.vec2(256.0, 256.0)
    max_dist = ti.math.distance(ti.math.vec2(0.0, 0.0), center)
    for w, x, y in topo:
        dist = ti.math.distance((x, y), center) / max_dist
        topo[w, x, y] = 1.0 - dist**2

@ti.data_oriented
class PopulationManager:
    roles = ['topography', 'controller']
    def __init__(self, expt):
        self.expt = expt

        # Neat algorithms for evolving CPPNs for both roles
        self.neat = {
            'topography': Neat(
                num_inputs=2, num_outputs=1,
                num_individuals=c.NUM_INDIVIDUALS),
            'controller': Neat(
                num_inputs=agent.NUM_INPUTS, num_outputs=agent.NUM_OUTPUTS,
                num_individuals=c.NUM_INDIVIDUALS)
        }

        # Assignments of CPPNs to simulated worlds.
        self.world_assignments = {}

        # Actuators for interfacing between the CPPNs and the simulation.
        self.actuators = {
            'topography': NeatRenderers(
                num_worlds=c.NUM_WORLDS, num_rows=c.WORLD_SIZE),
            'controller': NeatControllers(
                num_worlds=c.NUM_WORLDS, num_activations=c.NUM_OBJECTS)
        }

    def populate_simulator(self, simulator):
        self.actuators['topography'].render_all(simulator.topographies)
        # render_fixed_topology(simulator.topographies)
        simulator.controllers = self.actuators['controller']
        simulator.randomize_objects()

    def get_best_individuals(self, scores):
        world_index = np.argmax(scores['overall'])
        return (
            self.actuators['topography'].render_one(
                self.world_assignments['topography'][world_index],
                c.WORLD_SHAPE),
            self.actuators['controller'].get_one(
                self.world_assignments['controller'][world_index]))

    def update_actuators(self, role):
        self.world_assignments[role] = random_world_assignments()
        self.actuators[role].update(
            self.neat[role].curr_pop, self.world_assignments[role])

    def randomize(self):
        for role in self.roles:
            self.neat[role].random_population()
            self.update_actuators(role)

    def propagate(self, scores):
        for role in self.roles:
            selections = pair_select(scores[role])
            self.neat[role].propagate(selections)
            self.update_actuators(role)

    def get_scores(self, metrics):
        return get_scores(metrics, self.expt.fitness, self.world_assignments)

roles = PopulationManager.roles

def get_scores(metrics, fitness, world_assignments=None):
    if world_assignments is None:
        world_assignments = {role: [0] for role in roles}

    scores = {
        role: reduce_fitness(
            fitness[role](metrics),
            world_assignments[role])
        for role in roles
    }
    scores['overall'] = fitness['overall'](metrics)
    return scores

def coevolve(simulator, population_manager):
    population_manager.randomize()
    history = []

    progress = trange(c.NUM_GENERATIONS)
    scores = {'overall': np.zeros(1)}
    for generation in progress:
        progress.set_description(f'Score == {scores["overall"].mean():4.2f}')

        population_manager.populate_simulator(simulator)
        metrics = simulator.run()
        scores = population_manager.get_scores(metrics)

        history.append({'generation': generation} | metrics | scores)
        if generation + 1 < c.NUM_GENERATIONS:
            population_manager.propagate(scores)

    return (population_manager.get_best_individuals(scores), history)
