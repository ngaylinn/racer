"""Manage multiple populations to evolve and simulate.

This project coevolves CPPNs representing the topography of simulated
environments and controllers for balls in those environments. To do this, we
run many simulations at once in many "parallel worlds." In each generation,
we randomly match up all c.NUM_INDIVIDUALS controllers and topographies
c.NUM_MATCH_UPS times, and each pairing is simulated in parallel. Each
evolutionary experiment is run c.NUM_TRIALS times to account for variance
between runs. This makes for c.NUM_WORLDS parallel simulations.

The main purpose of this script is to manage indexing across parallel data
structures. We have c.NUM_TRIALS * c.NUM_INDIVIDUALS instances of the Neat
algorithm, but only individuals in the same trial are allowed to breed with
each other, so selection must respect this. These CPPNs have to be distributed
across c.NUM_WORLDS simulations, with actuators which allocate the memory
needed to actually use the CPPNs. The metrics computed by those simulations
then need to be summarized and attributed to individual topographies and
controllers to compute their fitness.
"""

import itertools

import numpy as np
import pandas as pd
import taichi as ti

from coevolve import pair_select
import constants as c
from neatchi import Neat, ActivationMaps, Actuators

import ball


def pair_select_per_trial(scores):
    """Select parents and mates, respecting trial boundaries.

    This function takes fitness scores broken down by trial and performs
    selection on each batch of fitness scores. It then merges the result into a
    single array of selections representing all trials to pass onto the Neatchi
    library. This way, we can create one logical population for each trial
    within a single Neat population, enforcing that individuals in different
    trials never mate without exposing that constraint to the Neatchi library.
    """
    matches_per_trial = [
        pair_select(df['fitness'].to_numpy()) + (trial * c.NUM_INDIVIDUALS)
        for trial, df in scores.groupby('trial')
    ]
    return np.concatenate(matches_per_trial)


def fixed_world_assignments():
    return np.tile(
        np.arange(c.NUM_INDIVIDUALS, dtype=np.int32),
        c.NUM_TRIALS * c.NUM_MATCH_UPS)


def random_world_assignments():
    return np.concatenate([
        np.random.permutation(c.NUM_INDIVIDUALS)
        for _ in range(c.NUM_TRIALS * c.NUM_MATCH_UPS)], dtype=np.int32)


def index_dict_of_arrays(dict_of_arrays, index):
    return {
        key: values[index]
        for key, values in dict_of_arrays.items()
    }


@ti.data_oriented
class PopulationManager:
    roles = ['topography', 'controller']
    def __init__(self):
        # Neat algorithms for evolving CPPNs for both roles.
        self.neat = {
            'topography': Neat(
                num_inputs=2, num_outputs=1,
                num_individuals=(c.NUM_INDIVIDUALS * c.NUM_TRIALS),
                is_recurrent=False),
            'controller': Neat(
                num_inputs=ball.NUM_INPUTS, num_outputs=ball.NUM_OUTPUTS,
                num_individuals=(c.NUM_INDIVIDUALS * c.NUM_TRIALS),
                is_recurrent=True)
        }

        # Actuators for interfacing between the CPPNs and the simulation.
        self.actuators = {
            'topography': ActivationMaps(
                num_worlds=c.NUM_WORLDS,
                num_individuals=(c.NUM_INDIVIDUALS * c.NUM_TRIALS),
                map_size=c.WORLD_SIZE),
            'controller':  Actuators(
                num_worlds=c.NUM_WORLDS, num_activations=c.NUM_BALLS)
        }

        # Assignments of CPPNs to simulated worlds.
        self.world_assignments = {
            role: np.zeros(c.NUM_WORLDS, dtype=np.int32)
            for role in self.roles
        }

    def populate_simulator(self, simulator):
        simulator.controllers = self.actuators['controller']
        simulator.topographies = self.actuators['topography']
        simulator.randomize_balls()

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
            selections = pair_select_per_trial(
                scores[scores['role'] == role])
            self.neat[role].propagate(selections)
            self.update_actuators(role)

    def annotate_metrics(self, metrics):
        # Associate each world with the individuals of each role that
        # interacted there, and which experiment trial that corresponds to.
        for role in self.roles:
            metrics[f'{role}_index'] = self.world_assignments[role]
        metrics['trial'] = np.repeat(
            np.arange(c.NUM_TRIALS), c.NUM_INDIVIDUALS * c.NUM_MATCH_UPS)
        return metrics

    def get_scores(self, fitness, metrics):
        frames = []
        for trial, trial_df in metrics.groupby('trial'):
            for role in self.roles:
                frames.append(pd.DataFrame({
                    'trial': trial,
                    'role': role,
                    'individual': np.arange(c.NUM_INDIVIDUALS),
                    # This averages metrics across instances THEN computes
                    # fitness, which is much more efficient than computing
                    # fitness then averaging the results, but may not work for
                    # other fitness functions.
                    'fitness': fitness[role](
                        trial_df.groupby(f'{role}_index').mean())
                }))
            frames.append(pd.DataFrame([{
                'trial': trial,
                'role': 'overall',
                'fitness': fitness['overall'](trial_df).mean()
            }]))
        return pd.concat(frames)
