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
    frames = []
    selections_per_trial = []
    for trial, trial_df in scores.groupby('trial'):
        selections = pair_select(trial_df['fitness'].to_numpy())
        frames.append(pd.DataFrame({
            'individual': np.arange(c.NUM_INDIVIDUALS, dtype=np.int32),
            'trial': trial,
            'parent': selections[:, 0],
            'mate': selections[:, 1],
        }))

        # Trials are just sub-populations within a single NeatPopulation, so
        # translate from trial-relative indexing to absolute indexing.
        np.add(selections, trial * c.NUM_INDIVIDUALS, out=selections)
        selections_per_trial.append(selections)

    return np.concatenate(selections_per_trial), pd.concat(frames)


def fixed_world_assignments():
    return np.concatenate([
        # Assign individuals to the same NUM_MATCHUPS pairings for each trial.
        np.tile(
            np.arange(c.NUM_INDIVIDUALS, dtype=np.int32),
            c.NUM_MATCH_UPS
        # Trials are just sub-populations within a single NeatPopulation, so
        # translate from trial-relative indexing to absolute indexing.
        ) + trial * c.NUM_INDIVIDUALS
        for trial in range(c.NUM_TRIALS)])


def random_world_assignments():
    return np.concatenate([
        # Randomly assign individuals to NUM_MATCHUPS pairings for each trial.
        np.concatenate([
            np.random.permutation(c.NUM_INDIVIDUALS).astype(np.int32) \
            # Trials are just sub-populations within a single NeatPopulation,
            # so translate from trial-relative indexing to absolute indexing.
            + trial * c.NUM_INDIVIDUALS
            for _ in range(c.NUM_MATCH_UPS)
        ])
        for trial in range(c.NUM_TRIALS)
    ])


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
        all_genealogies = []
        for role in self.roles:
            # Get selections and genealogy log.
            selections, genealogy = pair_select_per_trial(
                scores[scores['role'] == role])

            # Apply selections to the population / actuators.
            # TODO: Ideally, we should record which matches performed crossover
            # in the genealogy.
            self.neat[role].propagate(selections)
            self.update_actuators(role)

            # Log the feel genealogy of the populations.
            genealogy['role'] = role
            all_genealogies.append(genealogy)
        return pd.concat(all_genealogies)

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
        for role in self.roles:
            # Calculate the role-specific fitness score for all worlds, then
            # average each individual's score across all match-ups in each of
            # the trials.
            frames.append(pd.DataFrame({
                'trial': metrics['trial'].to_numpy(),
                'role': role,
                'individual': metrics[f'{role}_index'].to_numpy(),
                'fitness': fitness[role](metrics)
            }).groupby(['trial', 'role', 'individual']).mean().reset_index())

        # Also calculate an overall score for each trial, not attributed to any
        # individuals.
        frames.append(pd.DataFrame({
            'trial': metrics['trial'].to_numpy(),
            'role': 'overall',
            'individual': None,
            'fitness': fitness['overall'](metrics)
        }).groupby(['trial', 'role']).mean().reset_index())

        return pd.concat(frames)
