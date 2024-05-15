"""Run a series of experiments with evolved agents in an evolved environment.

In this experiment, we coevolve CPPNs to represent both environments and
controllers for agents in this environment. The agents are balls, and their
environment is an uneven topography that they roll on.

This script runs a series of experiments which evaluate fitness in different
ways. For each one, it captures a video of the most fit resulting simulation,
charts showing the pace of evolution in each experiment and comparing across
experiments, and a CSV dump of all metrics and scores over the course of each
generation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import taichi as ti
import seaborn as sns

from coevolve import coevolve
import constants as c
from population_manager import PopulationManager
from simulator import Simulator
import visualize


ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32,
        debug=False)


# These global objects capture all GPU memory allocations for the life of this
# program. Don't attempt to delete these objects, as the GPU memory will not be
# freed! Taichi makes it difficult to do otherwise.
population_manager = PopulationManager()
simulator = Simulator() # debug=True)


def summarize_single_experiment(experiment):
    """Generate a fitness chart for one experiment."""
    data = experiment.get_scores()
    fig = sns.relplot(
        data=data, x='generation', y='fitness', hue='role', kind='line')
    fig.set(title=experiment.name)
    fig.savefig(f'output/chart_{experiment.name}.png')
    plt.close()


def get_all_experiment_scores():
    """Load data from all experiments into a single DataFrame."""
    all_scores = []
    for experiment in experiments:
        scores = experiment.get_scores()
        scores['experiment'] = experiment.name
        all_scores.append(scores)
    return pd.concat(all_scores)


def summarize_all_experiments():
    """Generate a fitness chart comparing results across experiments."""
    scores = get_all_experiment_scores()
    scores = scores[scores['role'] == 'overall']
    fig = sns.relplot(data=scores, x='generation', y='fitness',
                      hue='experiment', kind='line')
    fig.set(title='Overall scores across experiments')
    fig.savefig('output/chart_overall.png')
    plt.close()


class Experiment:
    def __init__(self, name, fitness):
        self.name = name
        # A dict of fitness scores per role (ie, for topographies, controllers,
        # and the simulation overall). The overall score is the same for all
        # experiments, but all experiments define per-role fitness differently.
        self.fitness = {
            'overall': go_forward_and_dont_crash
        } | fitness
        self.video_path = Path(f'output/video_{name}.mp4')

    def log_path(self, log_name):
        return Path(f'output/log_{self.name}_{log_name}.csv')

    def run(self):
        print(f'Running {c.NUM_TRIALS} trials of {self.name} '
              f'with {c.NUM_WORLDS} parallel simulations:')
        best_world_index, logs = coevolve(
            simulator, population_manager, self.fitness)
        self.save_simulation(best_world_index)
        self.save_logs(logs)
        summarize_single_experiment(self)

    def save_logs(self, logs):
        logs.metrics.to_csv(self.log_path('metrics'), index=False)
        logs.scores.to_csv(self.log_path('scores'), index=False)
        logs.genealogy.to_csv(self.log_path('genealogy'), index=False)

    def get_scores(self):
        return pd.read_csv(self.log_path('scores'))

    def save_simulation(self, world_index):
        def get_scores(metrics):
            return {
                key: self.fitness[key](metrics).iloc[0]
                for key in self.fitness
            }
        #visualize.show(simulator, world_index, get_scores)
        visualize.save(simulator, world_index, get_scores, self.video_path)


# Convenience functions for fitness calculations.
def go_forward(metrics):
    # Including the angular displacement term seems to evovle circulating
    # behavior in a simple, fixed environment, but when the topography
    # coevolves with the controllers, what you tend to get is gentle slopes
    # with agents that mostly just passively roll. Perhaps this is because
    # angular displacement is disrupted by collisions, which are unpredictable
    # and hard to manage. On the other hand, since this fitness function
    # correlates with collisions, it produces better coevolutionary dynamics,
    # since the two populations have complementary goals.
    return metrics['lin_disp'] # / (1 + metrics['ang_disp'])


def dont_crash(metrics):
    return 1 / (1 + metrics['hits'])


def go_forward_and_dont_crash(metrics):
    return go_forward(metrics) * dont_crash(metrics)


# The moderate versions of the above functions produce slightly better results
# overall, and more healthy evolutionary dynamics in the B1 and B2 conditions.
# This is probably because without this moderation, neither population is
# actually trying to optimize the overall goal, so they don't produce pairings
# that are fit overall (one population gains fitness while the other crashes).
def moderate_go_forward(metrics):
    return go_forward_and_dont_crash(metrics) * go_forward(metrics)


def moderate_dont_crash(metrics):
    return go_forward_and_dont_crash(metrics) * dont_crash(metrics)


# All the experiments run by this script.
experiments = [
    Experiment(
        name='condition_a',
        fitness={
            'topography': go_forward_and_dont_crash,
            'controller': go_forward_and_dont_crash}),
    Experiment(
        name='condition_b1',
        fitness={
            'topography': moderate_dont_crash,
            'controller': moderate_go_forward}),
    Experiment(
        name='condition_b2',
        fitness={
            'topography': moderate_go_forward,
            'controller': moderate_dont_crash}),
]


# Actually run all the experiments and generate all outputs. This takes about
# five minutes on my laptop.
if __name__ == '__main__':
    sns.set_style('darkgrid')
    for experiment in experiments:
        experiment.run()
    summarize_all_experiments()
