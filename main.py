import functools
from pathlib import Path

import einops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import taichi as ti
import seaborn as sns

from coevolve import PopulationManager, coevolve, get_scores
import constants as c
from simulator import Simulator
import visualize

# TODO: Increase NUM_TRIALS and restore
ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32,
        debug=False)

@ti.kernel
def render_fixed_topology(topo: ti.template()):
    center = ti.math.vec2(256.0, 256.0)
    max_dist = ti.math.distance(ti.math.vec2(0.0, 0.0), center)
    for w, x, y in topo:
        dist = ti.math.distance((x, y), center) / max_dist
        topo[w, x, y] = 1.0 - dist**2

def load_one_experiment_data(experiment):
    one_experiment_data = pd.DataFrame()
    for trial in range(c.NUM_TRIALS):
        trial_data = pd.read_csv(experiment.history_path(trial))
        trial_data['trial'] = trial
        one_experiment_data = pd.concat((one_experiment_data, trial_data))
    return one_experiment_data

def visualize_single_experiment(experiment):
    one_experiment_data = load_one_experiment_data(experiment)
    fig = sns.relplot(data=one_experiment_data, x='generation', y='score',
                      hue='role', kind='line')
    fig.set(title=experiment.name)
    fig.savefig(f'output/{experiment.name}.png')
    plt.close()

def load_all_experiments_data():
    all_experiments_data = pd.DataFrame()
    for experiment in experiments:
        one_experiment_data = load_one_experiment_data(experiment)
        one_experiment_data['experiment'] = experiment.name
        all_experiments_data = pd.concat((
            all_experiments_data, one_experiment_data))
    return all_experiments_data

def visualize_all_experiments():
    all_experiments_data = load_all_experiments_data()
    all_experiments_data = all_experiments_data.where(
        all_experiments_data['role'] == 'overall').dropna()
    fig = sns.relplot(data=all_experiments_data, x='generation', y='score',
                      hue='experiment', kind='line')
    fig.set(title='Overall scores across experiments')
    fig.savefig('output/overall.png')
    plt.close()

class Experiment:
    def __init__(self, name, fitness):
        self.name = name
        self.fitness = {
            'overall': lambda metrics: metrics['dist'] * metrics['inv_hits']
        } | fitness

    def video_path(self, trial):
        return Path(f'output/{self.name}_{trial}.mp4')

    def history_path(self, trial):
        return Path(f'output/{self.name}_{trial}.csv')

    def run(self):
        print(f'Running {c.NUM_TRIALS} trials of {self.name} '
              f'with {c.NUM_WORLDS} parallel simulations:')
        remaining_trials = [
            trial for trial in range(c.NUM_TRIALS)
            if not self.history_path(trial).exists()]
        if len(remaining_trials) == 0:
            print(f'Already ran {self.name}, reusing saved results.')
            return
        for trial in remaining_trials:
            (best_topography, best_controller), history = coevolve(
                Simulator(c.NUM_WORLDS), PopulationManager(self))
            self.record_history(history, trial)
            self.record_simulation(best_topography, best_controller, trial)
        visualize_single_experiment(self)

    def record_history(self, history, trial):
        filtered_history = []
        for event in history:
            for role in PopulationManager.roles + ['overall']:
                # Reorganize the history log so that we keep only the
                # population-average score for each generation, and break down
                # scores by role for easy visualization with Seaborn.
                filtered_history.append({
                    'generation': event['generation'],
                    'score': event[role].mean(),
                    'role': role
                })
        df = pd.DataFrame(filtered_history)
        df.to_csv(self.history_path(trial), index=False)

    def record_simulation(self, topography, controller, trial):
        simulator = Simulator()
        simulator.topographies.from_numpy(
            einops.repeat(topography, 'w h -> 1 w h'))
        # render_fixed_topology(simulator.topographies)
        simulator.controllers = controller
        # visualize.show(
        #     simulator,
        #     functools.partial(get_scores, fitness=self.fitness),
        #     debug=True)
        visualize.save(
            simulator,
            functools.partial(get_scores, fitness=self.fitness),
            self.video_path(trial))

experiments = [
    Experiment(
        name='condition_a',
        fitness={
            'topography': lambda metrics: metrics['dist'] * metrics['inv_hits'],
            'controller': lambda metrics: metrics['dist'] * metrics['inv_hits']}),
    Experiment(
        name='condition_b1',
        fitness={
            'topography': lambda metrics: metrics['inv_hits'],
            'controller': lambda metrics: metrics['dist']}),
    Experiment(
        name='condition_b2',
        fitness={
            'topography': lambda metrics: metrics['dist'],
            'controller': lambda metrics: metrics['inv_hits']}),
]

if __name__ == '__main__':
    sns.set_style('darkgrid')
    for experiment in experiments:
        experiment.run()
    visualize_all_experiments()
