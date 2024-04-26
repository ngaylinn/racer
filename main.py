import functools
import gc
from pathlib import Path

import einops
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
    fig.savefig(f'output/chart_{experiment.name}.png')
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
    fig.savefig('output/chart_overall.png')
    plt.close()

class Experiment:
    def __init__(self, name, fitness):
        self.name = name
        self.fitness = {
            'overall': go_forward_and_dont_crash
        } | fitness

    def video_path(self, trial):
        return Path(f'output/video_{self.name}_{trial}.mp4')

    def history_path(self, trial):
        return Path(f'output/history_{self.name}_{trial}.csv')

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
            # GPU memory management can be challenging, so make sure we free up
            # all defunct objects and associated GPU memory allocations as soon
            # as we're done with them.
            gc.collect()
            # print(pop_manager.neat['controller'].curr_pop)
            # print('c_n_cp', gc.get_referrers(pop_manager.neat['controller'].curr_pop))
            # print('c_n_np', len(gc.get_referrers(pop_manager.neat['controller'].next_pop)))
            # print('c_n_m', len(gc.get_referrers(pop_manager.neat['controller'].matches)))
            # print('t_n_cp', len(gc.get_referrers(pop_manager.neat['topography'].curr_pop)))
            # print('t_n_np', len(gc.get_referrers(pop_manager.neat['topography'].next_pop)))
            # print('t_n_m', len(gc.get_referrers(pop_manager.neat['topography'].matches)))
            # print('c', len(gc.get_referrers(pop_manager.actuators['controller'])))
            # print('r', len(gc.get_referrers(pop_manager.actuators['topography'])))

            # TODO: Restore.
            # self.record_history(history, trial)
            self.record_simulation(best_topography, best_controller, trial)

            # And clean up the temporary objects used to record the best
            # simulation, just to be thorough.
            del best_topography, best_controller
            gc.collect()
        # visualize_single_experiment(self)
        print('End run experiment')

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
        print('Begin record_simulation')
        simulator = Simulator()
        # TODO: It's problematic to duplicate this between here and
        # coevolve.PopulationManager, so maybe find a way to share.
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
        print('End record_simulation')

def go_forward(metrics):
    # Including the angular displacement term seems to evovle circulating
    # behavior in a simple, fixed environment, but when the topography
    # coevolves with the controllers, what you tend to get is gentle slopes
    # with agents that mostly just passively roll. Perhaps this is because
    # angular displacement is disrupted by collisions, which are unpredictable
    # and hard to manage. On the other hand, since this fitness function
    # correlates with collisions, it produces better coevolutionary dynamics,
    # since the two populations have complementary goals.
    return metrics['lin_disp'] #/ (1 + metrics['ang_disp'])

def dont_crash(metrics):
    return metrics['inv_hits']

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

experiments = [
    Experiment(
        name='condition_a',
        fitness={
            'topography': go_forward_and_dont_crash,
            'controller': go_forward_and_dont_crash}),
#    Experiment(
#        name='condition_b1',
#        fitness={
#            'topography': moderate_dont_crash,
#            'controller': moderate_go_forward}),
#    Experiment(
#        name='condition_b2',
#        fitness={
#            'topography': moderate_go_forward,
#            'controller': moderate_dont_crash}),
]

if __name__ == '__main__':
    print('Begin main')
    sns.set_style('darkgrid')
    for experiment in experiments:
        experiment.run()
    print('End main')
    # visualize_all_experiments()
