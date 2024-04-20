import einops
from neatchi import Neat, NeatControllers, NeatRenderers
import numpy as np
import taichi as ti
from tqdm import trange

import agent
from coevolve import Domain, CoOptimizer, coevolve, pair_select
import constants as c
import fitness
from simulator import Simulator
import visualize

# TODO: Increase NUM_TRIALS and restore
ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32,
        debug=False)


def visualize_population(topo_generators, fitness_scores):
    temp = ti.field(float, shape=(50, 128, 128))
    topo_generators.render_all(temp)
    topos = einops.rearrange(
        temp.to_numpy(), '(gr gc) tr tc -> (gr tr) (gc tc)',
        gr=10, gc=5)
    gui = ti.GUI('Racer', (128*10, 128*5),
                 background_color=0xffffff, show_gui=True)
    while gui.running:
        gui.set_image(topos)
        for i in range(50):
            x, y = divmod(i, 5)
            gui.text(f'{fitness_scores[i]:0.3f}',
                     (x / 10.0, (y + 1) / 5.0), color=0xFF00FF)
        gui.show()


@ti.kernel
def render_fixed_topology(topo: ti.template()):
    center = ti.math.vec2(256.0, 256.0)
    max_dist = ti.math.distance(ti.math.vec2(0.0, 0.0), center)
    for w, x, y in topo:
        dist = ti.math.distance((x, y), center) / max_dist
        topo[w, x, y] = 1.0 - dist**2


@ti.data_oriented
class RacerDomain(Domain):
    def __init__(self, simulator):
        self.simulator = simulator
        self.dist = ti.field(float, simulator.num_worlds)
        self.hits = ti.field(float, simulator.num_worlds)

    @ti.kernel
    def summarize_simulation(self):
        self.dist.fill(0.0)
        self.hits.fill(0.0)
        for w, o in ti.ndrange(*self.simulator.objects.shape):
            self.hits[w] += self.simulator.objects[w, o].hits
            self.dist[w] += self.simulator.objects[w, o].dist

    def get_metrics(self):
        self.summarize_simulation()
        return {
            'dist': np.nan_to_num(self.dist.to_numpy()),
            'hits': np.nan_to_num(self.hits.to_numpy())
        }


    def evaluate(self, interactions):
        topo_generators, controllers = interactions
        self.simulator.randomize_objects()
        #render_fixed_topology(self.simulator.topographies)
        topo_generators.render_all(self.simulator.topographies)
        #self.simulator.agents.from_numpy(controllers)
        self.simulator.controllers = controllers
        for _ in range(c.NUM_STEPS):
            self.simulator.step()
        return self.get_metrics()

class RacerCoOptimizer(CoOptimizer):
    def __init__(self, name):
        self.name = name

        self.topography_neat = Neat(
            num_inputs=2, num_outputs=1,
            num_individuals=c.NUM_INDIVIDUALS)
        self.topo_generators = NeatRenderers(
            num_worlds=c.NUM_WORLDS, num_rows=c.WORLD_SIZE)

        self.controller_neat = Neat(
            num_inputs=agent.NUM_INPUTS, num_outputs=agent.NUM_OUTPUTS,
            num_individuals=c.NUM_INDIVIDUALS)
        self.controllers = NeatControllers(
            num_worlds=c.NUM_WORLDS, num_activations=c.NUM_OBJECTS)

    def overall_score(self, metrics):
        return np.mean(metrics['dist'] / (1 + metrics['hits']))


def fixed_world_assignments():
    return np.tile(np.arange(c.NUM_INDIVIDUALS), c.NUM_TRIALS)

def random_world_assignments():
    return np.concatenate([
        np.random.permutation(c.NUM_INDIVIDUALS)
        for _ in range(c.NUM_TRIALS)])

def reduce_fitness(fitness_scores, world_assignments):
    scores = np.zeros(c.NUM_INDIVIDUALS)
    for score, individual in zip(fitness_scores, world_assignments):
        scores[individual] += score
    return scores / c.NUM_TRIALS


class ConditionACoOptimizer(RacerCoOptimizer):
    def __init__(self):
        super().__init__('condition_a')

    def get_interactions(self, scores=None):
        self.topography_world_assignments = random_world_assignments()
        self.controller_world_assignments = random_world_assignments()
        if scores is None:
            self.topo_generators.update(
                self.topography_neat.random_population(),
                self.topography_world_assignments)
            self.controllers.update(
                self.controller_neat.random_population(),
                self.controller_world_assignments)
        else:
            self.topo_generators.update(
                self.topography_neat.propagate(
                    pair_select(scores['topography'])),
                self.topography_world_assignments)
            self.controllers.update(
                self.controller_neat.propagate(
                    pair_select(scores['controller'])),
                self.controller_world_assignments)
        return (self.topo_generators, self.controllers)

    def score_interactions(self, metrics):
        combined = metrics['dist'] / (1 + metrics['hits'])
        # self.topography_neat.curr_pop.print_all()
        # visualize_population(self.topo_generators, combined)
        return {
            'topography': reduce_fitness(
                combined, self.topography_world_assignments),
            'controller': reduce_fitness(
                combined, self.controller_world_assignments),
            'combined': combined
        }

    def best_interaction(self, interactions, scores):
        topo_generators, controllers = interactions
        best_index = np.argmax(scores['combined'])
        return (topo_generators.render_one(best_index, c.WORLD_SHAPE),
                controllers.get_one(best_index))


class ConditionBCoOptimizer(RacerCoOptimizer):
    def __init__(self, case):
        assert case in (1, 2)
        self.case = case
        super().__init__(f'condition_b{case}')

    def get_interactions(self, scores=None):
        self.topography_world_assignments = random_world_assignments()
        self.controller_world_assignments = random_world_assignments()
        if scores is None:
            self.topo_generators.update(
                self.topography_neat.random_population(),
                self.topography_world_assignments)
            self.controllers.update(
                self.controller_neat.random_population(),
                self.topography_world_assignments)
        else:
            self.topo_generators.update(
                self.topography_neat.propagate(
                    pair_select(scores['topography'])),
                self.topography_world_assignments)
            self.controllers.update(
                self.controller_neat.propagate(
                    pair_select(scores['controller'])),
                self.topography_world_assignments)
        return (self.topo_generators, self.controllers)

    def score_interactions(self, metrics):
        dist = metrics['dist']
        inv_hits = 1 / (1 + metrics['hits'])
        return {
            'topography': reduce_fitness(
                dist if self.case == 1 else inv_hits,
                self.topography_world_assignments),
            'controller': reduce_fitness(
                inv_hits if self.case == 1 else dist,
                self.controller_world_assignments),
            # Note, this hasn't been reduced! It's sized to the number of
            # worlds, not the number of individuals, so that best_interaction
            # can find the pairing with the highest combined score.
            'combined': dist * inv_hits
        }

    def best_interaction(self, interactions, scores):
        topo_generators, controllers = interactions
        world_index = np.argmax(scores['combined'])
        topography_index = self.topography_world_assignments[world_index]
        controller_index = self.controller_world_assignments[world_index]
        return (topo_generators.render_one(topography_index, c.WORLD_SHAPE),
                controllers.get_one(controller_index))


def record_simulation(topography, controller, optimizer, trial):
    def get_scores(simulator):
        metrics = RacerDomain(simulator).get_metrics()
        return optimizer.score_interactions(metrics)

    simulator = Simulator()
    simulator.topographies.from_numpy(
        einops.repeat(topography, 'w h -> 1 w h'))
    #render_fixed_topology(simulator.topographies)
    simulator.controllers = controller
    # visualize.show(simulator, get_scores, debug=False)
    visualize.save(
        simulator, get_scores, f'output/{optimizer.name}_{trial}.mp4')


def summarize_results(history, optimizer, trial):
    # TODO: Generate charts and stuff.
    pass


def run_experiment(domain, cooptimizer):
    print(f'Running experiment {cooptimizer.name} '
          f'with {c.NUM_WORLDS} parallel simulations:')
    for trial in range(5):
        (best_topography, best_controller), history = coevolve(
            domain, cooptimizer, c.NUM_GENERATIONS)
        record_simulation(
            best_topography, best_controller, cooptimizer, trial)
        summarize_results(history, cooptimizer, trial)
    print('Done!\n')


if __name__ == '__main__':
    simulator = Simulator(c.NUM_WORLDS)
    domain = RacerDomain(simulator)
    run_experiment(domain, ConditionACoOptimizer())
    run_experiment(domain, ConditionBCoOptimizer(1))
    run_experiment(domain, ConditionBCoOptimizer(2))

    exit()

    np.random.seed(42)
    evaluator = fitness.Speedy(c.NUM_WORLDS)
    # try:
    #     best_agent = np.load('controller.npy')
    #     best_topo = np.load('topography.npy')
    # except Exception:
    best_agent, best_topo = evolve(evaluator)
    np.save('controller.npy', best_agent)
    np.save('topography.npy', best_topo)

    # Random agent
    # best_agent = agent.randomize(1)[0]

    # Null agent (does nothing)
    best_agent = np.zeros((), dtype=agent.AGENT_DTYPE)

    # Fixed topography
    # render_fixed_topology(simulator.topo)

    # Null topography (empty space)
    # np.zeros(c.WORLD_SHAPE, dtype=np.float32)

    simulator = Simulator()
    simulator.topographies.from_numpy(
        einops.repeat(best_topo, 'w h -> 1 w h'))
    simulator.agents.from_numpy(
        einops.repeat(np.array([best_agent]), '1 -> 1 o', o=c.NUM_OBJECTS))

    visualize.show(simulator, debug=False)
    # visualize.save(simulator, 'output/sample.mp4')
    # visualize.save_batch(simulator, 'output/sample{i}.mp4', 5)
