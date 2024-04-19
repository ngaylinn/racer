import einops
from neatchi import Neat
import numpy as np
import taichi as ti
from tqdm import trange

import agent
from coevolve import Domain, CoOptimizer, coevolve, pair_select
import constants as c
import fitness
from simulator import Simulator
import visualize

ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32,
        debug=False)


# Make sure all pairings of parents with mates put the most fit individual in
# the "parent" role. This tells the Neat algorithm to take excess genes from
# the fittest individual, which should bias evolution in a positive way.
def prioritize_fittest(parents, mates, fitness):
    for i in range(len(fitness)):
        p, m = parents[i], mates[i]
        if fitness[p] < fitness[m]:
            parents[i], mates[i] = mates[i], parents[i]

# def visualize_population(topo_generators, fitness_scores):
#     temp = ti.field(float, shape=(50, 128, 128))
#     topo_generators.render_all(temp)
#     topos = einops.rearrange(
#         temp.to_numpy(), '(gr gc) tr tc -> (gr tr) (gc tc)',
#         gr=10, gc=5)
#     gui = ti.GUI('Racer', (128*10, 128*5),
#                  background_color=0xffffff, show_gui=True)
#     while gui.running:
#         gui.set_image(topos)
#         for i in range(50):
#             x, y = divmod(i, 5)
#             gui.text(f'{fitness_scores[i]:0.3f}',
#                      (x / 10.0, (y + 1) / 5.0), color=0xFF00FF)
#         gui.show()
# 
# 
# def evolve(evaluator):
#     print(
#         f'Evolving {NUM_INDIVIDUALS} controllers and topography generators,\n'
#         f'with {c.NUM_TRIALS} trials each ({c.NUM_WORLDS} parallel simulations)')
#     # TODO: Restore
#     # agents = agent.randomize(NUM_INDIVIDUALS)
#     agents = np.zeros(NUM_INDIVIDUALS, dtype=agent.AGENT_DTYPE)
#     neat = Neat(num_inputs=2, num_outputs=1, num_individuals=NUM_INDIVIDUALS,
#                 num_repeats=c.NUM_TRIALS)
#     topo_generators = neat.random_population()
#     simulator = Simulator(c.NUM_WORLDS)
#     score = 0
#     progress = trange(c.NUM_GENERATIONS)
#     for generation in progress:
#         progress.set_description(f'Mean fitness = {score:4.2f}')
#         # Populate the simulator with random objects and copies of the evolved
#         # agents.
#         simulator.randomize_objects()
#         topo_generators.render_all(simulator.topographies)
#         simulator.agents.from_numpy(
#             einops.repeat(
#                 agents, 'i -> (i t) o',
#                 i=NUM_INDIVIDUALS, t=c.NUM_TRIALS, o=c.NUM_OBJECTS))
# 
#         # Actually simulate all those worlds.
#         for _ in range(c.NUM_STEPS):
#             simulator.step()
# 
#         # Score fitness for each world, then average scores across trials.
#         fitness_scores = einops.reduce(
#             evaluator.score_all(simulator), '(i t) -> i', np.mean,
#             i=NUM_INDIVIDUALS, t=c.NUM_TRIALS)
#         best_index = fitness_scores.argmax()
#         score = np.mean(fitness_scores)
# 
# 
#         if generation < c.NUM_GENERATIONS - 1:
#             parent_selections = fitness.select(fitness_scores, NUM_INDIVIDUALS)
#             mate_selections = fitness.select(fitness_scores, NUM_INDIVIDUALS)
#             prioritize_fittest(
#                 parent_selections, mate_selections, fitness_scores)
#             topo_generators = neat.propagate(parent_selections, mate_selections)
#             # TODO: Restore
#             # agents = agent.mutate(agents[parent_selections])
# 
#     # Note: Only works with NUM_TRIALS set to 1.
#     # visualize_population(topo_generators, fitness_scores)
#     return (agents[best_index].astype(agent.AGENT_DTYPE),
#             topo_generators.render_one(best_index, c.WORLD_SHAPE))
# 
@ti.kernel
def render_fixed_topology(topo: ti.template()):
    center = ti.math.vec2(256.0, 256.0)
    max_dist = ti.math.distance(ti.math.vec2(0.0, 0.0), center)
    for w, x, y in topo:
        dist = ti.math.distance((x, y), center) / max_dist
        # Draw a steep column in the center, surrounded by a shallow bowl.
        topo[w, x, y] = 1.0 - dist**2 # ti.select(dist < 0.1, 0.0, 1.0 - dist**2)




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

    def evaluate(self, interactions):
        topo_generators, controllers = interactions
        self.simulator.randomize_objects()
        #render_fixed_topology(self.simulator.topographies)
        topo_generators.render_all(self.simulator.topographies)
        #self.simulator.agents.from_numpy(controllers)
        self.simulator.controllers = controllers
        for _ in range(c.NUM_STEPS):
            self.simulator.step()
        self.summarize_simulation()
        return {
            'dist': np.nan_to_num(self.dist.to_numpy()),
            'hits': np.nan_to_num(self.hits.to_numpy())
        }

class RacerCoOptimizer(CoOptimizer):
    def __init__(self, name):
        self.name = name
        self.topography_neat = Neat(
            num_inputs=2, num_outputs=1,
            num_individuals=c.NUM_INDIVIDUALS, num_repeats=c.NUM_TRIALS)
        # self.controllers = np.zeros(c.NUM_INDIVIDUALS, dtype=agent.AGENT_DTYPE)
        self.controller_neat = Neat(
            num_inputs=agent.NUM_INPUTS, num_outputs=agent.NUM_OUTPUTS,
            num_individuals=c.NUM_INDIVIDUALS, num_repeats=c.NUM_TRIALS)

    def overall_score(self, metrics):
        return np.mean(metrics['dist'] / (1 + metrics['hits']))


def fixed_world_assignments():
    return np.tile(np.arange(c.NUM_INDIVIDUALS), c.NUM_TRIALS)

def random_world_assignments():
    return np.concatenate([
        np.random.permutation(c.NUM_INDIVIDUALS)
        for _ in range(c.NUM_TRIALS)])

def reduce_fitness(fitness_scores, world_assignments):
    result = np.zeros(c.NUM_INDIVIDUALS)
    for score, i in zip(fitness_scores, world_assignments):
        result[i] += score
    return result / c.NUM_TRIALS


class ConditionACoOptimizer(RacerCoOptimizer):
    def __init__(self):
        super().__init__('condition_a')

    def get_interactions(self, scores=None):
        if scores is None:
            topo_generators = self.topography_neat.random_population()
            controllers = self.controller_neat.random_population()
        else:
            matches = pair_select(scores['combined'])
            topo_generators = self.topography_neat.propagate(matches)
            controllers = self.controller_neat.propagate(matches)
        world_assignments = fixed_world_assignments()
        return (topo_generators,
                controllers.get_controllers(world_assignments, c.NUM_OBJECTS))

    def score_interactions(self, interactions, metrics):
        _, controllers = interactions
        world_assignments = controllers.world_assignments.to_numpy()
        combined = reduce_fitness(
            metrics['dist'] / (1 + metrics['hits']),
            world_assignments)
        return {
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
        if scores is None:
            topo_generators = self.topography_neat.random_population()
        else:
            topo_generators = self.topography_neat.propagate(
                pair_select(scores['topography']))
        return (topo_generators, self.controllers)

    def score_interactions(self, metrics):
        # TODO: Handle all topography / controller pairings.
        dist = metrics['dist']
        inv_hits = 1 / (1 + metrics['hits'])
        return {
            'topography': dist if self.case == 1 else inv_hits,
            'controller': inv_hits if self.case == 1 else dist,
            'combined': dist * inv_hits
        }

    def best_interaction(self, interactions, scores):
        topo_generators, controllers = interactions
        topo_scores, controller_scores = scores
        combined = topo_scores * controller_scores
        # TODO: What if the best controller and topography have different
        # indices?
        best_index = np.argmax(combined)
        return (topo_generators.render_one(best_index, c.WORLD_SHAPE),
                controllers.compile_one(best_index))


def record_simulation(topography, controller, name):
    simulator = Simulator()
    simulator.topographies.from_numpy(
        einops.repeat(topography, 'w h -> 1 w h'))
    #render_fixed_topology(simulator.topographies)
    simulator.controllers = controller
    visualize.show(simulator, debug=False)
    # visualize.save(simulator, f'output/{name}.mp4')


def summarize_results(history, name):
    # TODO: Generate charts and stuff.
    pass


def run_experiment(domain, cooptimizer):
    print(f'Running experiment {cooptimizer.name} '
          f'with {c.NUM_WORLDS} parallel simulations:')
    (best_topography, best_controller), history = coevolve(
        domain, cooptimizer, c.NUM_GENERATIONS)
    print('Summarizing results...')
    record_simulation(best_topography, best_controller, cooptimizer.name)
    summarize_results(history, cooptimizer.name)
    print('Done!\n')


if __name__ == '__main__':
    simulator = Simulator(c.NUM_WORLDS)
    domain = RacerDomain(simulator)
    run_experiment(domain, ConditionACoOptimizer())
    # run_experiment(domain, ConditionBCoOptimizer(1))
    # run_experiment(domain, ConditionBCoOptimizer(2))

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
