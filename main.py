import einops
from neatchi import Neat
import numpy as np
import taichi as ti
from tqdm import trange

import agent
import constants as c
import fitness
from simulator import Simulator
import visualize

ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32, random_seed=42,
        debug=False)

NUM_INDIVIDUALS = 50
NUM_WORLDS = NUM_INDIVIDUALS * c.NUM_TRIALS

# Make sure all pairings of parents with mates put the most fit individual in
# the "parent" role. This tells the Neat algorithm to take excess genes from
# the fittest individual, which should bias evolution in a positive way.
def prioritize_fittest(parents, mates, fitness):
    for i in range(len(fitness)):
        p, m = parents[i], mates[i]
        if fitness[p] < fitness[m]:
            parents[i], mates[i] = mates[i], parents[i]

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


def evolve(evaluator):
    print(
        f'Evolving {NUM_INDIVIDUALS} controllers and topography generators,\n'
        f'with {c.NUM_TRIALS} trials each ({NUM_WORLDS} parallel simulations)')
    # TODO: Restore
    # agents = agent.randomize(NUM_INDIVIDUALS)
    agents = np.zeros(NUM_INDIVIDUALS, dtype=agent.AGENT_DTYPE)
    neat = Neat(num_inputs=2, num_outputs=1, num_individuals=NUM_INDIVIDUALS,
                num_repeats=c.NUM_TRIALS)
    topo_generators = neat.random_population()
    simulator = Simulator(NUM_WORLDS)
    score = 0
    progress = trange(c.NUM_GENERATIONS)
    for generation in progress:
        progress.set_description(f'Mean fitness = {score:4.2f}')
        # Populate the simulator with random objects and copies of the evolved
        # agents.
        simulator.randomize_objects()
        topo_generators.render_all(simulator.topographies)
        simulator.agents.from_numpy(
            einops.repeat(
                agents, 'i -> (i t) o',
                i=NUM_INDIVIDUALS, t=c.NUM_TRIALS, o=c.NUM_OBJECTS))

        # Actually simulate all those worlds.
        for _ in range(c.NUM_STEPS):
            simulator.step()

        # Score fitness for each world, then average scores across trials.
        fitness_scores = einops.reduce(
            evaluator.score_all(simulator), '(i t) -> i', np.mean,
            i=NUM_INDIVIDUALS, t=c.NUM_TRIALS)
        best_index = fitness_scores.argmax()
        score = np.mean(fitness_scores)


        if generation < c.NUM_GENERATIONS - 1:
            parent_selections = fitness.select(fitness_scores, NUM_INDIVIDUALS)
            mate_selections = fitness.select(fitness_scores, NUM_INDIVIDUALS)
            prioritize_fittest(
                parent_selections, mate_selections, fitness_scores)
            topo_generators = neat.propagate(parent_selections, mate_selections)
            # TODO: Restore
            # agents = agent.mutate(agents[parent_selections])

    # Note: Only works with NUM_TRIALS set to 1.
    # visualize_population(topo_generators, fitness_scores)
    return (agents[best_index].astype(agent.AGENT_DTYPE),
            topo_generators.render_one(best_index, c.WORLD_SHAPE))

@ti.kernel
def render_fixed_topology(topo: ti.template()):
    center = ti.math.vec2(256.0, 256.0)
    max_dist = ti.math.distance(ti.math.vec2(0.0, 0.0), center)
    for x, y in topo:
        dist = ti.math.distance((x, y), center) / max_dist
        # Draw a steep column in the center, surrounded by a shallow bowl.
        topo[x, y] = 1.0 - dist**2 # ti.select(dist < 0.1, 0.0, 1.0 - dist**2)

if __name__ == '__main__':
    np.random.seed(42)
    evaluator = fitness.Speedy(NUM_WORLDS)
    try:
        best_agent = np.load('controller.npy')
        best_topo = np.load('topography.npy')
    except Exception:
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

    visualize.show(simulator, evaluator, debug=False)
    # visualize.save(simulator, evaluator, 'output/sample.mp4')
    # visualize.save_batch(simulator, evaluator, 'output/sample{i}.mp4', 5)
