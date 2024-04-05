import einops
import numpy as np
import taichi as ti
from tqdm import trange

import agent
import constants as c
import fitness
from simulator import Simulator
import visualize

ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32, random_seed=42)

NUM_AGENTS = 50
NUM_WORLDS = NUM_AGENTS * c.NUM_TRIALS


def evolve(evaluator):
    print(f'Evolving {NUM_AGENTS} controllers, '
          f'with {c.NUM_TRIALS} trials each '
          f'({NUM_WORLDS} parallel simulations)')
    agents = agent.randomize(NUM_AGENTS)
    simulator = Simulator(NUM_WORLDS)
    score = 0
    progress = trange(c.NUM_GENERATIONS)
    for generation in progress:
        progress.set_description(f'Mean fitness = {score:4.2f}')
        # Populate the simulator with random objects and copies of the evolved
        # agents.
        simulator.randomize_objects()
        simulator.agents.from_numpy(
            einops.repeat(
                agents, 'a -> (a t) o',
                a=NUM_AGENTS, t=c.NUM_TRIALS, o=c.NUM_OBJECTS))

        # Actually simulate all those worlds.
        for _ in range(c.NUM_STEPS):
            simulator.step()

        # Score fitness for each world, then average scores across trials.
        fitness_scores = einops.reduce(
            evaluator.score_all(simulator), '(a t) -> a', np.mean,
            a=NUM_AGENTS, t=c.NUM_TRIALS)
        best_index = fitness_scores.argmax()
        score = np.mean(fitness_scores)

        if generation < c.NUM_GENERATIONS - 1:
            selections = fitness.select(fitness_scores, NUM_AGENTS)
            agents = agent.mutate(agents[selections])

    return agents[best_index].astype(agent.AGENT_DTYPE)

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

    best_agent = evolve(evaluator)

    # Random agent
    # best_agent = agent.randomize(1)[0]

    # Null agent (does nothing)
    # best_agent = np.zeros((), dtype=agent.AGENT_DTYPE)

    simulator = Simulator()
    render_fixed_topology(simulator.topo)
    simulator.agents.from_numpy(
        einops.repeat(np.array([best_agent]), '1 -> 1 o', o=c.NUM_OBJECTS))

    visualize.show(simulator, evaluator, debug=False)
    # visualize.save(simulator, evaluator, 'output/sample.mp4')
    # visualize.save_batch(simulator, evaluator, 'output/sample{i}.mp4', 5)
