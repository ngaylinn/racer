from itertools import count

import numpy as np
import taichi as ti

import constants as c

index = count(0)

# Indexes for per-object view inputs.
POS_X = next(index)
POS_Y = next(index)
VEL_X = next(index)
VEL_Y = next(index)
SLP_X = next(index)
SLP_Y = next(index)
NRST_POS_X = next(index)
NRST_POS_Y = next(index)
NRST_VEL_X = next(index)
NRST_VEL_Y = next(index)
AVG_POS_X = next(index)
AVG_POS_Y = next(index)
NUM_INPUTS = next(index)

index = count(0)
ACC_X = next(index)
ACC_Y = next(index)
RAD = next(index)
NUM_OUTPUTS = next(index)

View = ti.types.vector(NUM_INPUTS, float)
Weights = ti.types.matrix(NUM_INPUTS, NUM_OUTPUTS, float)
Biases = ti.types.vector(NUM_OUTPUTS, float)
Activations = ti.types.vector(NUM_OUTPUTS, float)

AGENT_DTYPE = np.dtype([
    ('weights', f'{Weights.get_shape()}f4'),
    ('biases', f'{Biases.get_shape()}f4'),
    ('activation', f'{Activations.get_shape()}f4')
])


@ti.dataclass
class Agent:
    weights: Weights
    biases: Biases
    activation: Activations

    @ti.func
    def react(self, view):
        self.activation = ti.tanh(view @ self.weights + self.biases)
        return self.activation

    def heading(self):
        return ti.math.vec2(self.activation[ACC_X],
                            self.activation[ACC_Y])


def randomize(count):
    def init_values(shape):
        vals = (0.1 * np.random.randn(count * np.prod(shape)))
        return vals.astype(np.float32).reshape((count,) + shape)
    result = np.empty(count, dtype=AGENT_DTYPE)
    result['weights'] = init_values(Weights.get_shape())
    result['biases'] = init_values(Biases.get_shape())
    return result


def mutate(agents):
    def mutate_arr(arr):
        for I in np.ndindex(arr.shape):
            if np.random.random() < c.MUTATION_RATE:
                arr[I] = 2.0 * np.random.random() - 1.0
    result = agents.copy()
    mutate_arr(result['weights'])
    mutate_arr(result['biases'])
    return result

