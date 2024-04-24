from itertools import count

import taichi as ti

index = count(0)
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
Reaction = ti.types.vector(NUM_OUTPUTS, float)
