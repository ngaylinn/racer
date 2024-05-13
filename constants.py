"""Hyperparameters for this project, its simulations, and experiments.
"""

# How many individuals do we simulate, and how many times?
NUM_TRIALS = 5
NUM_INDIVIDUALS = 100
NUM_MATCH_UPS = 16
NUM_WORLDS = NUM_TRIALS * NUM_INDIVIDUALS * NUM_MATCH_UPS

# Properties of the simulation itself
NUM_STEPS = 1000
NUM_GENERATIONS = 100
WORLD_SIZE = 512
WORLD_SHAPE = (WORLD_SIZE, WORLD_SIZE)
DT = 0.01
VISCOSITY = 0.005

# Properties of the balls in the simulation.
NUM_BALLS = 6
MIN_BALL_RADIUS_PX = 2.5
MIN_BALL_RADIUS = MIN_BALL_RADIUS_PX / WORLD_SIZE
MAX_BALL_RADIUS_PX = 10
MAX_BALL_RADIUS = MAX_BALL_RADIUS_PX / WORLD_SIZE
VIEW_RADIUS_PX = 100
VIEW_RADIUS = VIEW_RADIUS_PX / WORLD_SIZE
