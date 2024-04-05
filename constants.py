NUM_STEPS = 1000
NUM_TRIALS = 10
NUM_GENERATIONS = 200
WORLD_SIZE = 512
WORLD_SHAPE = (WORLD_SIZE, WORLD_SIZE)
DT = 0.01

VISCOSITY = 0.005
EDGE_HEIGHT = 0.1

NUM_OBJECTS = 6
MIN_OBJ_RADIUS_PX = 5
MIN_OBJ_RADIUS = MIN_OBJ_RADIUS_PX / WORLD_SIZE
MAX_OBJ_RADIUS_PX = 30
MAX_OBJ_RADIUS = MAX_OBJ_RADIUS_PX / WORLD_SIZE

VIEW_RADIUS_PX = 100
VIEW_RADIUS = VIEW_RADIUS_PX / WORLD_SIZE

MUTATION_RATE = 0.001

MAX_ACC = 10.0
