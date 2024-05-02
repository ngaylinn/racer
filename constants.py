# TODO: Tune these values better. Right now, they're configured to fill the GPU
# with as many individuals, with each trial run separately. It would probably
# be better to run many trials in parallel, with fewer individuals, but that
# would take more extensive refactoring.

NUM_STEPS = 1000
NUM_MATCH_UPS = 16 # 10
NUM_TRIALS = 5
NUM_GENERATIONS = 100
NUM_INDIVIDUALS = 100 # 50
NUM_WORLDS = NUM_INDIVIDUALS * NUM_MATCH_UPS
WORLD_SIZE = 512
WORLD_SHAPE = (WORLD_SIZE, WORLD_SIZE)
DT = 0.01

VISCOSITY = 0.005
EDGE_HEIGHT = 0.1

NUM_OBJECTS = 6
MIN_OBJ_RADIUS_PX = 2.5
MIN_OBJ_RADIUS = MIN_OBJ_RADIUS_PX / WORLD_SIZE
MAX_OBJ_RADIUS_PX = 10
MAX_OBJ_RADIUS = MAX_OBJ_RADIUS_PX / WORLD_SIZE

VIEW_RADIUS_PX = 100
VIEW_RADIUS = VIEW_RADIUS_PX / WORLD_SIZE

MUTATION_RATE = 0.001

MAX_ACC = 10.0
