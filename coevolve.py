from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import trange

class Domain(ABC):
    def __init__(self):
        self.metrics = []

    @abstractmethod
    def evaluate(self, interactions):
        ...

    # List of Metrics p_i = f(X_0 ... X_n) -> R
    # 0 .. n : N roles in this domain.
    # X_i is a set of entities playing role i (a population)
    # A tuple (x_0 ... x_n) in X_0 x ... x X_n is an interaction
    # R is the set of outcomes

class CoOptimizer(ABC):
    @abstractmethod
    def get_interactions(self, scores=None):
        ...

    @abstractmethod
    def score_interactions(self, interactions, metrics):
        ...

    @abstractmethod
    def overall_score(self, metrics):
        ...

    @abstractmethod
    def best_interaction(self, scores):
        ...

    # For a Domain...
    # Generate candidate solutions (an interaction of entities from some subset of roles)
    # Provide an ordering of candidates

def coevolve(domain, cooptimizer, num_iterations):
    interactions = cooptimizer.get_interactions()
    history = []

    progress = trange(num_iterations)
    overall_score = 0.0
    for i in progress:
        progress.set_description(f'Score == {overall_score:4.2f}')
        metrics = domain.evaluate(interactions)
        scores = cooptimizer.score_interactions(metrics)
        overall_score = cooptimizer.overall_score(metrics)
        history.append(metrics | scores)
        if i + 1 < num_iterations:
            interactions = cooptimizer.get_interactions(scores)
    return (cooptimizer.best_interaction(interactions, scores),
            pd.DataFrame(history))
