"""
Baseline models for temporal link prediction.

Each baseline implements a score(src, dst, t) method that returns a scalar
score indicating the likelihood of an interaction (src, dst) at time t.

Baselines:
  1. RandomBaseline: random scores (lower bound)
  2. HistoricalFrequency: count of past (src, dst) interactions
  3. RecencyBaseline: inverse time since last (src, dst) interaction
  4. EdgeBank: binary memory of recently seen edges (TGB standard baseline)
"""

import numpy as np
from collections import defaultdict


class RandomBaseline:
    """Scores each candidate uniformly at random.

    Expected MRR = 1/21 ~ 0.048 (chance level with 20 negatives + 1 positive).
    Serves as an absolute lower bound.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.name = "Random"

    def fit(self, sources, destinations, timestamps):
        pass

    def score(self, src: int, dst: int, t: int) -> float:
        return self.rng.random()


class HistoricalFrequency:
    """Scores based on how often (src, dst) appeared in training data.

    Intuition: busy routes like KJFK-EGLL are more likely to have future
    flights than rare connections. This captures the static popularity
    pattern but ignores temporal dynamics.
    """

    def __init__(self):
        self.pair_counts = defaultdict(int)
        self.name = "HistFreq"

    def fit(self, sources, destinations, timestamps):
        """Count (src, dst) pairs in training data."""
        for s, d in zip(sources, destinations):
            self.pair_counts[(int(s), int(d))] += 1

    def score(self, src: int, dst: int, t: int) -> float:
        return float(self.pair_counts.get((src, dst), 0))


class RecencyBaseline:
    """Scores based on recency of the last (src, dst) interaction.

    score = 1 / (t - t_last + 1) if seen before, else 0.

    Intuition: a route with a recent flight is more likely to have
    another one soon. Captures temporal dynamics that HistFreq misses.
    """

    def __init__(self):
        self.last_seen = {}
        self.name = "Recency"

    def fit(self, sources, destinations, timestamps):
        """Record the last timestamp for each (src, dst) pair."""
        for s, d, t in zip(sources, destinations, timestamps):
            key = (int(s), int(d))
            self.last_seen[key] = int(t)

    def score(self, src: int, dst: int, t: int) -> float:
        last_t = self.last_seen.get((src, dst))
        if last_t is None:
            return 0.0
        dt = t - last_t
        if dt <= 0:
            return 1.0
        return 1.0 / dt


class EdgeBank:
    """Binary memory of recently seen edges.

    Maintains a set of (src, dst) pairs from a time window.
    score = 1 if (src, dst) is in the memory, else 0.

    This is the standard TGB baseline. We implement two variants:
    - "unlimited": remembers ALL training edges (no window)
    - "tw" (time window): only remembers edges from the last W seconds

    Reference: Poursafaei et al., "Towards Better Evaluation for Dynamic
    Link Prediction" (NeurIPS 2022)
    """

    def __init__(self, mode: str = "unlimited", time_window: int = None):
        self.mode = mode
        self.time_window = time_window
        self.memory = set()
        self.name = f"EdgeBank_{mode}"
        if mode == "tw":
            self.edge_times = defaultdict(list)

    def fit(self, sources, destinations, timestamps):
        """Build the edge memory from training data."""
        if self.mode == "unlimited":
            for s, d in zip(sources, destinations):
                self.memory.add((int(s), int(d)))
        elif self.mode == "tw":
            for s, d, t in zip(sources, destinations, timestamps):
                self.edge_times[(int(s), int(d))].append(int(t))

    def score(self, src: int, dst: int, t: int) -> float:
        if self.mode == "unlimited":
            return 1.0 if (src, dst) in self.memory else 0.0
        elif self.mode == "tw":
            times = self.edge_times.get((src, dst), [])
            for tt in reversed(times):
                if t - tt <= self.time_window:
                    return 1.0
                if tt < t - self.time_window:
                    break
            return 0.0
        return 0.0


class CommonNeighbors:
    """Structural heuristic: number of common neighbors.

    For directed graph, we use the union of in-neighbors and out-neighbors.
    score(src, dst) = |N(src) intersection N(dst)|

    This is a static heuristic computed from training data and ignores
    temporal information.
    """

    def __init__(self):
        self.neighbors = defaultdict(set)
        self.name = "CommonNeighbors"

    def fit(self, sources, destinations, timestamps):
        """Build neighbor sets from training edges."""
        for s, d in zip(sources, destinations):
            s, d = int(s), int(d)
            self.neighbors[s].add(d)
            self.neighbors[d].add(s)

    def score(self, src: int, dst: int, t: int) -> float:
        return float(len(self.neighbors[src] & self.neighbors[dst]))
