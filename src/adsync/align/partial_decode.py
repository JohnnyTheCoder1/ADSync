"""Sparse partial alignment with affine gaps and exact path score margins.

Each matched run needs three consecutive, rate-compatible observations.
Its score is the sum of correlation scores times the observation step. An
internal edit costs 0.75 plus 0.01 per unsupported source/target second;
unsupported leading/trailing seconds cost 0.01 each. The empty path is valid.
Scores are objective values, not probabilities or calibrated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from adsync.models import CandidateWindow

ALGORITHM = "partial-affine-gap-v1"
MAX_CANDIDATES = 40_000
GAP_OPEN = .75
GAP_EXTEND = .01
TIMING_EQUIVALENCE_SEC = .05


@dataclass(frozen=True)
class PartialNode:
    window_index: int
    candidate_index: int
    source_time: float
    target_time: float
    score: float
    provenance: str


@dataclass
class PartialPath:
    nodes: list[PartialNode]
    selected: list[int]
    runs: list[list[int]]
    path_score: float
    window_margins: list[float | None]
    input_candidates: int


def decode_partial_path(
    lattice: list[CandidateWindow],
    ad_duration: float,
    video_duration: float,
    *,
    window_sec: float = 2.,
    step_sec: float = 1.,
    max_stretch: float = .01,
    max_candidates: int = MAX_CANDIDATES,
) -> PartialPath:
    """Find the best partial path in O(M²) time and O(M + windows) memory.

    The admission limit is explicit: this operates on sparse hypotheses,
    never a dense audio-frame cost matrix. Alternative margins include paths
    that skip the selected observation, not just its competing candidates.
    Measurements within 50 ms describe the same placement for this comparison.
    """
    for name, value in (("ad_duration", ad_duration), ("video_duration", video_duration),
                        ("window_sec", window_sec), ("step_sec", step_sec)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(max_stretch) or not 0 <= max_stretch < 1:
        raise ValueError("max_stretch must be finite and in [0, 1)")
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")

    nodes: list[PartialNode] = []
    previous = -math.inf
    count = 0
    for wi, window in enumerate(lattice):
        time = window.source_center
        if (not math.isfinite(time) or not 0 <= time <= ad_duration or time <= previous
                or not math.isfinite(window.energy) or not math.isfinite(window.speech_score)):
            raise ValueError("Window times must be finite, ordered, unique and inside the source duration")
        previous = time
        count += len(window.candidates)
        for ci, candidate in enumerate(window.candidates):
            values = (candidate.offset_sec, candidate.score, candidate.peak_ratio, candidate.peak_sharpness)
            if not all(math.isfinite(v) for v in values) or not 0 <= candidate.score <= 1.0001:
                raise ValueError("Candidate measurements must be finite with a score in [0, 1]")
            target = time + candidate.offset_sec
            if candidate.score >= .3 and 0 <= target <= video_duration:
                nodes.append(PartialNode(wi, ci, float(time), float(target),
                                         min(1., float(candidate.score)), candidate.source))
    if len(nodes) > max_candidates:
        raise ValueError(f"Partial alignment has {len(nodes)} eligible candidates; limit is {max_candidates}. "
                         "Increase the analysis step or reduce hypotheses before decoding.")

    empty_score = -GAP_EXTEND * (ad_duration + video_duration)
    margins: list[float | None] = [None] * len(lattice)
    if not nodes:
        return PartialPath(nodes, [], [], empty_score, margins, count)

    source = np.array([n.source_time for n in nodes])
    target = np.array([n.target_time for n in nodes])
    windows = np.array([n.window_index for n in nodes])
    reward = np.array([n.score * step_sec for n in nodes])
    half = window_sec / 2
    prefix = -GAP_EXTEND * (np.maximum(0, source - half) + np.maximum(0, target - half))
    suffix = -GAP_EXTEND * (np.maximum(0, ad_duration - source - half)
                            + np.maximum(0, video_duration - target - half))

    def links(left, right):
        dt, dv = source[right] - source[left], target[right] - target[left]
        valid = (dt > 0) & (dv > 0)
        continuation = valid & (dt <= 1.5 * step_sec + 1e-9) & (np.abs(dv - dt) <= max_stretch * dt + 1e-9)
        edit = valid & ~continuation
        penalty = -GAP_OPEN - GAP_EXTEND * (np.maximum(0, dt - window_sec) + np.maximum(0, dv - window_sec))
        return continuation, edit, penalty

    # Ages 0, 1, 2 mean one, two, or at least three observations in this run.
    size = len(nodes)
    forward = np.full((size, 3), -np.inf)
    parents = np.full((size, 3), -1, dtype=np.int64)
    parent_ages = np.full((size, 3), -1, dtype=np.int8)
    for j in range(size):
        forward[j, 0] = prefix[j] + reward[j]
        if not j:
            continue
        cont, edit, penalty = links(slice(0, j), j)
        choices = np.where(edit, forward[:j, 2] + penalty, -np.inf)
        best = int(np.argmax(choices))
        if choices[best] > prefix[j]:
            forward[j, 0] = choices[best] + reward[j]
            parents[j, 0], parent_ages[j, 0] = best, 2
        choices = np.where(cont, forward[:j, 0], -np.inf)
        best = int(np.argmax(choices))
        forward[j, 1] = choices[best] + reward[j]
        if np.isfinite(choices[best]):
            parents[j, 1], parent_ages[j, 1] = best, 0
        choices = np.where(cont[:, None], forward[:j, 1:], -np.inf)
        flat = int(np.argmax(choices))
        best, age = divmod(flat, 2)
        forward[j, 2] = choices[best, age] + reward[j]
        if np.isfinite(choices[best, age]):
            parents[j, 2], parent_ages[j, 2] = best, age + 1

    endings = forward[:, 2] + suffix
    last = int(np.argmax(endings))
    optimum = max(empty_score, float(endings[last]))
    if optimum <= empty_score:
        return PartialPath(nodes, [], [], optimum, margins, count)
    selected, ages = [], []
    j, age = last, 2
    while j >= 0:
        selected.append(j)
        ages.append(age)
        j, age = int(parents[j, age]), int(parent_ages[j, age])
    selected.reverse()
    ages.reverse()
    runs: list[list[int]] = []
    for node, age in zip(selected, ages):
        if age == 0:
            runs.append([])
        runs[-1].append(node)

    backward = np.full((size, 3), -np.inf)
    for i in range(size - 1, -1, -1):
        backward[i, 2] = suffix[i]
        if i == size - 1:
            continue
        cont, edit, penalty = links(i, slice(i + 1, size))
        future = reward[i + 1:]
        backward[i, 0] = np.max(np.where(cont, future + backward[i + 1:, 1], -np.inf))
        continuation = np.max(np.where(cont, future + backward[i + 1:, 2], -np.inf))
        backward[i, 1] = continuation
        restart = np.max(np.where(edit, penalty + future + backward[i + 1:, 0], -np.inf))
        backward[i, 2] = max(backward[i, 2], continuation, restart)

    # A path omits a window by starting after it, ending before it, or crossing
    # it in one edge. Prefix maxima evaluate all those crossing edges without
    # materializing the quadratic graph or rerunning DP once per window.
    n_windows = len(lattice)
    start_at = np.full(n_windows, -np.inf)
    end_at = np.full(n_windows, -np.inf)
    np.maximum.at(start_at, windows, prefix + reward + backward[:, 0])
    np.maximum.at(end_at, windows, endings)
    bypass = np.full(n_windows, empty_score)
    if n_windows > 1:
        bypass[:-1] = np.maximum(bypass[:-1], np.maximum.accumulate(start_at[::-1])[::-1][1:])
        bypass[1:] = np.maximum(bypass[1:], np.maximum.accumulate(end_at)[:-1])
    first_node = np.searchsorted(windows, np.arange(n_windows))
    for j in range(1, size):
        wi = windows[j]
        if wi < 1:
            continue
        cont, edit, penalty = links(slice(0, j), j)
        continuation = np.maximum(forward[:j, 0] + backward[j, 1],
            np.maximum(forward[:j, 1], forward[:j, 2]) + backward[j, 2])
        scores = np.maximum(np.where(cont, continuation, -np.inf),
                            np.where(edit, forward[:j, 2] + penalty + backward[j, 0], -np.inf)) + reward[j]
        best_prefix = np.maximum.accumulate(scores)
        cuts = first_node[:wi] - 1
        valid = cuts >= 0
        indices = np.flatnonzero(valid)
        bypass[indices] = np.maximum(bypass[indices], best_prefix[cuts[valid]])

    conditional = np.max(forward + backward, axis=1)
    for node in selected:
        wi = nodes[node].window_index
        lo, hi = np.searchsorted(windows, [wi, wi + 1])
        distinct = np.abs(target[lo:hi] - target[node]) > TIMING_EQUIVALENCE_SEC
        alternative = max(float(bypass[wi]), float(np.max(conditional[lo:hi][distinct], initial=-np.inf)))
        margins[wi] = max(0., optimum - alternative)
    return PartialPath(nodes, selected, runs, optimum, margins, count)
