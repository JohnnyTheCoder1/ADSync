"""Viterbi / DP decoder — find the globally optimal offset path through the
candidate lattice.

Penalises offset jumps and curvature (second derivative of offset) so that
musically-confused intro regions are forced to follow the globally consistent
offset instead of chasing false local correlation peaks.
"""

from __future__ import annotations

import logging

import numpy as np

from adsync.models import CandidateWindow, WarpPoint

log = logging.getLogger("adsync")

# Sentinel for "no valid predecessor"
_INF = float("inf")


def decode_warp_path(
    lattice: list[CandidateWindow],
    *,
    lambda_jump: float = 2.0,
    lambda_curve: float = 5.0,
    lambda_speech: float = 0.3,
    drift_hint_ppm: float | None = None,
    offset_hint: float | None = None,
) -> list[WarpPoint]:
    """Run Viterbi over the candidate lattice and return the decoded path.

    Parameters
    ----------
    lattice : list[CandidateWindow]
        One entry per analysis window, each with 0..K offset candidates.
    lambda_jump : float
        Penalty per second of offset deviation from expected drift.
    lambda_curve : float
        Penalty per second of curvature (change in offset velocity).
    lambda_speech : float
        Bonus weight multiplied by speech_score * match_quality.
    drift_hint_ppm : float | None
        If available, the expected drift rate (from earlier pipeline stage).
        Used to compute expected offset change between windows so the jump
        penalty targets *deviations from drift*, not absolute jumps.
    offset_hint : float | None
        If available, a rough expected offset.  Biases the first window
        toward candidates near this value.

    Returns
    -------
    list[WarpPoint]
        One point per lattice window (including synthetic fill-ins for
        empty windows).
    """
    if not lattice:
        return []

    # ── Prepare the lattice: inject synthetic pass-through candidates ───
    # For windows with no candidates, create one that inherits the median
    # offset of the nearest non-empty window.
    prepared = _prepare_lattice(lattice, offset_hint)

    N = len(prepared)  # number of time steps
    Ks = [len(w.candidates) for w in prepared]

    if all(k == 0 for k in Ks):
        log.warning("No candidates in any window — returning empty warp path")
        return []

    # Drift expectation: offset change per second
    drift_per_sec = (drift_hint_ppm or 0.0) * 1e-6

    # ── DP tables ───────────────────────────────────────────────────────
    # cost[t][k] = minimum cost to reach candidate k at time step t
    # back[t][k] = index of predecessor candidate at t-1
    # prev_delta[t][k] = offset delta that got us to (t, k) from (t-1, back[t][k])
    cost: list[list[float]] = []
    back: list[list[int]] = []
    prev_delta: list[list[float]] = []

    # ── t = 0: initialise ──────────────────────────────────────────────
    c0: list[float] = []
    for k, cand in enumerate(prepared[0].candidates):
        local = -cand.score - lambda_speech * prepared[0].speech_score * cand.score
        # Bias toward offset_hint if available
        if offset_hint is not None:
            hint_penalty = 0.3 * abs(cand.offset_sec - offset_hint)
            local += hint_penalty
        c0.append(local)
    cost.append(c0)
    back.append([-1] * Ks[0])
    prev_delta.append([0.0] * Ks[0])

    # ── t = 1..N-1: forward pass ──────────────────────────────────────
    for t in range(1, N):
        K_cur = Ks[t]
        K_prev = Ks[t - 1]
        dt = prepared[t].source_center - prepared[t - 1].source_center
        expected_delta = drift_per_sec * dt  # expected offset change

        ct: list[float] = []
        bt: list[int] = []
        dt_list: list[float] = []

        for k in range(K_cur):
            cand = prepared[t].candidates[k]
            local_reward = -cand.score - lambda_speech * prepared[t].speech_score * cand.score

            best_cost = _INF
            best_j = 0
            best_delta = 0.0

            for j in range(K_prev):
                prev_cand = prepared[t - 1].candidates[j]
                delta = cand.offset_sec - prev_cand.offset_sec
                jump = abs(delta - expected_delta)
                jump_cost = lambda_jump * jump

                # Curvature: change in velocity
                if t >= 2:
                    prev_d = prev_delta[t - 1][j]
                    curvature = abs(delta - prev_d)
                    curve_cost = lambda_curve * curvature
                else:
                    curve_cost = 0.0

                total = cost[t - 1][j] + jump_cost + curve_cost + local_reward
                if total < best_cost:
                    best_cost = total
                    best_j = j
                    best_delta = delta

            ct.append(best_cost)
            bt.append(best_j)
            dt_list.append(best_delta)

        cost.append(ct)
        back.append(bt)
        prev_delta.append(dt_list)

    # ── Backtrack ──────────────────────────────────────────────────────
    # Find best final state
    final_costs = cost[-1]
    best_final = int(np.argmin(final_costs))
    total_cost = final_costs[best_final]

    path_indices: list[int] = [0] * N
    path_indices[-1] = best_final
    for t in range(N - 2, -1, -1):
        path_indices[t] = back[t + 1][path_indices[t + 1]]

    # ── Convert to WarpPoint list ─────────────────────────────────────
    points: list[WarpPoint] = []
    for t in range(N):
        w = prepared[t]
        k = path_indices[t]
        cand = w.candidates[k]

        source_time = w.source_center
        target_time = source_time + cand.offset_sec
        confidence = cand.score * (0.5 + 0.5 * w.speech_score)

        points.append(WarpPoint(
            source_time=source_time,
            target_time=target_time,
            confidence=confidence,
        ))

    mean_conf = float(np.mean([p.confidence for p in points])) if points else 0.0
    log.info(
        "Warp decoder: %d points, path cost=%.2f, mean conf=%.3f",
        len(points), total_cost, mean_conf,
    )
    return points, total_cost


def _prepare_lattice(
    lattice: list[CandidateWindow],
    offset_hint: float | None,
) -> list[CandidateWindow]:
    """Fill empty windows with a synthetic pass-through candidate.

    For windows with zero candidates, we inject a single low-confidence
    candidate that carries the offset forward from the nearest non-empty
    window.  This keeps the trellis fully connected.
    """
    from adsync.models import OffsetCandidate

    # Find the median offset across all candidates for a fallback
    all_offsets: list[float] = []
    for w in lattice:
        for c in w.candidates:
            all_offsets.append(c.offset_sec)

    if not all_offsets:
        # Completely empty lattice — use hint or 0
        fallback_offset = offset_hint if offset_hint is not None else 0.0
    else:
        fallback_offset = float(np.median(all_offsets))

    prepared: list[CandidateWindow] = []
    last_good_offset = fallback_offset

    for w in lattice:
        if w.candidates:
            last_good_offset = w.candidates[0].offset_sec  # best candidate
            prepared.append(w)
        else:
            # Synthetic pass-through
            synthetic = OffsetCandidate(
                offset_sec=last_good_offset,
                score=0.1,  # low confidence
                peak_sharpness=1.0,
                peak_ratio=1.0,
            )
            prepared.append(CandidateWindow(
                source_center=w.source_center,
                candidates=[synthetic],
                speech_score=0.0,
                energy=w.energy,
            ))

    return prepared
