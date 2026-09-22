"""Iterated Local Search für eine Rundtour (TSP): ein bereits gutes lokales Optimum wird mit einer Doppelbrücke (ils_kick.double_bridge,
feste Störstärke) gestört und neu abgestiegen - anders als Hill Climbing mit Neustarts wird die Tour dabei nicht weggeworfen, nur die
Umgebung der Störung neu durchsucht. Eine Iteration = Kick + Wiederabstieg; angenommen wird die neue Tour nur, wenn sie mindestens so
kurz wie die aktuelle ist (klassisches ILS, "better"), sonst geht es von der bisherigen Tour aus weiter. Budget = bewertete Nachbarn
(Kandidatenpaare bei Kandidatenliste + Don't-Look-Bits, bzw. bewertete Nachbarn beim vollen Rescan) - derselbe Vorschlags-Begriff wie in
der ganzen Trajektorien-Metaheuristiken-Linie; der Kick selbst zählt nicht (er ist keine bewertete Nachbarschaft, sondern eine feste
Konstruktion)."""

from dataclasses import dataclass, field

import numpy as np

import ils_dlb as DLB
import ils_kick as K
import ils_tour as T

LOCAL_SEARCHES = ("dlb", "full")
ACCEPT_RULES = ("better", "always")


@dataclass
class Run:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    evaluations: int = 0
    iterations: int = 0
    accepted: int = 0
    snapshots: list = field(default_factory=list)          # aktuelle Tour nach jeder Iteration (nur bei keep_snapshots=True)
    trace_iter: np.ndarray = None
    trace_length: np.ndarray = None                        # aktuelle (angenommene) Länge über die Iterationen
    trace_best: np.ndarray = None
    local_search: str = "dlb"
    n_bridges: int = 1
    accept: str = "better"


def run(D, start, cand=None, local_search="dlb", n_bridges=1, accept="better", budget=100000, seed=0, keep_snapshots=True, trace_points=300):
    """Ein Lauf. `cand` (Kandidatenlisten aus ils_dlb.build_candidate_lists) ist nur für `local_search="dlb"` nötig.
    `budget` = Zahl der bewerteten Nachbarn insgesamt (über alle Wiederabstiege zusammen); ein Kick wird nicht mitgezählt."""
    if local_search not in LOCAL_SEARCHES:
        raise ValueError(local_search)
    if accept not in ACCEPT_RULES:
        raise ValueError(accept)
    if local_search == "dlb" and cand is None:
        raise ValueError("local_search='dlb' braucht Kandidatenlisten (cand)")
    rng = np.random.default_rng(seed)
    evaluations = iterations = accepted = 0
    trace_every = max(1, budget // trace_points)

    # Erster Abstieg aus der Startlösung (touched=None: die ganze Tour muss optimiert werden, nicht nur eine Umgebung) -
    # das eigentliche ILS beginnt erst danach, mit Kicks AUF einem bereits lokal optimalen Tour.
    init_seed = int(rng.integers(0, 2**31 - 1))
    if local_search == "dlb":
        r0 = DLB.dlb_descend(D, start, cand, seed=init_seed, max_evaluations=budget)
        current, current_length = r0.tour, r0.length
    else:
        r0 = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=budget)
        current, current_length = r0.tour, r0.length
    evaluations += r0.evaluations
    best_tour, best_length = current.copy(), current_length
    snapshots = [current.copy()] if keep_snapshots else []
    tr_it, tr_len, tr_best = [evaluations], [current_length], [current_length]

    while evaluations < budget:
        remaining = budget - evaluations
        kicked, touched = K.double_bridge(current, rng, n_bridges=n_bridges)
        descend_seed = int(rng.integers(0, 2**31 - 1))
        if local_search == "dlb":
            r = DLB.dlb_descend(D, kicked, cand, seed=descend_seed, max_evaluations=remaining, touched=touched)
            candidate_tour, candidate_length, spent = r.tour, r.length, r.evaluations
        else:
            r = T.descend(D, kicked, "2opt", "first", keep_steps=False, max_evaluations=remaining)
            candidate_tour, candidate_length, spent = r.tour, r.length, r.evaluations
        evaluations += spent
        iterations += 1
        take = candidate_length <= current_length + 1e-9 if accept == "better" else True
        if take:
            current, current_length = candidate_tour, candidate_length
            accepted += 1
        if current_length < best_length - 1e-9:
            best_length, best_tour = current_length, current.copy()
        if keep_snapshots:
            snapshots.append(current.copy())
        if iterations % trace_every == 0 or evaluations >= budget:
            tr_it.append(evaluations)
            tr_len.append(current_length)
            tr_best.append(best_length)

    final_length = T.tour_length(current, D)                          # Rundungsfehler der Delta-Summen beseitigen
    best_length = T.tour_length(best_tour, D)
    return Run(best_tour, best_length, current, final_length, evaluations, iterations, accepted, snapshots,
               np.array(tr_it), np.array(tr_len), np.array(tr_best), local_search, n_bridges, accept)
