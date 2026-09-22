"""Auswertung der Iterated-Local-Search-Demo: ein Lauf gegen Hill Climbing mit Neustarts (voller Rescan UND Kandidatenliste + Don't-Look-Bits,
gleiches Bewertungsbudget), Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp); er überschätzt die wahre Lücke
um die Schrankenlücke (im Mittel unter 1 %). Ein Vorschlag ist eine bewertete Nachbarschaft - bei Kandidatenliste + Don't-Look-Bits ein
geprüftes Kandidatenpaar, beim vollen Rescan ein bewerteter Nachbar im Abstieg; beide Neustart-Varianten zählen ihre Bewertungen genauso."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import ils_algorithm as ILS
import ils_constants as C
import ils_dlb as DLB
import ils_scenario as S
import ils_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    local_search: str = C.DEFAULT_LOCAL_SEARCH
    n_bridges: int = C.DEFAULT_N_BRIDGES
    accept: str = C.DEFAULT_ACCEPT
    budget: int = C.DEFAULT_BUDGET
    start: str = C.DEFAULT_START
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    """1-Baum-Schranke; Ziel des Subgradientenverfahrens ist die Länge eines guten lokalen Optimums (Nächster Nachbar + 2-opt/Or-opt, steilster Abstieg)."""
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


@lru_cache(maxsize=256)
def candidate_lists(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    return DLB.build_candidate_lists(D)


def make_start(settings, D):
    if settings.start == "nearest":
        return T.nearest_neighbor_tour(D)
    return T.random_tour(len(D), np.random.default_rng(settings.chain_seed))


def hill_climbing_restarts(D, budget, seed):
    """Hill Climbing mit Neustarts, voller Rescan (wie in der Wurzel-Demo): der erste Abstieg wird immer zu Ende geführt, weitere nur mit dem Rest des Budgets."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), "2opt", "first", keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


def dlb_restarts(D, cand, budget, seed):
    """Wie hill_climbing_restarts, mit Kandidatenliste + Don't-Look-Bits statt vollem Rescan - die FAIRE Vergleichsgröße für ILS,
    das dieselbe lokale Suche benutzt. Gibt (beste Tour, Starts, verbrauchte Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    used, starts, best_tour, best_length = 0, 0, None, None
    while used < budget or best_tour is None:
        cap = None if best_tour is None else budget - used
        r = DLB.dlb_descend(D, T.random_tour(len(D), rng), cand, seed=starts, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best_length is None or r.length < best_length:
            best_length, best_tour = r.length, r.tour
    return best_tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    start_tour: np.ndarray
    run: object
    seconds: float
    hc: object                      # ein Hill-Climbing-Abstieg (voller Rescan) aus derselben Startlösung
    hc_seconds: float
    hcr_tour: np.ndarray             # Hill Climbing mit Neustarts, voller Rescan, gleiches Budget
    hcr_starts: int
    hcr_seconds: float
    dlbr_tour: np.ndarray             # Hill Climbing mit Neustarts, Kandidatenliste + DLB, gleiches Budget (faire Vergleichsgröße)
    dlbr_starts: int
    dlbr_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def final_gap(self):
        return self.gap_of(self.run.final_length)

    @property
    def hc_gap(self):
        return self.gap_of(self.hc.length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def dlbr_gap(self):
        return self.gap_of(T.tour_length(self.dlbr_tour, self.D))

    @property
    def start_gap(self):
        return self.gap_of(T.tour_length(self.start_tour, self.D))

    @property
    def accept_rate(self):
        return self.run.accepted / max(self.run.iterations, 1)


def analyse(settings, keep_snapshots=True, with_hc=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    start = make_start(settings, D)
    cand = candidate_lists(settings.n, settings.cluster_share, settings.seed) if settings.local_search == "dlb" else None
    t0 = time.perf_counter()
    run = ILS.run(D, start, cand=cand, local_search=settings.local_search, n_bridges=settings.n_bridges, accept=settings.accept,
                   budget=settings.budget, seed=settings.chain_seed, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    hc = hc_tour = dlbr_tour = None
    hc_seconds = hcr_seconds = dlbr_seconds = 0.0
    hcr_starts = dlbr_starts = 0
    if with_hc:
        t0 = time.perf_counter()
        hc = T.descend(D, start, "2opt", "first", keep_steps=False)
        hc_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hc_tour, hcr_starts, _ = hill_climbing_restarts(D, settings.budget, settings.chain_seed)
        hcr_seconds = time.perf_counter() - t0
        cand_hc = cand if cand is not None else DLB.build_candidate_lists(D)
        t0 = time.perf_counter()
        dlbr_tour, dlbr_starts, _ = dlb_restarts(D, cand_hc, settings.budget, settings.chain_seed)
        dlbr_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, start, run, seconds, hc, hc_seconds, hc_tour, hcr_starts, hcr_seconds,
                     dlbr_tour, dlbr_starts, dlbr_seconds, T.count_crossings(inst.xy, run.best_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.3                  # so viel besser als Hill Climbing mit Neustarts (Kandidatenliste + DLB) gilt als Sieg (Prozentpunkte)
LOSE_MARGIN = 0.3                 # so viel schlechter gilt als Niederlage


def verdict(a):
    """Code: beats_hc (deutlich besser als Hill Climbing mit Neustarts UND Kandidatenliste + Don't-Look-Bits bei gleichem Budget - die faire
    Vergleichsgröße, da ILS dieselbe lokale Suche nutzt), hc_wins (Neustarts sind besser), comparable. Gilt für diesen einen Lauf - die Ketten streuen."""
    if a.gap <= a.dlbr_gap - WIN_MARGIN:
        return "beats_hc"
    if a.dlbr_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes` (Instanz- und Ketten-Seed von `base` werden überschrieben)."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "final": a.final_gap, "hc": a.hc_gap, "hcr": a.hcr_gap, "dlbr": a.dlbr_gap, "hcr_starts": a.hcr_starts,
                         "dlbr_starts": a.dlbr_starts, "seconds": a.seconds, "hc_seconds": a.hc_seconds, "accept_rate": a.accept_rate,
                         "iterations": a.run.iterations, "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "n_bridges": (1, 2, 3, 5, 8),
                "accept": ILS.ACCEPT_RULES, "local_search": ILS.LOCAL_SEARCHES, "n": (10, 20, 40, 60, 100, 150, 200),
                "cluster_share": (0, 25, 50, 75, 100), "start": ("random", "nearest")}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarschaften)", "n_bridges": "Störstärke (Doppelbrücken je Kick)", "accept": "Annahme",
                "local_search": "Lokale Suche", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)", "start": "Startlösung"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: ILS (beste Tour) und ein Hill-Climbing-Abstieg (voller Rescan) aus derselben zufälligen Startlösung."""
    ils_gaps, hc_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch, start="random"), keep_snapshots=False)
        ils_gaps.append(a.gap)
        hc_gaps.append(a.hc_gap)
    return {"ils": np.array(ils_gaps), "hc": np.array(hc_gaps)}
