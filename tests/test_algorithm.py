"""ils_algorithm.run: Budget-Buchführung, Monotonie der besten/aktuellen Tour unter "better"-Annahme, Determinismus,
Regressionsschutz für beide Nachbarschaften der lokalen Suche (Kandidatenliste+DLB, voller Rescan), "always"-Annahme
(Random Walk über lokale Optima) als Gegenstück."""

import numpy as np
import pytest

import ils_algorithm as ILS
import ils_dlb as DLB
import ils_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


@pytest.mark.parametrize("local_search", ILS.LOCAL_SEARCHES)
def test_tour_stays_valid_and_length_matches_recomputed_length(local_search):
    D = _instance(20, 1)
    cand = DLB.build_candidate_lists(D) if local_search == "dlb" else None
    start = T.random_tour(20, np.random.default_rng(0))
    r = ILS.run(D, start, cand=cand, local_search=local_search, budget=5000, seed=3)
    assert sorted(r.best_tour.tolist()) == list(range(20))
    assert sorted(r.final_tour.tolist()) == list(range(20))
    assert r.best_length == pytest.approx(T.tour_length(r.best_tour, D), abs=1e-6)
    assert r.final_length == pytest.approx(T.tour_length(r.final_tour, D), abs=1e-6)


@pytest.mark.parametrize("local_search", ILS.LOCAL_SEARCHES)
def test_budget_is_respected_and_mostly_exhausted(local_search):
    D = _instance(25, 2)
    cand = DLB.build_candidate_lists(D) if local_search == "dlb" else None
    start = T.random_tour(25, np.random.default_rng(1))
    budget = 8000
    r = ILS.run(D, start, cand=cand, local_search=local_search, budget=budget, seed=1)
    assert r.evaluations >= budget                                     # die letzte Bewertung darf das Budget knapp überschreiten, aber nicht viel
    assert r.evaluations <= budget + 200
    assert r.iterations >= 1


def test_best_length_is_never_worse_than_the_first_local_optimum():
    D = _instance(30, 3)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(30, np.random.default_rng(2))
    first_descent = DLB.dlb_descend(D, start, cand, seed=0)
    r = ILS.run(D, start, cand=cand, local_search="dlb", budget=20000, seed=2)
    assert r.best_length <= first_descent.length + 1e-6


def test_better_acceptance_keeps_the_current_length_monotone_non_increasing():
    D = _instance(20, 4)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(20, np.random.default_rng(3))
    r = ILS.run(D, start, cand=cand, local_search="dlb", accept="better", budget=15000, seed=4)
    assert all(b <= a + 1e-6 for a, b in zip(r.trace_length, r.trace_length[1:]))


def test_always_accept_is_a_random_walk_and_can_get_worse_than_better_acceptance():
    D = _instance(30, 5)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(30, np.random.default_rng(4))
    better = ILS.run(D, start, cand=cand, local_search="dlb", accept="better", budget=30000, seed=5)
    always = ILS.run(D, start, cand=cand, local_search="dlb", accept="always", budget=30000, seed=5)
    assert always.final_length >= better.final_length - 1e-6          # "always" wandert weiter, "better" bleibt am besten Punkt
    assert always.accepted == always.iterations                       # bei "always" wird jede Iteration angenommen


def test_deterministic_for_a_seed_and_different_for_another():
    D = _instance(25, 6)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(25, np.random.default_rng(5))
    a = ILS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=11)
    b = ILS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=11)
    c = ILS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=20)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.best_tour, c.best_tour) or a.evaluations != c.evaluations


def test_dlb_and_full_rescan_reach_comparable_quality_on_a_small_instance():
    """Kleine Instanz, großes Budget: beide Varianten der lokalen Suche sollten nahe beieinander landen - der Unterschied
    ist der Bewertungsaufwand je Wiederabstieg (siehe test_claims.py), nicht die erreichbare Güte bei genug Budget."""
    D = _instance(15, 7)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(15, np.random.default_rng(6))
    dlb = ILS.run(D, start, cand=cand, local_search="dlb", budget=40000, seed=1)
    full = ILS.run(D, start, local_search="full", budget=40000, seed=1)
    assert abs(dlb.best_length - full.best_length) / full.best_length < 0.05


def test_unknown_local_search_and_accept_rule_are_rejected():
    D = _instance(10, 0)
    start = T.random_tour(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        ILS.run(D, start, cand=DLB.build_candidate_lists(D), local_search="nonsense", budget=1000)
    with pytest.raises(ValueError):
        ILS.run(D, start, cand=DLB.build_candidate_lists(D), accept="nonsense", budget=1000)
    with pytest.raises(ValueError):
        ILS.run(D, start, local_search="dlb", cand=None, budget=1000)    # dlb braucht Kandidatenlisten


def test_snapshots_are_only_kept_when_requested():
    D = _instance(12, 0)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(12, np.random.default_rng(0))
    with_snaps = ILS.run(D, start, cand=cand, budget=3000, seed=0, keep_snapshots=True)
    without = ILS.run(D, start, cand=cand, budget=3000, seed=0, keep_snapshots=False)
    assert len(with_snaps.snapshots) == with_snaps.iterations + 1       # Start + eine je Iteration
    assert without.snapshots == []


def test_stronger_kicks_touch_more_nodes_and_use_more_evaluations_per_iteration():
    D = _instance(40, 8)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(40, np.random.default_rng(7))
    weak = ILS.run(D, start, cand=cand, local_search="dlb", n_bridges=1, budget=20000, seed=3)
    strong = ILS.run(D, start, cand=cand, local_search="dlb", n_bridges=5, budget=20000, seed=3)
    assert strong.iterations < weak.iterations                          # jede Iteration kostet bei staerkerer Stoerung mehr Bewertungen
