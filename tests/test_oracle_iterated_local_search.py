"""Unabhängiges Orakel für Iterated Local Search: (1) die Doppelbrücke wird aus Eingabe und Ausgabe zurückgerechnet (genau ein Schnittpunkt-Tripel, Stücke S1 S3 S2 S4,
Mindestlänge je Stück, genau drei Kanten ändern sich, `touched` = Endpunkte der neuen Kanten); (2) die ILS-Schleife wird mit festem Zufallsstrom nachsimuliert
(Zähler, Tour nach jeder Iteration, Verlaufskurven, Annahmeregeln); (3) Nachbarschaftsdelta jedes 2-opt-Zuges gegen die volle Neubewertung."""

import numpy as np
import pytest

import ils_algorithm as ILS
import ils_dlb as DLB
import ils_kick as K
import ils_tour as T


def _edges(t):
    return {frozenset((t[k], t[(k + 1) % len(t)])) for k in range(len(t))}


def test_double_bridge_is_decoded_back_to_exactly_one_cut_triple():
    rng = np.random.default_rng(1)
    for it in range(60):
        n = int(rng.integers(10, 30))
        t = [0] + rng.permutation(np.arange(1, n)).tolist()
        out, touched = K.double_bridge(np.array(t), np.random.default_rng(it))
        out = out.tolist()
        cuts = [(p1, p2, p3) for p1 in range(1, n) for p2 in range(p1 + 1, n) for p3 in range(p2 + 1, n)
                if t[:p1] + t[p2:p3] + t[p1:p2] + t[p3:] == out]
        assert len(cuts) == 1
        p1, p2, p3 = cuts[0]
        assert min(p1, p2 - p1, p3 - p2, n - p3) >= K.MIN_SEGMENT and out[0] == 0
        assert len(_edges(t) - _edges(out)) == 3
        assert {v for e in _edges(out) - _edges(t) for v in e} == set(touched)


def _reference_run(D, start, cand, local_search, n_bridges, accept, budget, seed, trace_points=300):
    rng = np.random.default_rng(seed)
    init_seed = int(rng.integers(0, 2**31 - 1))
    if local_search == "dlb":
        r0 = DLB.dlb_descend(D, start, cand, seed=init_seed, max_evaluations=budget)
    else:
        r0 = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=budget)
    cur, cur_len, ev = r0.tour.copy(), r0.length, r0.evaluations
    best, best_len, snaps, trace = cur.copy(), cur_len, [cur.copy()], [(ev, cur_len, cur_len)]
    its = acc = 0
    every = max(1, budget // trace_points)
    next_trace = (ev // every + 1) * every                                    # Verlaufspunkte alle `every` BEWERTETEN Nachbarn
    while ev < budget:
        kicked, touched = K.double_bridge(cur, rng, n_bridges=n_bridges)
        ds = int(rng.integers(0, 2**31 - 1))
        if local_search == "dlb":
            r = DLB.dlb_descend(D, kicked, cand, seed=ds, max_evaluations=budget - ev, touched=touched)
        else:
            r = T.descend(D, kicked, "2opt", "first", keep_steps=False, max_evaluations=budget - ev)
        ev += r.evaluations
        its += 1
        if accept == "always" or r.length <= cur_len + 1e-9:
            cur, cur_len = r.tour.copy(), r.length
            acc += 1
        if cur_len < best_len - 1e-9:
            best, best_len = cur.copy(), cur_len
        snaps.append(cur.copy())
        if ev >= next_trace or ev >= budget:
            trace.append((ev, cur_len, best_len))
            next_trace = (ev // every + 1) * every
    return best, cur, ev, its, acc, snaps, np.array(trace)


@pytest.mark.parametrize("local_search", ["dlb", "full"])
@pytest.mark.parametrize("accept", ["better", "always"])
def test_run_matches_a_resimulation_with_the_same_random_stream(local_search, accept):
    rng = np.random.default_rng(11)
    for it in range(6):
        n = int(rng.integers(10, 24))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        cand = DLB.build_candidate_lists(D, 5)
        budget = 4000 if local_search == "dlb" else 20000
        start = T.random_tour(n, np.random.default_rng(it))
        nb = 1 + it % 3
        got = ILS.run(D, start, cand=cand, local_search=local_search, n_bridges=nb, accept=accept, budget=budget, seed=it)
        best, cur, ev, its, acc, snaps, trace = _reference_run(D, start, cand, local_search, nb, accept, budget, it)
        assert (got.evaluations, got.iterations, got.accepted) == (ev, its, acc)
        assert np.array_equal(got.best_tour, best) and np.array_equal(got.final_tour, cur)
        assert len(got.snapshots) == len(snaps) and all(np.array_equal(a, b) for a, b in zip(got.snapshots, snaps))
        assert np.array_equal(got.trace_iter, trace[:, 0]) and np.allclose(got.trace_length, trace[:, 1]) and np.allclose(got.trace_best, trace[:, 2])
        assert got.best_length == pytest.approx(min(T.tour_length(s, D) for s in got.snapshots), abs=1e-7)
        assert got.trace_iter[-1] == got.evaluations and np.all(np.diff(got.trace_iter) > 0)
        if got.iterations > 100:
            assert len(got.trace_iter) > 40                                   # die Kurve hat viele Punkte (früher nur wenige: Iterationen statt Bewertungen gezählt)
        if local_search == "dlb":
            assert got.evaluations <= budget                                  # die Kandidatenlisten-Suche hält das Budget exakt ein


def test_every_two_opt_delta_equals_the_recomputed_length_change():
    rng = np.random.default_rng(2)
    for _ in range(25):
        n = int(rng.integers(5, 14))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        t = T.random_tour(n, rng)
        delta, ok = T._delta_2opt(t, D)
        for i, j in zip(*np.where(ok)):
            new = t.tolist()[:i + 1] + t.tolist()[i + 1:j + 1][::-1] + t.tolist()[j + 1:]
            assert T.tour_length(np.array(new), D) - T.tour_length(t, D) == pytest.approx(delta[i, j], abs=1e-9)
