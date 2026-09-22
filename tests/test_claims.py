"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt.
Die Kette ist bis auf die Erzeugung der Zufallszahlen deterministisch; die Bänder sind weit genug für Unterschiede zwischen numpy-Versionen und
Plattformen. Positive UND negative Aussagen: wo Iterated Local Search nicht besser ist als Hill Climbing mit Neustarts, steht das hier ebenso
als Test wie dort, wo es gewinnt. Rechenzeiten sind nur als Größenordnung geprüft."""

from functools import lru_cache

import numpy as np
import pytest

import ils_constants as C
import ils_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 0.64, 0.5)
    near(std["dlbr"], 0.68, 0.5)
    near(std["hcr"], 4.88, 1.2)
    assert std["gap"] < std["dlbr"] + 0.3                              # ILS mindestens gleichauf mit der fairen Vergleichsgröße


# --- Budget-Sweep ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,ils,dlbr,tol", [
    (10000, 1.36, 2.35, 0.8), (25000, 1.03, 1.37, 0.6), (50000, 0.79, 0.97, 0.5), (100000, 0.70, 0.78, 0.4),
    (200000, 0.64, 0.68, 0.4), (500000, 0.60, 0.61, 0.35), (1000000, 0.58, 0.59, 0.3), (2000000, 0.58, 0.58, 0.3),
])
def test_budget_sweep_numbers(budget, ils, dlbr, tol):
    row = cfg(budget=budget)
    near(row["gap"], ils, tol)
    near(row["dlbr"], dlbr, tol)


def test_the_advantage_over_dlb_restarts_shrinks_with_the_budget():
    small = cfg(budget=10000)
    large = cfg(budget=2000000)
    small_edge = small["dlbr"] - small["gap"]
    large_edge = large["dlbr"] - large["gap"]
    assert small_edge > large_edge + 0.3                               # der Vorsprung schrumpft deutlich, verschwindet aber nicht ins Negative
    assert large_edge > -0.3                                           # bei großem Budget praktisch gleichauf, ILS nicht schlechter


def test_full_rescan_restarts_need_about_seventy_four_thousand_evaluations_for_one_descent_at_sixty_stops():
    # unter diesem Budget kann kein zweiter Neustart mehr passen: identischer Wert bei 10/25/50/100 Tausend
    v10, v25, v50, v100 = (cfg(budget=b)["hcr"] for b in (10000, 25000, 50000, 100000))
    assert v10 == pytest.approx(v25, abs=0.05) == pytest.approx(v50, abs=0.05) == pytest.approx(v100, abs=0.05)
    near(v100, 7.88, 0.6)
    assert cfg(budget=500000)["hcr"] < v100 - 3.0                      # ab 500 Tausend passt ein zweiter Neustart


# --- Störstärke -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_bridges,gap,tol", [(1, 1.03, 0.6), (2, 0.85, 0.5), (3, 0.78, 0.5), (5, 0.87, 0.5), (8, 1.11, 0.6)])
def test_n_bridges_sweep_numbers_at_twenty_five_thousand(n_bridges, gap, tol):
    near(cfg(budget=25000, n_bridges=n_bridges)["gap"], gap, tol)


def test_three_bridges_beats_both_one_and_eight_at_twenty_five_thousand():
    one = cfg(budget=25000, n_bridges=1)["gap"]
    three = cfg(budget=25000, n_bridges=3)["gap"]
    eight = cfg(budget=25000, n_bridges=8)["gap"]
    assert three < one and three < eight                               # ein Sweet Spot, keine monotone Kurve


# --- Annahme --------------------------------------------------------------------------------------------------------------------------------


def test_accept_rule_numbers_at_two_hundred_thousand():
    better = cfg(accept="better")
    always = cfg(accept="always")
    near(better["gap"], 0.64, 0.4)
    near(better["final"], 0.64, 0.4)
    near(always["gap"], 0.67, 0.4)
    near(always["final"], 14.18, 3.0)
    assert always["final"] > always["gap"] + 8.0                       # Random Walk: letzte Tour weit schlechter als die beste
    assert abs(better["final"] - better["gap"]) < 0.1                  # "nur besser": beste und letzte Tour praktisch identisch


# --- Lokale Suche -----------------------------------------------------------------------------------------------------------------------------


def test_local_search_numbers_at_two_hundred_thousand():
    dlb = cfg(local_search="dlb")
    full = cfg(local_search="full")
    near(dlb["gap"], 0.64, 0.4)
    near(full["gap"], 3.58, 1.2)
    near(dlb["iterations"], 2744, 500)
    near(full["iterations"], 12, 6)
    assert dlb["iterations"] > 50 * full["iterations"]                 # Kandidatenliste + DLB erlaubt weit mehr Iterationen


# --- Große Instanz ----------------------------------------------------------------------------------------------------------------------------


def test_large_instance_at_two_hundred_stops_one_million_budget():
    row = ev.run_config(ev.Settings(n=200), budget=1000000)
    near(row["gap"], 1.9, 1.0)
    assert row["gap"] < row["dlbr"] - 0.5                              # der Vorsprung ist bei größeren Instanzen nicht kleiner


# --- Startlösung ------------------------------------------------------------------------------------------------------------------------------


def test_start_solution_barely_matters_for_ils_unlike_a_single_descent():
    random_ = cfg(start="random")
    nearest = cfg(start="nearest")
    near(random_["gap"], nearest["gap"], 0.3)
    assert random_["hc"] > nearest["hc"] + 0.3                         # ein einzelner Abstieg spürt die Startlösung noch deutlich


# --- Kandidatenliste + Don't-Look-Bits: Korrektheitszahlen (aus dem Kern der Hill-Climbing-Demo übernommen) ---------------------------------


def test_temperature_unit_free_instance_generation_matches_the_hill_climbing_demo():
    inst, D = ev.instance(60, 0, 100000)
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)
