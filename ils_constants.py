"""Konstanten der Iterated-Local-Search-Demo: Szenario (wortgleich zur hill-climbing-demo/simulated-annealing-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_START = "random"
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- ILS-eigene Regler ------------------------------------------------------------------------------------------
LOCAL_SEARCH_LABELS = {"dlb": "Kandidatenliste + Don't-Look-Bits", "full": "Voller Rescan"}
DEFAULT_LOCAL_SEARCH = "dlb"
N_BRIDGES_MIN, N_BRIDGES_MAX, DEFAULT_N_BRIDGES = 1, 8, 1
ACCEPT_LABELS = {"better": "Nur besser", "always": "Immer (Random Walk)"}
DEFAULT_ACCEPT = "better"
DEFAULT_BUDGET = 200000

# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), 60 gleichverteilte Stopps, Abstand zur Schranke (2026-09-22):
#   Budget        10T    25T    50T    100T   200T   500T   1M     2M
#   ILS (DLB)     1.36   1.03   0.79   0.70   0.64   0.60   0.58   0.58
#   HC-Neustarts, Kandidatenliste+DLB (fair)  2.35  1.37  0.97  0.78  0.68  0.61  0.59  0.58
#   HC-Neustarts, voller Rescan               7.88  7.88  7.88  7.88  4.88  2.93  2.54  1.87
# ILS zieht bei knappem Budget klar vor der fairen Vergleichsgröße (Neustarts + Kandidatenliste+DLB) vorbei - der Vorsprung
# schrumpft mit wachsendem Budget und verschwindet ab etwa 500 Tausend fast ganz (beide nutzen dieselbe billige lokale Suche,
# ab genug Neustarts holt der Neustart die Wiederverwendung ein). Gegen den vollen Rescan (die in der Wurzel-Demo gezeigte,
# bewusst einfache Variante) gewinnt ILS bei jedem Budget deutlich - aber das ist, wie schon bei Simulated Annealing gelernt,
# vor allem ein Beleg für den Wert von Kandidatenliste+DLB, nicht spezifisch für Iterated Local Search.
# Störstärke-Sweep (Budget 25 Tausend): 1 Doppelbrücke 1.03 %, 2: 0.85 %, 3: 0.78 % (bestes gemessenes), 5: 0.87 %, 8: 1.11 % -
# ein Sweet Spot bei 2-3, nicht bei 1 (dem literaturüblichen Standardwert) oder bei sehr starken Störungen.
# Annahme (Budget 200 Tausend): "nur besser" beste 0.64 %/letzte 0.64 %; "immer annehmen" beste 0.67 %/letzte 14.18 % -
# der Random Walk verlässt die guten Touren praktisch vollständig wieder, obwohl die BESTE gefundene Tour kaum schlechter ist.
# Lokale Suche (Budget 200 Tausend): Kandidatenliste+DLB 0.64 % bei 2744 Iterationen, voller Rescan 3.58 % bei nur 12 Iterationen.


def _preset(local_search=DEFAULT_LOCAL_SEARCH, n_bridges=DEFAULT_N_BRIDGES, accept=DEFAULT_ACCEPT, budget=DEFAULT_BUDGET, n=DEFAULT_N, start=DEFAULT_START):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "local_search": local_search, "n_bridges": n_bridges, "accept": accept,
            "budget": budget, "start": start, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleines Budget (10 Tausend)": _preset(budget=10000),
    "Voller Rescan (langsam)": _preset(local_search="full"),
    "Immer annehmen (Random Walk)": _preset(accept="always"),
    "Zu starke Störung (8 Doppelbrücken)": _preset(budget=10000, n_bridges=8),
    "Kalibrierte Störung (3 Doppelbrücken)": _preset(budget=25000, n_bridges=3),
    "Nächster Nachbar als Start": _preset(start="nearest"),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; Hill Climbing bei gleichem Bewertungsbudget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Kandidatenliste + DLB, 1 Doppelbrücke, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 0.64 % über der Schranke - Hill Climbing mit Neustarts über Kandidatenliste + DLB (dieselbe lokale Suche, fairer Vergleich) 0.68 %, mit vollem Rescan 4.88 %.",
    "Kleines Budget (10 Tausend)": "Nur 10 Tausend Vorschläge: 1.36 % über der Schranke gegen 2.35 % für Hill Climbing mit Neustarts (Kandidatenliste + DLB) - der Vorsprung der Wiederverwendung ist bei knappem Budget am größten (ein voller Rescan-Abstieg braucht bei 60 Stopps allein schon rund 74 Tausend Bewertungen).",
    "Voller Rescan (langsam)": "Dieselbe Suche wie die Wurzel-Demo (Hill Climbing), ohne Kandidatenliste: 3.58 % über der Schranke bei nur rund 12 Iterationen (200 Tausend Vorschläge) - Kandidatenliste + DLB (0.64 %, 2744 Iterationen) ist keine Nebensache, sondern die Voraussetzung, damit sich Iterated Local Search überhaupt lohnt.",
    "Immer annehmen (Random Walk)": "Jede neue Tour wird angenommen, auch schlechtere: die beste gefundene Tour bleibt fast gleich gut (0.67 % gegen 0.64 % bei 'nur besser'), aber die LETZTE Tour der Kette liegt bei 14.18 % über der Schranke - ohne die Annahmeregel verlässt die Suche die guten Touren wieder.",
    "Zu starke Störung (8 Doppelbrücken)": "8 Doppelbrücken je Kick bei knappem Budget (10 Tausend): 1.71 % über der Schranke gegen 1.36 % bei nur einer Doppelbrücke - jede Iteration kostet mehr, es passen weniger davon ins Budget.",
    "Kalibrierte Störung (3 Doppelbrücken)": "3 Doppelbrücken je Kick bei 25 Tausend Vorschlägen: 0.78 % über der Schranke - besser als 1 Doppelbrücke (1.03 %, der literaturübliche Standardwert) oder 8 (1.11 %): ein Sweet Spot, kein Rand.",
    "Nächster Nachbar als Start": "Die erste Doppelbrücke vergisst die Startlösung schnell: Nächster Nachbar 0.64 % gegen zufällig 0.64 % über der Schranke - kein messbarer Unterschied (anders als beim einzelnen Hill-Climbing-Abstieg: 5.4 % gegen 6.8 %).",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps, 1 Million Vorschläge: 1.90 % über der Schranke gegen 4.71 % für Hill Climbing mit Neustarts (Kandidatenliste + DLB) bei gleichem Budget - der Vorsprung wächst mit der Instanzgröße.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable"},
    "Kleines Budget (10 Tausend)": {"beats_hc", "comparable", "hc_wins"},
    "Voller Rescan (langsam)": {"hc_wins", "comparable"},
    "Immer annehmen (Random Walk)": {"beats_hc", "comparable"},
    "Zu starke Störung (8 Doppelbrücken)": {"beats_hc", "comparable"},
    "Kalibrierte Störung (3 Doppelbrücken)": {"beats_hc", "comparable"},
    "Nächster Nachbar als Start": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps, 1 Million)": {"beats_hc", "comparable"},
}
