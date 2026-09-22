"""Iterated Local Search - eine Lieferrunde, die ein gutes Optimum stört statt es wegzuwerfen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: dieselbe Suche wie die Wurzel (Hill Climbing) und die
Simulated-Annealing-Demo, auf derselben Lieferrunde - aber statt zu akzeptieren, dass die Tour verlängert wird (Simulated Annealing),
oder komplett neu zu starten (Hill Climbing mit Neustarts), wird ein bereits gutes lokales Optimum gezielt gestört (Doppelbrücken-Zug)
und neu abgestiegen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import ils_constants as C
import ils_dlb as DLB
import ils_tour as T
from ils_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, sweep, verdict
from ils_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from ils_visualization import build_budget, build_instance, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Iterated Local Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🌉 Iterated Local Search – eine Lieferrunde, die ein gutes Optimum stört statt es wegzuwerfen")
st.markdown(
    """
**Hill Climbing mit Neustarts** wirft eine fertige Tour weg und fängt komplett neu an, sobald kein Zug mehr verbessert. **Iterated Local Search** (ILS) tut das Gegenteil: es **stört** eine bereits gute Tour gezielt (ein **Doppelbrücken-Zug** – eine Umordnung, die kein einzelner 2-opt-Zug rückgängig machen kann) und
steigt von dort **neu ab**. Der Großteil der Tour bleibt dabei erhalten; nur die Umgebung der Störung muss neu optimiert werden – das macht jede Iteration billig, wenn die lokale Suche das ausnutzt (Kandidatenliste + Don't-Look-Bits, nicht der volle Rescan).
Angenommen wird die neue Tour nur, wenn sie mindestens so kurz ist wie die vorige; sonst geht es von der alten aus weiter. Lohnt sich Wiederverwenden gegenüber Wegwerfen – bei gleichem Bewertungsbudget? Das misst diese Demo, gegen dieselbe untere Schranke wie die Geschwister-Demos.
"""
)
st.caption(
    "Drittes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe: dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) und der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/) - ein Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen. "
    "Die Linie hat keinen Konvergenzpunkt: VNS und ALNS bauen später auf Iterated Local Search auf (die Störstärke wird dort systematisch statt fest), Tabu Search und GRASP sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

with st.expander("So funktioniert Iterated Local Search", expanded=True):
    st.markdown(
        """
1. **Erster Abstieg.** Aus der Startlösung wird ganz normal lokal optimiert (2-opt), bis kein Zug mehr verbessert – wie beim Hill Climbing.
2. **Doppelbrücken-Zug.** Drei zufällige Schnittpunkte teilen die Tour in vier Stücke; zwei mittlere Stücke tauschen den Platz. Das ändert mindestens drei Kanten gleichzeitig – anders als ein zufälliger 2-opt-"Kick" lässt sich das durch **keinen einzelnen** 2-opt-Zug rückgängig machen (geprüft: siehe Tests).
   Die **Störstärke** ist die Zahl der nacheinander ausgeführten Doppelbrücken.
3. **Wiederabstieg.** Nur die Umgebung der Störung wird neu durchsucht (Kandidatenliste + Don't-Look-Bits), nicht die ganze Tour – das macht jede Iteration billig. Zum Vergleich lässt sich auch der volle Rescan wählen (langsam, siehe Experiment).
4. **Annahme.** Ist die neue Tour mindestens so kurz wie die vorige, wird sie übernommen; sonst geht es von der vorigen aus weiter. Gemerkt wird immer die kürzeste je besuchte Tour.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke** (Held-Karp), wie in den Geschwister-Demos. Verglichen wird mit **Hill Climbing mit Neustarts** bei gleichem Bewertungsbudget – in zwei Varianten: voller Rescan (Kontinuität mit der Wurzel-Demo) und Kandidatenliste + DLB (die **faire** Vergleichsgröße, da ILS dieselbe lokale Suche nutzt).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Bei 200 Tausend Vorschlägen liegt die beste Tour bei 60 Stopps im Mittel 0.64 % über der Schranke; bei 200 Stopps und 1 Million Vorschlägen 1.90 %.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    local_search = st.selectbox(
        "Lokale Suche", list(C.LOCAL_SEARCH_LABELS), key="local_search_select", format_func=lambda k: C.LOCAL_SEARCH_LABELS[k],
        help="Wie der Wiederabstieg nach einer Störung sucht. Kandidatenliste + Don't-Look-Bits durchsucht nur die Umgebung der Störung (schnell: 0.64 % über der Schranke bei 200 Tausend Vorschlägen, rund 2744 Iterationen); "
             "der volle Rescan bewertet nach jedem Zug wieder alle Nachbarn (langsam: 3.58 % bei nur rund 12 Iterationen). Der Unterschied ist keine Nebensache - ohne Kandidatenliste lohnt sich Iterated Local Search kaum.",
    )
    n_bridges = st.slider(
        "Störstärke (Doppelbrücken je Kick)", C.N_BRIDGES_MIN, C.N_BRIDGES_MAX, key="n_bridges_slider",
        help="Wie viele Doppelbrücken-Züge nacheinander je Kick ausgeführt werden. Bei 25 Tausend Vorschlägen liegt die beste Tour bei 1 Doppelbrücke 1.03 %, bei 2: 0.85 %, bei 3: 0.78 % (bestes gemessen), bei 5: 0.87 %, bei 8: 1.11 % über der Schranke - "
             "ein Sweet Spot bei 2-3, nicht beim literaturüblichen Standardwert 1 und nicht bei starken Störungen (jede Iteration kostet dann mehr, es passen weniger ins Budget).",
    )
    accept = st.selectbox(
        "Annahme", list(C.ACCEPT_LABELS), key="accept_select", format_func=lambda k: C.ACCEPT_LABELS[k],
        help="Nur besser: die neue Tour wird nur übernommen, wenn sie mindestens so kurz ist (klassisches ILS). Immer: jede neue Tour wird übernommen, auch schlechtere (Random Walk über lokale Optima) - "
             "bei 200 Tausend Vorschlägen liegt die BESTE gefundene Tour kaum schlechter (0.67 % gegen 0.64 %), aber die LETZTE Tour der Kette bei 14.18 % statt 0.64 %: ohne die Annahmeregel verlässt die Suche die guten Touren wieder.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarschaften)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Wie viele Nachbarschaften insgesamt bewertet werden (bei Kandidatenliste + DLB ein geprüftes Kandidatenpaar, beim vollen Rescan ein bewerteter Nachbar). Bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen liegt die beste Tour "
             "1.36 / 1.03 / 0.79 / 0.70 / 0.64 / 0.60 / 0.58 / 0.58 % über der Schranke; Hill Climbing mit Neustarts (Kandidatenliste + DLB, faire Vergleichsgröße) 2.35 / 1.37 / 0.97 / 0.78 / 0.68 / 0.61 / 0.59 / 0.58 %. "
             "Der Vorsprung von Iterated Local Search ist bei knappem Budget am größten und schrumpft mit wachsendem Budget fast auf null.",
    )
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällige Reihenfolge oder Nächster Nachbar. Der erste Abstieg und die vielen folgenden Kicks vergessen die Startlösung schnell: kein messbarer Unterschied bei der besten Tour.",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufällige Startlösung, die Doppelbrücken-Schnittpunkte und die Reihenfolge der Wiederabstiege.",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "local_search_select": local_search,
    "n_bridges_slider": int(n_bridges), "accept_select": accept, "budget_select": int(budget), "start_radio": start, "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), local_search, int(n_bridges), accept, int(budget), start, int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- Iterated Local Search in Aktion ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Iterated Local Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Erster Abstieg", 3: "3 · Kicks", 4: "4 · Ergebnis"}
if "ils_step" not in st.session_state or st.session_state.get("ils_step_owner") != data_key:
    st.session_state["ils_step"] = 1
    st.session_state["ils_step_owner"] = data_key
    st.session_state.pop("ils_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ils_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
iteration = n_snaps - 1
play_kicks = False
if step == 3 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="ils_iter", help="Die Tour nach dieser Iteration (0 = nach dem ersten Abstieg, vor dem ersten Kick).")
    with itplay_col:
        play_kicks = st.button("▶️ Kicks abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**Tour nach dem ersten Abstieg** – {a.start_gap:.1f} % über der Schranke vor, {run.trace_length[0]:.0f} km ({100 * (run.trace_length[0] - a.bound) / a.bound:.1f} % über der Schranke) nach dem Abstieg")
            c1.plotly_chart(build_tour(xy, run.snapshots[0]), width="stretch", key="s2_map")
            c2.markdown("**Vor den Kicks**")
            c2.metric("Startlösung", f"{a.start_gap:.1f} %", help="Abstand zur Schranke der Startlösung, vor jeder Optimierung.")
            c2.metric("Nach dem ersten Abstieg", f"{100 * (run.trace_length[0] - a.bound) / a.bound:.1f} %", help="Abstand zur Schranke, bevor die erste Doppelbrücke angewendet wird.")
        elif current_step == 3:
            st.markdown("**Länge der aktuellen und der besten Tour über die bewerteten Nachbarschaften**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.hc.length, T.tour_length(a.hcr_tour, a.D), T.tour_length(a.dlbr_tour, a.D)), width="stretch", key=f"s3_trace_{it}")
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s3_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Iterated Local Search: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s4_ils")
            c2.markdown(f"**Hill Climbing mit Neustarts (Kandidatenliste + DLB)** ({a.dlbr_starts} Abstiege, gleiches Budget, fairer Vergleich) – {a.dlbr_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.dlbr_tour), width="stretch", key="s4_hc")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for f in _frames():
                _render(3, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 4
elif play_kicks:
    for f in _frames():
        _render(3, f)
        time.sleep(0.1)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption("Der erste Abstieg optimiert lokal (2-opt), bis kein Zug mehr verbessert - wie ein einzelner Hill-Climbing-Abstieg. Ab hier beginnen die Kicks.")
elif step == 3:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarschaften in {run.iterations} Iterationen; angenommen wurden {a.accept_rate:.0%} davon. Die aktuelle Tour (blau) folgt der besten (rot), solange 'nur besser' angenommen wird.")
else:
    st.caption(f"Links die beste Tour aus {run.iterations} Iterationen, rechts die beste Tour aus {a.dlbr_starts} Neustarts mit Kandidatenliste + DLB bei gleichem Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Kette gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung: die Ketten streuen (siehe Streuung unten), Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("Ein Hill-Climbing-Abstieg", f"{a.hc_gap:.1f} %", delta=f"{_fmt_int(a.hc.evaluations)} Bewertungen", delta_color="off", help="Ein Abstieg (voller Rescan) aus derselben Startlösung.")
m3.metric("HC-Neustarts (Kandidatenliste + DLB)", f"{a.dlbr_gap:.1f} %", delta=f"{a.dlbr_starts} Abstiege, fairer Vergleich", delta_color="off", help="So viele Abstiege mit derselben lokalen Suche, wie ins Budget passen; die beste Tour zählt.")
m4.metric("Angenommene Iterationen", f"{a.accept_rate:.0%}", delta=f"{run.accepted} von {run.iterations}", delta_color="off", help="Anteil der Kicks, deren Wiederabstieg mindestens so kurz wie die vorige Tour war.")

if code == "beats_hc":
    st.success(f"✅ Besser als Hill Climbing mit Neustarts (Kandidatenliste + DLB, fairer Vergleich bei gleichem Budget): {a.gap:.1f} % über der Schranke gegen {a.dlbr_gap:.1f} % ({a.dlbr_starts} Abstiege). Die Wiederverwendung eines guten Optimums schlägt den Neustart. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Iterated Local Search {a.gap:.1f} %, Hill Climbing mit Neustarts (Kandidatenliste + DLB) {a.dlbr_gap:.1f} % über der Schranke. Bei großem Budget holen die Neustarts auf; eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Hill Climbing mit Neustarts (Kandidatenliste + DLB) ist besser: {a.dlbr_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget ({a.dlbr_starts} Abstiege). Beim vollen Rescan als lokale Suche lohnt sich die Wiederverwendung nicht - und eine einzelne Kette kann Pech haben.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarschaften", "Rechenzeit"],
              "Beste Tour": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "Letzte Tour": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "Ein Abstieg": [f"{a.hc.length:.1f}", f"{a.hc_gap:.2f} %", _fmt_int(a.hc.evaluations), unit_time(a.hc_seconds)],
              "HC-Neustarts (voller Rescan)": [f"{T.tour_length(a.hcr_tour, a.D):.1f}", f"{a.hcr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.hcr_seconds)],
              "HC-Neustarts (Kandidatenliste + DLB)": [f"{T.tour_length(a.dlbr_tour, a.D):.1f}", f"{a.dlbr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.dlbr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Lokale Suche", "Störstärke", "Annahme", "Budget", "Iterationen", "Startlösung", "Kreuzungen der besten Tour"],
              "Einstellung": [C.LOCAL_SEARCH_LABELS[settings.local_search], f"{settings.n_bridges}", C.ACCEPT_LABELS[settings.accept], _fmt_int(settings.budget),
                              f"{run.iterations}", C.START_LABELS[settings.start], f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist eine bewertete Nachbarschaft - bei Kandidatenliste + DLB ein geprüftes Kandidatenpaar, beim vollen Rescan ein bewerteter Nachbar, dieselbe Einheit wie in den Geschwister-Demos. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, Störstärke, Suche und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param in ("accept", "local_search", "start")
    labels = {"accept": C.ACCEPT_LABELS, "local_search": C.LOCAL_SEARCH_LABELS, "start": C.START_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Die Streuung ist die Standardabweichung der Läufe, nicht des Mittels. Gestrichelt: ein Hill-Climbing-Abstieg; gepunktet: Hill Climbing mit Neustarts (Kandidatenliste + DLB, fairer Vergleich).")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: wann lohnt sich die Wiederverwendung?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 60 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "ILS (%)": [f"{r['gap']:.2f}" for r in rows_b],
              "HC-Neustarts, Kandidatenliste+DLB (%)": [f"{r['dlbr']:.2f}" for r in rows_b], "HC-Neustarts, voller Rescan (%)": [f"{r['hcr']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, 1 Doppelbrücke, Kandidatenliste + DLB). Bei 10 Tausend Vorschlägen: ILS **1.36 %**, Neustarts mit derselben lokalen Suche **2.35 %** - der Vorsprung der Wiederverwendung ist hier am größten. "
               "Er schrumpft mit wachsendem Budget (200 Tausend: 0.64 gegen 0.68 %) und verschwindet ab etwa 500 Tausend fast ganz (0.60 gegen 0.61 %, 2 Millionen: 0.58 gegen 0.58 %) - ab genug Neustarts holt der Neustart die Wiederverwendung ein, weil beide dieselbe billige lokale Suche nutzen. "
               "Gegen den vollen Rescan (die bewusst einfache Variante der Wurzel-Demo) gewinnt ILS bei jedem Budget deutlich - das ist aber, wie schon bei Simulated Annealing gelernt, vor allem ein Beleg für Kandidatenliste + DLB, nicht spezifisch für Iterated Local Search (siehe das Experiment unten).")

st.markdown("---")

st.subheader("🔬 Störstärke: zu schwach, zu stark, oder ein Sweet Spot?")
if st.button("Störstärke 1 bis 8 bei knappem Budget durchfahren (dauert etwa 20 Sekunden)", key="bridges_start"):
    st.session_state["bridges_on"] = True
if st.session_state.get("bridges_on"):
    with st.spinner("Rechne 5 Störstärken × 5 Instanzen × 3 Ketten..."):
        rows_nb = _sweep("n_bridges", replace(base_sweep, budget=25000))
    st.plotly_chart(build_sweep(rows_nb, SWEEP_LABELS["n_bridges"]), width="stretch", key="bridges_chart")
    st.table({"Doppelbrücken je Kick": [f"{r['value']}" for r in rows_nb], "ILS (%)": [f"{r['gap']:.2f}" for r in rows_nb], "Iterationen": [f"{r['iterations']:.0f}" for r in rows_nb]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten, 25 Tausend Vorschläge (60 Stopps, Kandidatenliste + DLB): 1 Doppelbrücke **1.03 %**, 2: 0.85 %, 3: **0.78 %** (bestes gemessen), 5: 0.87 %, 8: **1.11 %** über der Schranke. "
               "Ein Sweet Spot bei 2-3, nicht bei 1 (dem literaturüblichen Standardwert) - schwache Störungen machen kaum Fortschritt je Iteration, starke kosten mehr Bewertungen je Iteration (weniger Iterationen passen ins Budget), ohne die Güte je Iteration entsprechend zu verbessern.")

st.markdown("---")

st.subheader("🔬 Annahme: warum nicht einfach alles annehmen?")
if st.button("'Nur besser' gegen 'immer annehmen' vergleichen (dauert etwa 10 Sekunden)", key="accept_start"):
    st.session_state["accept_on"] = True
if st.session_state.get("accept_on"):
    with st.spinner("Rechne beide Annahmeregeln × 5 Instanzen × 3 Ketten..."):
        rows_acc = _sweep("accept", base_sweep)
    st.table({"Annahme": [C.ACCEPT_LABELS[r["value"]] for r in rows_acc], "Beste Tour (%)": [f"{r['gap']:.2f}" for r in rows_acc], "Letzte Tour (%)": [f"{r['final']:.2f}" for r in rows_acc]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (200 Tausend Vorschläge). 'Nur besser': beste **0.64 %**, letzte **0.64 %** - identisch, die Kette verlässt ein einmal erreichtes Optimum nie wieder. "
               "'Immer annehmen' (Random Walk über lokale Optima): beste **0.67 %** (kaum schlechter - die beste Tour wird weiterhin gemerkt), letzte **14.18 %** (!) - ohne die Annahmeregel wandert die Kette weit von jeder guten Tour weg. "
               "Die Annahmeregel kostet nichts bei der besten Tour, entscheidet aber vollständig darüber, wo die Kette am Ende steht.")

st.markdown("---")

st.subheader("🔬 Lokale Suche: warum Kandidatenliste + Don't-Look-Bits?")
if st.button("Kandidatenliste + DLB gegen vollen Rescan vergleichen (dauert etwa 15 Sekunden)", key="ls_start"):
    st.session_state["ls_on"] = True
if st.session_state.get("ls_on"):
    with st.spinner("Rechne beide lokalen Suchen × 5 Instanzen × 3 Ketten..."):
        rows_ls = _sweep("local_search", base_sweep)
    st.table({"Lokale Suche": [C.LOCAL_SEARCH_LABELS[r["value"]] for r in rows_ls], "Beste Tour (%)": [f"{r['gap']:.2f}" for r in rows_ls], "Iterationen": [f"{r['iterations']:.0f}" for r in rows_ls]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (200 Tausend Vorschläge, 1 Doppelbrücke). Kandidatenliste + DLB: **0.64 %** bei rund **2744** Iterationen; voller Rescan: **3.58 %** bei nur rund **12** Iterationen - derselbe Faktor wie beim Hill-Climbing-Neustart-Vergleich. "
               "Iterated Local Search lebt davon, dass ein Wiederabstieg billig ist; ohne Kandidatenliste ist er das nicht, und die Wiederverwendung bringt kaum noch etwas gegenüber einem Neustart.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 10 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten und 20 Abstiege..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["ils"], sp["hc"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("Iterated Local Search: Mittel ± Streuung", f"{sp['ils'].mean():.2f} ± {sp['ils'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Ein Hill-Climbing-Abstieg: Mittel ± Streuung", f"{sp['hc'].mean():.2f} ± {sp['hc'].std():.2f} %", help="Ein Abstieg je Kette aus derselben zufälligen Startlösung.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (die Startlösung wechselt mit).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 60 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget); zum Vergleich Hill Climbing mit Neustarts (voller Rescan, wie in der Wurzel-Demo).")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die lokale Suche ist billig** | Ohne Kandidatenliste + DLB (voller Rescan): **3.58 %** über der Schranke bei nur rund 12 Iterationen statt 0.64 % bei rund 2744 (200 Tausend Vorschläge) - die Wiederverwendung bringt fast nichts mehr. | Kein Nachfolger nötig - die Lehre gilt für jede iterative Metaheuristik, die viele billige Wiederabstiege braucht (**VNS**, **ALNS**, **Tabu Search**) |
| **Die Störstärke passt** | Zu schwach (1 Doppelbrücke, 25 Tausend Vorschläge): 1.03 %. Zu stark (8): **1.11 %**. Am besten: 2-3 Doppelbrücken (**0.78 %**) - ein Sweet Spot, keine monotone Kurve. | **VNS** (Störstärke steigt systematisch statt fest gewählt zu werden) |
| **Die Annahmeregel filtert** | "Immer annehmen": beste Tour kaum schlechter (0.67 % gegen 0.64 %), aber die letzte Tour der Kette **14.18 %** statt 0.64 % - ein Random Walk verlässt gute Touren wieder. | **Simulated Annealing** (probabilistische statt harte Annahme), **Tabu Search** (Gedächtnis gegen Rückwege) |
| **Das Budget reicht für viele Iterationen** | Bei 10 Tausend Vorschlägen (rund 125 Iterationen) ist der Vorsprung vor Neustarts am größten (1.36 gegen 2.35 %); bei 2 Millionen (viele Zehntausend Iterationen) verschwindet er fast (0.58 gegen 0.58 %). | **ALNS** (lernt, welcher Umbau sich lohnt, statt blind zu kicken) |
| **Die Störung ist strukturell, nicht zufällig** | Ein zufälliger 2-opt-"Kick" würde der nachfolgende 2-opt-Abstieg oft sofort rückgängig machen; die Doppelbrücke lässt sich durch keinen einzelnen 2-opt-Zug rückgängig machen (geprüft). | **Lin-Kernighan** (chained LK = ILS mit LK als innerer Suche statt 2-opt) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie (noch nicht gebaut): VNS (systematisch wachsende Störstärke) und ALNS (lernt, welcher Umbau sich lohnt) bauen direkt auf Iterated Local Search auf; Tabu Search, GRASP und der Nachbarschafts-Zweig "
    "(Lin-Kernighan, VLSN, VRP-Nachbarschaften) sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Doppelbrücken-Zug.** Drei Schnittpunkte $1 \le p_1 < p_2 < p_3 \le N-1$ teilen $\pi$ in vier Stücke $S_1, S_2, S_3, S_4$ (zyklisch, $S_1$ vorn). Die Störung setzt sie als $S_1, S_3, S_2, S_4$ neu zusammen. Mindestlänge 2 je Stück (bei ausreichend großem $N$) verhindert, dass der Zug zu einem einzelnen 2-opt-Zug entartet
(beide mittleren Stücke der Länge 1 wären ein einfacher Knotentausch).

**Iteration.** $\pi' = \text{Kick}(\pi)$, dann $\pi'' = \text{LocalSearch}(\pi')$ (2-opt bis zum lokalen Optimum, mit Kandidatenliste + Don't-Look-Bits nur um die vom Kick betroffenen Knoten). Angenommen: $\pi \leftarrow \pi''$ falls $L(\pi'') \le L(\pi)$, sonst bleibt $\pi$ unverändert. Gemerkt wird $\arg\min L$ über alle je besuchten $\pi''$.

**Kandidatenliste + Don't-Look-Bits.** Wie in der Hill-Climbing-Demo: für jeden Knoten die $k=5$ nächsten Nachbarn, sortiert; die Suche nach einem verbessernden 2-opt-Zug bricht ab, sobald der nächste Kandidat weiter entfernt ist als die zu entfernende Kante (Bentley 1992; Johnson & McGeoch). Nach einem Kick startet die Warteschlange nur mit den
Endpunkten der neuen Kanten, nicht mit allen $N$ Knoten - das macht den Wiederabstieg billig.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. **Hill Climbing mit Neustarts**, zwei Varianten bei gleichem Budget: voller Rescan (Kontinuität mit der Wurzel-Demo) und Kandidatenliste + DLB (fairer Vergleich, da ILS dieselbe lokale Suche nutzt).

**Grenzen.** (1) Die lokale Suche muss billig sein, sonst verpufft der Vorteil der Wiederverwendung. (2) Die Störstärke ist ein Kompromiss (zu schwach: kein Fortschritt; zu stark: zu teuer je Iteration). (3) Die Annahmeregel entscheidet, wo die Kette am Ende steht, nicht nur, wie gut die beste je gefundene Tour ist. (4) Der Vorsprung vor Neustarts schrumpft mit dem Budget.

Implementiert in `ils_kick.py` (Doppelbrücke), `ils_dlb.py` (Kandidatenliste + Don't-Look-Bits, mit dem `touched`-Kurzweg), `ils_algorithm.py` (die ILS-Schleife), `ils_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der Hill-Climbing-Demo), `ils_scenario.py` (Instanzen), `ils_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
