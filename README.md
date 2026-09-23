# Iterated Local Search – eine Lieferrunde, die ein gutes Optimum stört statt es wegzuwerfen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-iterated-local-search-demo.streamlit.app/)**

Drittes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
dieselbe Rundtour wie in der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) und der [simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.
Anders als **Hill Climbing mit Neustarts** (eine fertige Tour wegwerfen, komplett neu anfangen) oder **Simulated Annealing** (Verschlechterungen mit sinkender Wahrscheinlichkeit annehmen) tut **Iterated Local Search** (ILS) etwas drittes: eine bereits gute Tour wird gezielt **gestört** (ein **Doppelbrücken-Zug**, den kein einzelner
2-opt-Zug rückgängig machen kann) und **neu abgestiegen** – der Großteil der Tour bleibt erhalten, nur die Umgebung der Störung wird neu optimiert.

**Einordnung in die Reihe (die Kanten des Graphen):** Iterated Local Search ist eine weitere Antwort auf die Schwäche der Wurzel (Hill Climbing bleibt im ersten lokalen Optimum stecken) und zugleich die **Grundlage** für zwei spätere Stücke.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum, statt neu zu starten)  [dieses Stück]
  │     └─ VNS → ALNS  (Störstärke wächst systematisch; lernt, welcher Umbau sich lohnt)   [nicht gebaut]
  │           ^ Querkante zu Lin-Kernighan (chained LK = ILS mit LK als innerer Suche)
  ├─ Tabu Search              (Gedächtnis gegen Rückwege)                        [nicht gebaut]
  └─ GRASP                    (randomisierte Konstruktion, viele Starts)         [nicht gebaut]
```

Ergebnis in Kürze: **Wiederverwenden schlägt Wegwerfen, aber nur mit der richtigen lokalen Suche und nur, solange das Budget knapp ist.** 60 Stopps, 200 Tausend Vorschläge (bewertete Nachbarschaften): die beste Tour liegt im Mittel **0.64 %** über der Schranke; Hill Climbing mit Neustarts und **derselben** lokalen Suche
(Kandidatenliste + Don't-Look-Bits, die faire Vergleichsgröße) **0.68 %** – bei 10 Tausend Vorschlägen ist der Vorsprung viel größer (**1.36 %** gegen **2.35 %**) und schrumpft mit wachsendem Budget fast auf null (2 Millionen: 0.58 gegen 0.58 %). Ohne Kandidatenliste (voller Rescan, wie die Wurzel-Demo) bringt die Wiederverwendung dagegen kaum
etwas (**3.58 %** bei nur rund 12 Iterationen). Und die Annahmeregel entscheidet fast alles darüber, wo die Kette am Ende steht: "immer annehmen" findet fast dieselbe beste Tour (0.67 % statt 0.64 %), aber die *letzte* Tour der Kette liegt bei **14.18 %** statt 0.64 %.

| Frage | Ergebnis (60 gleichverteilte Stopps, Kandidatenliste + DLB, 1 Doppelbrücke, "nur besser", 200 Tausend Vorschläge, zufällige Startlösung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke; Hill Climbing bei demselben Bewertungsbudget) |
|---|---|
| Standardfall | ✅ Beste Tour **0.64 %** über der Schranke gegen **0.68 %** für Hill Climbing mit Neustarts, Kandidatenliste + DLB (fairer Vergleich) und **4.88 %** für Neustarts mit vollem Rescan (wie die Wurzel-Demo) |
| **Budget** | ✅ Beste Tour bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen Vorschlägen **1.36 / 1.03 / 0.79 / 0.70 / 0.64 / 0.60 / 0.58 / 0.58 %**; Neustarts mit Kandidatenliste + DLB **2.35 / 1.37 / 0.97 / 0.78 / 0.68 / 0.61 / 0.59 / 0.58 %**. Der Vorsprung ist bei knappem Budget am größten und verschwindet ab etwa 500 Tausend fast ganz |
| **Lokale Suche** | ❌❗ Voller Rescan statt Kandidatenliste + DLB: **3.58 %** bei nur rund **12** Iterationen statt 0.64 % bei rund **2744** – die Wiederverwendung lohnt sich nur, wenn der Wiederabstieg billig ist |
| **Störstärke** | ⚠️ 1 / 2 / 3 / 5 / 8 Doppelbrücken je Kick (25 Tausend Vorschläge): **1.03 / 0.85 / 0.78 / 0.87 / 1.11 %** – ein Sweet Spot bei 2-3, nicht beim literaturüblichen Standardwert 1 und nicht bei starken Störungen |
| **Annahme** | ❗ "Nur besser": beste **0.64 %**, letzte **0.64 %** (identisch). "Immer annehmen" (Random Walk): beste **0.67 %** (kaum schlechter), letzte **14.18 %** – ohne die Annahmeregel verlässt die Kette gute Touren wieder |
| **Startlösung** | ➖ Zufällig 0.64 %, Nächster Nachbar 0.64 % über der Schranke – kein messbarer Unterschied (ein einzelner Hill-Climbing-Abstieg spürt die Startlösung dagegen deutlich: 6.8 gegen 5.4 %) |
| **Größe** | ✅ 200 Stopps, 1 Million Vorschläge: **1.90 %** gegen **4.71 %** für Hill Climbing mit Neustarts (Kandidatenliste + DLB) – der Vorsprung wächst mit der Instanzgröße, nicht umgekehrt |
| **Neustarts ohne Kandidatenliste (voller Rescan)** | ✅ Bei 10 / 25 / 50 / 100 Tausend Vorschlägen identisch **7.88 %** (ein Abstieg braucht bei 60 Stopps schon fast 74 Tausend Bewertungen, kein zweiter Neustart passt); erst ab 500 Tausend sinkt es (2.93 %) |

## Was die Demo zeigt

1. **Iterated Local Search in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Erster Abstieg** (die Tour vor dem ersten Kick) → **Kicks** (Iterations-Regler + ▶️ Kicks abspielen: Länge der aktuellen/besten Tour über die bewerteten Nachbarschaften, dazu die Tour nach der gewählten Iteration) → **Ergebnis** (beste Tour neben der besten aus Hill Climbing mit Neustarts, Kandidatenliste + DLB).
2. **Was die Kette gefunden hat:** beste und letzte Tour, ein Abstieg, Hill Climbing mit Neustarts (beide Varianten), angenommene Iterationen; Urteil (`beats_hc` → `comparable` → `hc_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, Störstärke, Annahme, lokale Suche, Stopps, Gruppen und Startlösung (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen; **Störstärke** (1 bis 8 Doppelbrücken bei knappem Budget – der Sweet-Spot-Fund); **Annahme** ("nur besser" gegen "immer annehmen"); **Lokale Suche** (Kandidatenliste + DLB gegen vollen Rescan); **Streuung** über 20 Ketten; **Skalierung** von 20 bis 200 Stopps.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (die lokale Suche ist billig, die Störstärke passt, die Annahmeregel filtert, das Budget reicht für viele Iterationen, die Störung ist strukturell).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Lokale Suche** (Kandidatenliste + DLB / voller Rescan), **Störstärke** (1–8 Doppelbrücken je Kick), **Annahme** (nur besser / immer),
**Budget** (10 Tausend bis 2 Millionen bewertete Nachbarschaften), **Startlösung**, Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

Die einzelne Standardinstanz landet hier durch Zufall sehr nah am echten Optimum (0.1 % statt der 0.64 % im Mittel über fünf Instanzen); die Mittelwerte oben in der Tabelle sind die belastbaren Zahlen. Presets wie "Immer annehmen" oder "Zu starke Störung" zeigen ihre Pointe trotzdem klar (siehe App-Hilfetexte für die genauen Einzelwerte).

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_hc / comparable |
| Kleines Budget (10 Tausend) | beats_hc / comparable / hc_wins |
| Voller Rescan (langsam) | hc_wins / comparable |
| Immer annehmen (Random Walk) | beats_hc / comparable |
| Zu starke Störung (8 Doppelbrücken) | beats_hc / comparable |
| Kalibrierte Störung (3 Doppelbrücken) | beats_hc / comparable |
| Nächster Nachbar als Start | beats_hc / comparable / hc_wins |
| Große Instanz (200 Stopps, 1 Million) | beats_hc |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`ils_scenario.py`, `ils_tour.py`): wortgleiche Kopie aus der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) (per Test gegen eingefrorene Werte), über die [simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), die bereits das Bewertungsbudget `max_evaluations` ergänzt hat.
- **Doppelbrücken-Zug** (`ils_kick.py`): drei zufällige Schnittpunkte teilen die Tour in vier Stücke; die mittleren beiden tauschen den Platz. Mindestlänge 2 je Stück verhindert, dass der Zug zu einem einzelnen 2-opt-Zug entartet (geprüft: erschöpfend gegen alle 2-opt-Züge auf kleinen Instanzen, keine einzige Doppelbrücke ist darunter).
- **Kandidatenliste + Don't-Look-Bits** (`ils_dlb.py`, aus der Hill-Climbing-Demo übernommen und um den `touched`-Kurzweg erweitert): nach einem Kick startet die Warteschlange nur mit den Endpunkten der neuen Kanten, nicht mit allen Knoten – das macht den Wiederabstieg billig (geprüft: erreicht dasselbe Lokaloptimum wie ein voller Scan, mit weit weniger Bewertungen).
- **ILS-Schleife** (`ils_algorithm.py`): erster Abstieg aus der Startlösung, dann Kick → Wiederabstieg → Annahme ("nur besser": übernehmen, wenn mindestens so kurz, sonst bei der vorigen Tour bleiben; "immer": Random Walk über lokale Optima), bis das Bewertungsbudget erschöpft ist. Ein Vorschlag = eine bewertete Nachbarschaft, dieselbe Einheit wie in den Geschwister-Demos.
- **Hill Climbing mit Neustarts, zwei Varianten** (`ils_evaluation.py`): voller Rescan (Kontinuität mit der Wurzel-Demo) und Kandidatenliste + DLB (die **faire** Vergleichsgröße, da ILS dieselbe lokale Suche nutzt); beide: Abstiege aus zufälligen Startlösungen, bis das Budget erreicht ist.
- **Auswertung** (`ils_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung (vor dem Bau notiert): "ILS schlägt Neustarts, weil es Struktur wiederverwendet"** – **bestätigt, aber nur mit der richtigen lokalen Suche und nur bei knappem Budget**: mit vollem Rescan (kein Kurzweg für den Wiederabstieg) verliert ILS klar gegen Neustarts (3.58 % gegen 0.68 % bei Kandidatenliste + DLB) – ohne einen billigen Wiederabstieg ist "Struktur wiederverwenden" kein Vorteil, weil jede Iteration fast so teuer ist wie ein Neustart. Und selbst mit der günstigen lokalen Suche schrumpft der Vorsprung mit wachsendem Budget (1.36 gegen 2.35 % bei 10 Tausend, 0.58 gegen 0.58 % bei 2 Millionen): ab genug Neustarts holt der Neustart die Wiederverwendung ein, weil beide dieselbe billige lokale Suche nutzen.
- **Die Störstärke ist ein Kompromiss, keine "mehr ist besser"-Größe.** 1 Doppelbrücke (der in der Literatur übliche Standardwert) ist bei knappem Budget NICHT optimal (1.03 %); der Sweet Spot liegt bei 2-3 (0.78-0.85 %); bei 8 wird es wieder schlechter (1.11 %) – nicht vorhergesagt, sondern gemessen.
- **Die Annahmeregel entscheidet fast alles über die LETZTE Tour, kaum etwas über die BESTE.** "Immer annehmen" findet praktisch dieselbe beste Tour (0.67 % gegen 0.64 %, die beste Tour wird ja unabhängig von der Annahme gemerkt), aber die Kette selbst wandert bei 200 Tausend Vorschlägen bis auf **14.18 %** über der Schranke weg – ein Befund, der zeigt, wie wichtig "nur besser" für die Praxis ist (wo meist die letzte, nicht die beste Tour gebraucht wird), obwohl er die Demo-Kennzahl "beste Tour" kaum berührt.
- **Startlösung ist bei ILS fast egal** (0.64 % gegen 0.64 %), obwohl sie bei einem einzelnen Hill-Climbing-Abstieg noch deutlich zählt (6.8 gegen 5.4 %) – die vielen Kicks vergessen die Startlösung schneller, als ein einzelner Abstieg es könnte.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Doppelbrücken-Zug:** gültige Permutation, kein No-op, erschöpfende Prüfung gegen alle 2-opt-Züge auf kleinen Instanzen (keine einzige Doppelbrücke ist ein verkleideter 2-opt-Zug, ab Mindestlänge 2 je Stück).
- **Kandidatenliste + Don't-Look-Bits mit `touched`-Kurzweg:** `touched=None` verhält sich exakt wie das Original (aus der Hill-Climbing-Demo übernommene Korrektheitstests); der Kurzweg nach einem Kick erreicht dasselbe Lokaloptimum wie ein voller Scan, mit weit weniger Bewertungen, und ändert nachweislich nur die Kanten der betroffenen Knoten (unabhängige Prüfung: jeder unbeteiligte Knoten behält seine beiden Nachbarn).
- **ILS-Schleife:** Budget-Buchführung (Summe der Bewertungen über alle Wiederabstiege), Monotonie der aktuellen Länge unter "nur besser", Determinismus je Seed, Regressionsschutz für beide lokalen Suchen; "immer annehmen" ist nachweislich ein Random Walk (jede Iteration wird angenommen).
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Störstärke-, Annahme-, lokale-Suche- und Größen-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 8 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Iterations-Regler, ▶️ Abspielen und ▶️ Kicks abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Störstärke, Annahme, lokale Suche, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `ils_kick.py` | Doppelbrücken-Zug (die Störung) |
| `ils_dlb.py` | Kandidatenliste + Don't-Look-Bits für 2-opt, mit `touched`-Kurzweg für den Wiederabstieg nach einer Störung |
| `ils_algorithm.py` | Die ILS-Schleife: erster Abstieg, Kick, Wiederabstieg, Annahme |
| `ils_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `ils_scenario.py`, `ils_constants.py` | Instanzen (gleichverteilt, in Gruppen); Konstanten, Presets |
| `ils_evaluation.py` | Analyse, Urteil, Hill Climbing mit Neustarts (beide Varianten), Sweeps, Streuung, Skalierung |
| `ils_presets.py`, `ils_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Doppelbrücken-Zug, Kandidatenliste + Don't-Look-Bits, ILS-Schleife, übernommener Kern, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
