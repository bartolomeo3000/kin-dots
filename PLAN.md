# kin-dots: plan projektu

Symulacja ewolucji kropek-agentów, w której altruizm (kosztowne pomaganie innym) **może** się pojawić, ale nie jest nigdzie zapisany w regułach. Cel: zaobserwować go, a potem eksperymentami kontrolnymi pokazać, *skąd* się bierze (pokrewieństwo przestrzenne, znaczniki, czy coś innego).

---

## 1. Co poprawiam względem pierwotnego pomysłu

| Pierwotnie | Po poprawce | Dlaczego |
|---|---|---|
| Stałe skłonności („dzielę się z p=0.3”) | Mała **reguła decyzyjna** z wagami w genomie (sigmoida z kilku wejść) | Pozwala wyewoluować zachowania *warunkowe* (dzielę się tylko gdy jestem syty / tylko z głodnym / tylko z podobnym), co jest dużo ciekawsze i bliższe biologii. |
| Potomek obok rodzica = krewni blisko | To samo, **plus wolne miejsce i elastyczna gęstość** | Znana pułapka (Taylor 1992): w „lepkiej” populacji o stałym zagęszczeniu krewni nie tylko sobie pomagają, ale też ze sobą konkurują i efekty się znoszą. Altruizm wyewoluuje tylko, gdy pomoc zwiększa lokalną liczebność (jest gdzie się rozmnożyć). |
| Kolor jako jeden znacznik | Znacznik **dziedziczony niezależnie** od genów zachowania, z osobnym tempem mutacji | Dzięki temu da się rozróżnić prawdziwe rozpoznawanie krewnych od „zielonej brody” i od oszustów z dobrym kolorem. |
| „Zaobserwujemy klany” | **Pomiar wprost**: pokrewieństwo dawca–biorca z drzewa genealogicznego, reguła Hamiltona i równanie Price’a liczone na żywo | Bez liczb łatwo zobaczyć wzór, którego nie ma. |
| Ostrzeganie przed drapieżnikiem | Przeniesione do v2 | Wymaga osobnego gatunku drapieżnika; MVP najpierw musi pokazać najprostszą formę altruizmu. |
| — | Nowa forma altruizmu: **powściągliwość w jedzeniu** | Jedzenie odrasta wolniej, gdy pole jest wyjedzone do zera. Zostawienie trochę to dar dla sąsiadów (tragedia wspólnego pastwiska). Kosztuje osobnika, nie wymaga żadnej nowej akcji. |

## 2. Model (MVP)

**Świat.** Torus 2D, ciągłe pozycje, podzielony na komórki siatki z zasobem jedzenia. Jedzenie odrasta logistycznie: `food += g·food·(1 − food/K)` plus mały dopływ bazowy. Dzięki temu wyjedzenie do zera jest kosztowne dla wszystkich w okolicy.

**Agent.** Pozycja, energia, wiek, `id`, `parentId`, `lineage` i genom. Każdy krok kosztuje bazowy metabolizm plus koszt ruchu.

**Percepcja.** Agent widzi agentów w promieniu `R` (przez spatial hash): odległość, znacznik, energię.

**Akcje w każdym kroku** (wybierane przez genom):
1. **Ruch**: w stronę jedzenia, od obcych, do swoich (wagi w genomie).
2. **Jedzenie**: zjada ułamek `h` zasobu ze swojej komórki (`h` jest genem, to kanał powściągliwości).
3. **Dzielenie się**: z sąsiadem `j` z prawdopodobieństwem `σ(w·x)`, gdzie `x = [1, podobieństwo_znacznika, moja_energia, energia_j]`. Dawca traci `c`, biorca zyskuje `b` (parametr `b/c` to kluczowy suwak).
4. **Atak**: analogiczna sigmoida. Wynik zależy od energii obu stron, a zwycięzca przejmuje ułamek energii przegranego (drapieżnictwo i kanibalizm).
5. **Rozmnażanie**: po przekroczeniu progu energii agent dzieli ją z potomkiem. Potomek ląduje w odległości ~`N(0, d)` (`d` = promień rozproszenia, kluczowy parametr kontrolny).

**Genom** (~12 liczb rzeczywistych, mutacja gaussowska):
`w_share[4]`, `w_attack[4]`, `harvest_h`, `move_weights[3]`, `tag[3]` (RGB, osobne tempo mutacji).

**Zasada nadrzędna:** nigdzie nie ma nagrody grupowej ani reguły „pomagaj krewnym”. Każda pomocna akcja jest dla osobnika czystą stratą energii.

## 3. Co mierzymy

- Średnie i rozkłady genów w czasie (skłonność do dzielenia się, ataku, powściągliwość).
- **Pokrewieństwo dawca–biorca**: odległość w drzewie genealogicznym (wspólny przodek) dla każdej darowizny, porównana z losową parą sąsiadów.
- **Reguła Hamiltona na żywo**: średnie `r·b` vs `c` dla wykonanych darowizn.
- **Równanie Price’a** dla cechy „dzielenie się”: rozbicie zmiany średniej na składnik selekcji (`cov(w, z)`) i transmisji.
- Klastry: liczba i rozmiar skupisk o podobnym znaczniku, odsetek oszustów (dobry znacznik + niskie dzielenie się) w każdym z nich.
- Odsetek darowizn do bezpośrednich potomków. Trzeba to odróżnić, bo opieka rodzicielska to trywialny altruizm.

## 4. Eksperymenty

| ID | Warunek | Oczekiwanie |
|---|---|---|
| E0 | Bazowy: lokalne narodziny, dziedziczne znaczniki | Altruizm się pojawia, widoczne klany i cykle oszustów. |
| E1 | **Globalne rozproszenie** potomstwa | Altruizm zanika, bo nie ma struktury przestrzennej. |
| E2 | Znaczniki losowe przy każdym narodzeniu | Zostaje tylko altruizm „przestrzenny”, ślepy na znaczniki. |
| E3 | Dzielenie się nie widzi znacznika (`w_share[1] = 0` na sztywno) | Jak wyżej. Porównanie z E0 pokazuje, ile daje rozpoznawanie. |
| E4 | Przemiatanie `b/c` × `d` | Diagram fazowy: gdzie altruizm przeżywa. Granica powinna przypominać `r·b > c`. |
| E5 | Stała gęstość (brak wolnego miejsca) | Test pułapki Taylora: altruizm słabnie mimo lokalnych narodzin. |

Każdy eksperyment to **N ≥ 10 ziaren**, a wynikiem jest rozkład, nie pojedynczy przebieg.

## 5. Architektura

- **TypeScript + Vite**, render na Canvas 2D (WebGL tylko jeśli zabraknie wydajności).
- `src/core/`: czysta, deterministyczna symulacja (seeded RNG, typed arrays, spatial hash) bez zależności od DOM.
- `src/ui/`: canvas, suwaki parametrów, wykresy na żywo (uPlot), kliknięcie w agenta pokazuje jego genom i linię przodków.
- `src/headless/`: runner w Node do eksperymentów wsadowych, wyniki w CSV/JSON.
- `analysis/`: notebook Pythona do wykresów z wyników wsadowych.
- Cel wydajnościowy: ~2–5 tys. agentów przy 60 FPS w przeglądarce, a headless jak najszybciej.

## 6. Kamienie milowe

1. **M1: Szkielet.** Vite + TS, seeded RNG, świat z jedzeniem, agenci z ruchem, jedzeniem, metabolizmem, rozmnażaniem i śmiercią. Populacja stabilna.
2. **M2: Interakcje.** Dzielenie się i atak przez sigmoidy z genomu, mutacje, znaczniki jako kolor.
3. **M3: Instrumentacja.** Drzewo genealogiczne, metryki z sekcji 3, panel wykresów.
4. **M4: UI.** Suwaki (`b`, `c`, `d`, mutacje, odrost jedzenia), pauza/krok, inspektor agenta, presety eksperymentów E0–E5.
5. **M5: Headless + eksperymenty.** Runner wsadowy, przemiatanie parametrów, diagram fazowy E4.
6. **M6 (v2).** Drapieżniki i sygnały ostrzegawcze, rozmnażanie płciowe, ewoluujące sieci neuronowe zamiast sigmoid.

## 7. Ryzyka

- **Nic ciekawego nie wyewoluuje.** Strojenie parametrów może zająć dużo czasu. Zaczynamy od korzystnych warunków (`b/c ≈ 3`, mały `d`) i dopiero potem je utrudniamy.
- **Ukryta nagroda w regułach.** Na przykład darowizna zwiększa szansę narodzin u dawcy. Każda reguła przechodzi przegląd pod kątem „czy osobnik zyskuje bezpośrednio?”.
- **Opieka rodzicielska udaje altruizm.** Mierzymy ją osobno (sekcja 3).
- **Wydajność drzewa genealogicznego.** Przycinamy wymarłe gałęzie i liczymy pokrewieństwo na próbce.
