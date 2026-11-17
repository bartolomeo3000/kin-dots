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

**Python 3.11+, NumPy, Numba, pygame.** Projekt zarządzany przez `pyproject.toml` (uv lub pip + venv).

- **Stan jako tablice, nie obiekty.** Agenci to zestaw tablic NumPy (`x`, `y`, `energy`, `age`, `parent_id`, `genome[N, G]`, `alive`), a jedzenie to tablica 2D. Martwe sloty są ponownie używane (wolna lista), żeby uniknąć ciągłego kopiowania.
- **Wektorowo w NumPy:** odrost jedzenia, metabolizm, śmierć i mutacje, czyli wszystko, co liczy się dla każdego agenta niezależnie.
- **Numba (`@njit`) dla pętli z sąsiadami:** budowa spatial hasha, dzielenie się, atak, ruch w stronę jedzenia lub sąsiadów. Kod wygląda jak zwykła pętla `for`, ale działa mniej więcej z szybkością C.
- **Determinizm:** jedno ziarno → `np.random.Generator`, a w funkcjach Numby jawnie przekazywane ziarno. Ten sam config i to samo ziarno dają ten sam przebieg.
- **Parametry:** jeden `dataclass` `Params` serializowany do YAML/JSON. Presety E0–E5 to pliki w `configs/`.

Struktura:

```
kin_dots/
  core/          # czysta symulacja, bez grafiki
    params.py    #   dataclass Params
    world.py     #   stan (tablice) + step()
    kernels.py   #   funkcje @njit (sąsiedzi, interakcje)
    genealogy.py #   drzewo przodków, pokrewieństwo
    metrics.py   #   Hamilton, Price, klastry
  viz/
    app.py       # okno pygame: kropki, suwaki, pauza/krok, inspektor agenta, mini-wykresy
  experiments/
    run.py       # CLI: config + ziarna → CSV/Parquet, równolegle przez multiprocessing
configs/         # presety E0–E5
analysis/        # notebooki Jupyter (pandas + matplotlib)
tests/           # pytest: zachowanie energii, determinizm, brak ukrytych nagród
```

Cele wydajnościowe:
- podgląd: ~3–5 tys. agentów przy ≥30 FPS (render pygame co krok lub co k kroków),
- headless: ≥1000 kroków/s przy tej samej populacji, bez renderu.

Uwaga: pierwsze uruchomienie trwa kilka sekund dłużej, bo Numba kompiluje funkcje (`cache=True` zapisuje wynik na dysku).

## 6. Kamienie milowe

1. **M1: Szkielet.** `pyproject.toml`, `Params`, seeded RNG, świat z jedzeniem, agenci z ruchem, jedzeniem, metabolizmem, rozmnażaniem i śmiercią. Prosty podgląd w pygame. Populacja stabilna. Testy determinizmu i bilansu energii.
2. **M2: Interakcje.** Spatial hash w Numbie, dzielenie się i atak przez sigmoidy z genomu, mutacje, znaczniki jako kolor kropki.
3. **M3: Instrumentacja.** Drzewo genealogiczne, metryki z sekcji 3, mini-wykresy w oknie i zapis szeregów czasowych.
4. **M4: UI.** Suwaki (`b`, `c`, `d`, mutacje, odrost jedzenia), pauza/krok, inspektor agenta po kliknięciu, ładowanie presetów E0–E5.
5. **M5: Eksperymenty.** `experiments/run.py` z `multiprocessing`, przemiatanie parametrów, notebook z diagramem fazowym E4.
6. **M6 (v2).** Drapieżniki i sygnały ostrzegawcze, rozmnażanie płciowe, ewoluujące sieci neuronowe zamiast sigmoid.

## 7. Ryzyka

- **Nic ciekawego nie wyewoluuje.** Strojenie parametrów może zająć dużo czasu. Zaczynamy od korzystnych warunków (`b/c ≈ 3`, mały `d`) i dopiero potem je utrudniamy.
- **Ukryta nagroda w regułach.** Na przykład darowizna zwiększa szansę narodzin u dawcy. Każda reguła przechodzi przegląd pod kątem „czy osobnik zyskuje bezpośrednio?”.
- **Opieka rodzicielska udaje altruizm.** Mierzymy ją osobno (sekcja 3).
- **Wydajność drzewa genealogicznego.** Przycinamy wymarłe gałęzie i liczymy pokrewieństwo na próbce.

## 8. Notatki z implementacji i pierwsze wyniki

### M2 (2026-11-17)

- **Dzielenie się nie może tworzyć energii.** Przy `b > c` dwie sąsiednie kropki oddające sobie energię nawzajem tworzą ją z niczego. Populacja dochodzi wtedy do limitu tablic w ~250 krokach, a średnia energia rośnie do ~300. Dlatego `share_benefit ≤ share_cost` (domyślnie 0.9 vs 1.0). „b” z reguły Hamiltona musi powstać z tego, *komu* się pomaga (np. ratunek głodującego krewnego), a nie z samego transferu.
- **Wynik bazowy (E0 vs E1, 30k kroków, 2 ziarna): altruizm nie ewoluuje.**
  - Dzielenie się spada z ~3% do ~0.1% spotkań w obu warunkach. Atak też zanika.
  - Kropki ewoluują *unikanie* podobnych do siebie (`move_kin` ≈ −2), co wygląda na ucieczkę przed konkurencją z krewnymi.
  - `harvest` dochodzi do maksimum, czyli zachłanność.
- **Przyczyny (hipotezy do sprawdzenia):**
  1. *Energia jest liniowo przeliczana na potomstwo.* Oddanie 1 jednostki krewnemu, który dostaje 0.9, to strata dla całej rodziny. Pomoc opłaca się dopiero przy nieliniowości: ratowanie przed śmiercią głodową lub dopchnięcie do progu rozmnażania. Obecnie mało kto umiera z głodu (dominują walki i starość).
  2. *Znacznik nie niesie informacji o pokrewieństwie.* Przy `tag_mutation_sd = 0.02` cała populacja ma podobne kolory również przy globalnym rozproszeniu (średnie podobieństwo spotkanych ~0.8 w E0 *i* w E1). Rozpoznawanie „po kolorze” nic więc nie daje.
  3. *Konkurencja z krewnymi* (pułapka Taylora) przy lokalnych narodzinach i pełnym zagęszczeniu.
- **Następne kroki:**
  - M3: mierzyć prawdziwe pokrewieństwo z drzewa genealogicznego, a nie z koloru.
  - Wprowadzić nieliniowość wartości energii (np. większa śmiertelność głodowa, zmienne w czasie łaty jedzenia).
  - Zwiększyć tempo mutacji znacznika.

### Poprawki modelu (2026-11-17, po M2)

Przesiew wariantów (skrypt `kin_dots/experiments/sweep.py`, 30–40k kroków, CSV w `results/`):

| Zmiana | Efekt |
|---|---|
| `tag_mutation_sd` 0.02 → 0.08 | Kolor zaczyna mówić coś o pokrewieństwie. Bez tej zmiany cała populacja ma podobne kolory. |
| Pory roku (`season_period=600`, `season_amplitude=0.9`) | Pojawia się śmierć głodowa, czyli nieliniowość wartości energii. **Bez pór roku dzielenie się zawsze zanika.** |
| Wolny ruch (`move_speed` 1 → 0.1) | Rodziny zostają razem. `assortment` (podobieństwo sąsiadów minus losowej pary) rośnie z ~0.1 do ~0.24. |
| Bez walk (`attacks_enabled=false`) | Walki odpowiadały za ~80% śmierci i zagłuszały wszystko inne. |

Pomiar `assortment` jest liczony z koloru, więc to tylko przybliżenie pokrewieństwa. Prawdziwe pokrewieństwo da dopiero M3.

**Wynik 1: dzielenie się z krewnymi** (`configs/kin_altruism.yaml`)

| Warunek | Ziarna | Dzielenie się po 40k kroków | `share_w1` (podobieństwo) |
|---|---|---|---|
| biorca = spotkany sąsiad | 5/5 | 6.7–8% spotkań, stabilne lub rośnie | +5.5 do +7.3 |
| biorca bez rozpoznawania koloru (`kin_altruism_tagblind`) | 2/2 | ~5.7% | ~0 (nie widzi koloru) |
| biorca = losowa kropka z mapy (`kin_altruism_random_recipient`) | 3/3 | 0.1–0.3%, zanika | −2.6 do −4.4 |

Wyewoluowana reguła to *„biedny oddaje bogatszemu krewnemu”*:
- `share_w2` ≈ −8 (dzielę się, gdy sam mam mało),
- `share_w3` ≈ +4 (daję temu, kto ma dużo),
- średnia energia dawcy ~6.4, biorcy ~12.4, próg rozmnażania 20.

Kropki z małą szansą na przeżycie chudej pory oddają energię krewnym, którzy są bliżej rozmnożenia. Mechanizm przypomina podział na pomocników i rozmnażających się. Sama bliskość krewnych wystarcza (wariant bez koloru też działa), a rozpoznawanie po kolorze podnosi poziom i kieruje pomoc do podobnych. Gdy biorca jest niespokrewniony, przy identycznej ekologii, altruizm znika. Jego źródłem jest więc pokrewieństwo, a nie same reguły świata.

**Wynik 2: powściągliwość wobec krewnych** (`configs/kin_restraint.yaml`, z walkami)
- Dzielenie się zanika, ale `attack_w1` < 0 (od −0.7 do −3.6) w 4/4 przebiegach z potomstwem obok rodzica. Zaatakowani są mniej podobni niż spotykani (0.76–0.81 vs 0.82–0.83).
- Przy potomstwie rozrzuconym po mapie nie ma tego efektu (2/2 przebiegi).

**Znane problemy:**
- Kontrola z potomstwem rozrzuconym po mapie (`dispersal_sd < 0`) **wymiera** w warunku `kin_altruism` (bez walk, z porami roku), także przy `move_speed` 0.3. Potomstwo ląduje na jałowych obszarach i ginie w pierwszej chudej porze. Dlatego czystą kontrolą jest losowy biorca, a nie globalne rozproszenie.
- Przy `move_speed` 0.1 łaty jedzenia długo stoją puste, bo kropki nie nadążają ich zasiedlać.

### M3: prawdziwe pokrewieństwo (2026-11-17)

Nowe pomiary (`kin_dots/core/genealogy.py`, `kin_dots/core/kinship.py`):
- **Drzewo genealogiczne:** rodzic, pokolenie i krok narodzin każdej kropki.
- **Dziennik interakcji:** kto spotkał, wspomógł albo zaatakował kogo, plus losowe pary jako punkt odniesienia.
- **`r`:** pokrewieństwo genetyczne na genach zachowania (bez koloru). 1 oznacza klony, 0 losową parę.
- **Bliskość w drzewie:** pokolenia do wspólnego przodka.
- **Komu się pomaga:** odsetek pomocy dla własnych dzieci i rodziców.
- **Reguła Hamiltona** w jednostkach energii.
- **Równanie Price'a** dla każdego genu (okno 600 kroków, czyli jedna pora roku).
- **Klany i oszuści:** klan to grupa sąsiadów, którzy są bliskimi krewnymi (odległość w drzewie ≤ 4).

Pomiary nie zmieniają przebiegu symulacji (osobny generator losowy, sprawdzone testem). Kosztują ~19% czasu kroku.

Wyniki (40k kroków, 2 ziarna na warunek, okna co 8000 kroków; `results/m3/`):

| | `kin_altruism` | `kin_restraint` | losowy biorca |
|---|---|---|---|
| `r` spotykanych sąsiadów | 0.61–0.87 | 0.69–0.85 | 0.82–0.92 |
| `r` wspomaganych | **0.75–0.88, w 10/10 oknach > spotykani** (w pierwszym oknie minimalnie) | — | **≈ 0** (−0.16…0.05) |
| bliscy krewni (≤4) wśród wspomaganych vs spotykanych | **0.46–0.70 vs 0.30–0.66** (wyżej w 8/10 oknach) | — | ~0 vs 0.40–0.86 |
| pomoc dla własnego dziecka | ~15% | — | 0% |
| `r` atakowanych | — | **0.52–0.81, w 10/10 oknach < spotykani** | — |
| `r·b − c` (energia) | **−0.20 … −0.33** | | −1 |
| selekcja na `share_w1` (Price) | **dodatnia w 10/10 oknach** (0.03–0.34) | zmienna | zmienna |

**Wnioski:**
- **Pomoc trafia do krewnych bardziej, niż wynikałoby z samego sąsiedztwa.** `r` i odsetek bliskich krewnych są wyższe u wspomaganych niż u przypadkowo spotkanych. Tylko ~15% pomocy to opieka nad własnym dzieckiem. Reszta trafia do rodzeństwa, kuzynów i innych krewnych, więc to nie jest zwykła opieka rodzicielska.
- **Powściągliwość wobec krewnych potwierdzona genealogicznie:** atakowani są w każdym oknie mniej spokrewnieni niż spotykani.
- **Selekcja (Price) w każdym oknie pcha w górę wagę „dziel się z podobnym” (`share_w1`).** Bias ogólnej szczodrości (`share_w0`) jest zwykle pchany w dół. Ewoluuje więc pomaganie *wybiórcze*, nie ogólna szczodrość.
- **Naiwna reguła Hamiltona w jednostkach energii mówi „nie opłaca się”** (`r·b − c < 0`), a dzielenie się i tak jest selekcjonowane. Prawdziwe `b` i `c` trzeba liczyć w potomstwie, nie w energii. Energia oddana przez kropkę, która i tak umrze z głodu, kosztuje ją prawie nic (zob. `share_w2 < 0`).
- **Oszustów prawie nie ma** (0–3% członków klanów). Cykle „klan → oszuści → upadek” jeszcze się nie pojawiły. Możliwe, że wymagają dłuższych przebiegów albo silniejszego altruizmu.

**Do zrobienia później:**
- Regresja Queller/Grafen: `b` i `c` w jednostkach fitnessu (liczba potomków w następnym oknie w zależności od energii oddanej i otrzymanej).
- Liczba klanów mocno zależy od pory roku (od 0 do ~600). `sweep` wyrównuje teraz okna raportów do wielokrotności okresu pór roku. Powyższa tabela powstała jeszcze bez tego wyrównania, dlatego klanów w niej nie ma.

### Drapieżniki i krzyk ostrzegawczy (2026-11-17)

**Model** (`kin_dots/core/predation.py`, parametry `n_predators` itd.):
- Drapieżniki nie ewoluują. Gonią najbliższą kropkę w zasięgu wzroku i po udanym polowaniu robią przerwę na trawienie.
- Kropka zauważa drapieżnika z prawdopodobieństwem `detect_prob` i się chowa. Schowaną 5× trudniej złapać.
- Nowe geny `alarm_w0..3` decydują o krzyku na podstawie: podobieństwa sąsiadów, własnej energii i liczby sąsiadów. Krzyk chowa wszystkich sąsiadów.
- Koszt dla krzyczącego: nie może się schować, płaci energię, a drapieżnik, który go widzi, wybiera go pierwszego (ostatnie można wyłączyć: `predators_target_callers`).
- Kontrola `alarm_to_random`: ostrzeżenie trafia do losowych kropek z mapy, a nie do sąsiadów.
- Genom ma teraz 19 genów. Dla tego samego ziarna liczby różnią się więc od wcześniejszych przebiegów. Wnioski z dzielenia się i powściągliwości sprawdziłem ponownie i nadal się trzymają (`results/recheck`).

**Wyniki** (40k kroków, 2 ziarna na warunek; `results/pred1..3`):

| Warunek | Krzyk (na wykrycie): start → koniec | Uwagi |
|---|---|---|
| 200 drapieżników, bez pór roku, `detect_prob` 0.3 | 13% → 3.6% (2/2), kontrola tak samo | krzyk zanika |
| jw., `detect_prob` 0.05 | 18% → 14% (2/2) | powoli zanika |
| jw., drapieżnik nie wybiera krzyczących | 20% → 16–19% | ≈ neutralny, kontrola tak samo |
| **60 drapieżników + pory roku** | **wzrost w 3/4 przebiegów (do 39%, 60%, 76%)**, w 1 spadek do 5% | w jednym `alarm_w1` → +3.5 (krzycz przy krewnych) |
| jw., kontrola `alarm_to_random` | 1/2 wzrost (40%), 1/2 bez zmian (19%) | |

**Wnioski (wstępne):**
- **Bez pór roku krzyk nie ewoluuje.** Populacja stoi na granicy pojemności środowiska, `r` sąsiadów spada do ~0.59 (wobec ~0.85 przy porach roku), a uratowany sąsiad to też konkurent o jedzenie (pułapka Taylora). Wygląda to na ten sam mechanizm, który blokował dzielenie się.
- **Z porami roku krzyk często rośnie, ale wynik jest niejednoznaczny.** Kontrola też raz wzrosła, a w chudej porze populacja spada do ~100–450 kropek, więc dryf losowy jest silny.
- **Do rozstrzygnięcia:** więcej ziaren (≥ 5 na warunek) i większy świat (populacja w chudej porze ≥ 1000).
- **Ekologia przy wolnym ruchu:** kropki żyją w wyspowych koloniach, które wyjadają do zera, a resztę mapy pokrywa jedzenie, do którego nie docierają. Warto rozważyć szybszy ruch w porze żyznej albo większe rozproszenie potomstwa.

Presety: `configs/kin_alarm.yaml`, `configs/kin_alarm_random_warn.yaml`.
