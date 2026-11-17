# kin-dots

Ewolucyjna symulacja kropek-agentów, które jedzą, walczą, rozmnażają się i mogą (ale nie muszą) wyewoluować altruizm. Nic w regułach nie każe im pomagać krewnym. Sprawdzamy, czy i dlaczego zaczną to robić.

Szczegóły modelu, metryki, eksperymenty i kamienie milowe: [PLAN.md](PLAN.md).

**Status:** M3 gotowy, dodane drapieżniki i krzyk ostrzegawczy. Dzielenie się z krewnymi i powściągliwość wobec krewnych ewoluują, a ich źródłem jest pokrewieństwo (kontrole). Krzyk ostrzegawczy: wynik niejednoznaczny. Szczegóły: sekcja 8 w [PLAN.md](PLAN.md).

## Instalacja

Natywny Python 3.12 przez [uv](https://docs.astral.sh/uv/) (`brew install uv`):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

## Uruchomienie

Podgląd na żywo (pygame):

```bash
.venv/bin/python -m kin_dots.viz.app
```

Klawisze: `SPACE` pauza, `→` jeden krok, `↑/↓` szybkość, `C` kolorowanie (znacznik / p(dzielenie się z krewnym) / p(atak na obcego) / p(krzyk przy krewnych) / gen harvest / energia), `F` warstwa jedzenia, `R` restart, `N` nowe ziarno, `Esc` wyjście.

Bez grafiki, zapis szeregu czasowego do CSV:

```bash
.venv/bin/python -m kin_dots.experiments.run --steps 5000 --out results/run.csv
```

Obie komendy przyjmują `--config configs/default.yaml` (lub własny YAML) i `--seed N`.

Gotowe presety w `configs/` (opis w sekcji 8 w [PLAN.md](PLAN.md)):

- `kin_altruism.yaml`: warunek, w którym ewoluuje dzielenie się z krewnymi.
- `kin_altruism_random_recipient.yaml`: kontrola z biorcą losowanym z całej mapy, w której altruizm znika.
- `kin_altruism_tagblind.yaml`: kontrola, w której dzielenie się nie widzi koloru.
- `kin_restraint.yaml`: wariant z walkami, w którym ewoluuje powściągliwość wobec krewnych.
- `kin_alarm.yaml` / `kin_alarm_random_warn.yaml`: drapieżniki (czerwone trójkąty) i krzyk ostrzegawczy (białe kółka); druga wersja to kontrola.

```bash
.venv/bin/python -m kin_dots.viz.app --config configs/kin_altruism.yaml --speed 16
```

Porównanie wielu warunków i ziaren naraz (równolegle; `--table kin` pokazuje metryki pokrewieństwa, `--table behaviour` geny i zachowanie, `--table pred` drapieżniki i krzyk):

```bash
.venv/bin/python -m kin_dots.experiments.sweep --steps 40000 --seeds 0 1 2 \
  --cond kin season_period=600 season_amplitude=0.9 move_speed=0.1 attacks_enabled=False \
  --cond kontrola season_period=600 season_amplitude=0.9 move_speed=0.1 attacks_enabled=False share_to_random=True
```

Testy:

```bash
.venv/bin/pytest
```

## Układ kodu

- `kin_dots/core/params.py`: wszystkie parametry (`Params`).
- `kin_dots/core/genome.py`: układ genomu (które kolumny to które geny).
- `kin_dots/core/world.py`: stan jako tablice NumPy i krok symulacji.
- `kin_dots/core/kernels.py`: funkcje Numby (sąsiedzi, dzielenie się, walki, ruch społeczny).
- `kin_dots/core/genealogy.py`: drzewo genealogiczne i dziennik interakcji.
- `kin_dots/core/kinship.py`: metryki M3 (pokrewieństwo, Hamilton, Price, klany i oszuści).
- `kin_dots/core/predation.py`: drapieżniki, wykrywanie, krzyk ostrzegawczy (Numba).
- `kin_dots/viz/app.py`: okno pygame.
- `kin_dots/experiments/run.py`: runner bez grafiki.
- `kin_dots/experiments/sweep.py`: wiele warunków i ziaren równolegle, tabela porównawcza.
- `configs/`: presety parametrów.
- `tests/`: testy determinizmu, bilansu energii i jedzenia oraz poprawności stanu.
