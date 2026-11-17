# kin-dots

Ewolucyjna symulacja kropek-agentów, które jedzą, walczą, rozmnażają się i mogą (ale nie muszą) wyewoluować altruizm. Nic w regułach nie każe im pomagać krewnym. Sprawdzamy, czy i dlaczego zaczną to robić.

Szczegóły modelu, metryki, eksperymenty i kamienie milowe: [PLAN.md](PLAN.md).

**Status:** M2 gotowy (dzielenie się, atak, ruch do swoich i od obcych, statystyki interakcji). Pierwsze wyniki i wnioski: sekcja 8 w [PLAN.md](PLAN.md).

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

Klawisze: `SPACE` pauza, `→` jeden krok, `↑/↓` szybkość, `C` kolorowanie (znacznik / p(dzielenie się z krewnym) / p(atak na obcego) / gen harvest / energia), `F` warstwa jedzenia, `R` restart, `N` nowe ziarno, `Esc` wyjście.

Bez grafiki, zapis szeregu czasowego do CSV:

```bash
.venv/bin/python -m kin_dots.experiments.run --steps 5000 --out results/run.csv
```

Obie komendy przyjmują `--config configs/default.yaml` (lub własny YAML) i `--seed N`.

Testy:

```bash
.venv/bin/pytest
```

## Układ kodu

- `kin_dots/core/params.py`: wszystkie parametry (`Params`).
- `kin_dots/core/genome.py`: układ genomu (które kolumny to które geny).
- `kin_dots/core/world.py`: stan jako tablice NumPy i krok symulacji.
- `kin_dots/core/kernels.py`: funkcje Numby (sąsiedzi, dzielenie się, walki, ruch społeczny).
- `kin_dots/viz/app.py`: okno pygame.
- `kin_dots/experiments/run.py`: runner bez grafiki.
- `configs/`: presety parametrów.
- `tests/`: testy determinizmu, bilansu energii i jedzenia oraz poprawności stanu.
