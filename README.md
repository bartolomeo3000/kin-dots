# kin-dots

An evolution simulation of dot agents that eat, reproduce and die, and that may (or may not) evolve altruism. Nothing in the rules tells them to help relatives, and they cannot recognise relatives. We check whether they start doing it anyway, and why.

**Main result: self-sacrifice.** A dot can give all its energy to a neighbour and die. Two genes decide how likely it is to do so. Over ~30,000 steps this rises from ~5% to **~70% of all deaths** when the gift goes to a neighbour (usually a relative). In the control, where the gift goes to a random dot anywhere on the map, it stays at **5–10%**. The two worlds differ in that one rule only.

**Start here: [Dots that give their lives](dots-that-give-their-lives.md)**, an explainer for newcomers: the rules, the experiment, the result and why it happens.

![The two demo worlds side by side](docs/demo_compare.png)

**Other threads:**

- Sharing with kin and restraint towards kin also evolve, and the controls show that relatedness is what drives them.
- Alarm calls against predators: the result was inconclusive.
- Two species with a front line, fighting only the other species and defending kin: parked.

## Installation

Native Python 3.12 via [uv](https://docs.astral.sh/uv/) (`brew install uv`):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

On Apple Silicon use a native (arm64) Python. An x86 Python under Rosetta crashes in Numba.

## The self-sacrifice demo

Two worlds side by side, same start and same random seed. Left: the gift goes to a neighbour. Right: the control, where the gift goes to a stranger.

```bash
./demo.sh            # seed 0, up to 64 steps per frame
./demo.sh 3 128      # seed 3, up to 128 steps per frame
```

What you see:

- Each world, with its population. Next to the count are last year's mean, min and max.
- The **share of deaths that are self-sacrifice**: all sacrifices of the last year divided by all deaths of the last year.
- Side panels with the **family colours**: a hue histogram now, and the same histogram over the whole run (time runs downward; a vertical band is one family).
- Two plots over the whole run: the sacrifice share, and the raw population.

Keys:

| key | action |
|---|---|
| `SPACE` | pause |
| `↑` / `↓` | faster / slower |
| `C` | colour mode: `p(give)`, `energy`, `tag` (inherited family colour), `family map (yearly)` |
| `F` | show / hide grass |
| `G` | show / hide the gifts of the last step (white lines) |
| `H` | show / hide the side panels |
| `R` | restart with the same seed |
| `Esc` / `Q` | quit |

A clear difference shows after ~10,000 steps, a few minutes at speed 64. The window fits the screen and the simulation runs in a fixed time budget per frame, so the window stays responsive. If a world dies out in the first winters (an unlucky seed), start with another seed.

There is also a port of the demo as an app for [Evolife](https://evolife.telecom-paris.fr) (J.-L. Dessalles's framework). It is not included in this repository.

## Figures and analyses

All figures are in `docs/`. Every script prints its own usage at the top of the file.

| figure | what it shows | made by |
|---|---|---|
| `demo_sacrifice.png` | share of deaths that are sacrifice, gene `b`, yearly mean population | `kin_dots.experiments.sweep` + `analysis/plot_sacrifice.py` |
| `demo_sacrifice_genes.png` | gene `a`, and p(sacrifice) as a function of own energy over time | `analysis/sacrifice_genes.py` |
| `demo_population.png` | raw population at every step | `analysis/population_runs.py` + `analysis/plot_population.py` |
| `demo_age_at_death.png`, `..._detail.png` | age at death by cause (sacrifice / hunger / old age), mean vs median, children before death, time of year | `analysis/age_at_death.py` |
| `demo_sacrifice_rate.png` | sacrifices per dot per step: through the year, over the run, by age | `analysis/age_at_death.py` |
| `demo_tags_yearly.gif` | yearly map of family colours over 150 years, with hue strips | `analysis/tag_movie.py` |

For example, to rebuild the main figure (~8 minutes on 6 cores):

```bash
.venv/bin/python -m kin_dots.experiments.sweep --steps 30000 --seeds 0 1 2 --windows 50 --procs 6 --out results/demo_curves \
  --cond-config sacrifice-local configs/demo_sacrifice.yaml --cond-config sacrifice-strangers configs/demo_sacrifice_strangers.yaml
.venv/bin/python analysis/plot_sacrifice.py results/demo_curves docs/demo_sacrifice.png
```

## The general simulation

Single-world live view (pygame):

```bash
.venv/bin/python -m kin_dots.viz.app --config configs/kin_altruism.yaml --speed 16
```

Keys:

- `SPACE` pause, `→` one step (when paused), `↑/↓` speed.
- `C` colour mode: `tag`, `species`, `p(share|kin)`, `p(attack|stranger)`, `p(alarm|kin)`, `p(defend|kin)`, `harvest`, `energy`.
- `F` food layer, `G` gifts, `R` restart, `N` new seed, `Esc` quit.

Headless, writing a time series to CSV:

```bash
.venv/bin/python -m kin_dots.experiments.run --steps 5000 --out results/run.csv
```

Both commands take `--config <file.yaml>` (default parameters otherwise) and `--seed N`.

Two worlds side by side with any pair of presets:

```bash
.venv/bin/python -m kin_dots.viz.compare --left configs/kin_altruism.yaml --right configs/kin_altruism_random_recipient.yaml \
  --left-label "help a neighbour" --right-label "help a random dot" --metric share_rate --speed 16
```

Many conditions and seeds in parallel, with a comparison table. `--table` picks the columns: `kin` relatedness metrics, `behaviour` genes and behaviour, `pred` predators and alarm calls, `species` species and defence.

```bash
.venv/bin/python -m kin_dots.experiments.sweep --steps 40000 --seeds 0 1 2 \
  --cond kin season_period=600 season_amplitude=0.9 move_speed=0.1 attacks_enabled=False \
  --cond control season_period=600 season_amplitude=0.9 move_speed=0.1 attacks_enabled=False share_to_random=True
```

Run sweeps from a file or the command line, not from stdin: on macOS, multiprocessing from stdin hangs.

### Presets in `configs/`

- `demo_sacrifice.yaml` / `demo_sacrifice_strangers.yaml`: the self-sacrifice demo and its control.
- `kin_altruism.yaml`: sharing with kin evolves.
- `kin_altruism_random_recipient.yaml`: control where the recipient is drawn from the whole map; altruism disappears.
- `kin_altruism_tagblind.yaml`: control where the sharing decision cannot see colour.
- `kin_restraint.yaml`: a variant with fights in which restraint towards kin evolves.
- `kin_alarm.yaml` / `kin_alarm_random_warn.yaml`: predators (red triangles) and alarm calls (white circles); the second is the control.
- `two_species.yaml` / `two_species_fights_within.yaml`: two species (dots and squares), fighting only the other species, defending a neighbour; the second is the control.
- `default.yaml`: the defaults.

## Tests

```bash
.venv/bin/pytest
```

The tests cover determinism (same seed, same run), exact energy and food bookkeeping, the kinship metrics, predators and the two-species variant.

## Code layout

- `kin_dots/core/params.py`: all parameters (`Params`).
- `kin_dots/core/genome.py`: genome layout (which columns are which genes).
- `kin_dots/core/world.py`: state as NumPy arrays and the simulation step.
- `kin_dots/core/kernels.py`: Numba kernels (neighbours, sharing and sacrifice, fights, social movement).
- `kin_dots/core/genealogy.py`: family tree and interaction log.
- `kin_dots/core/kinship.py`: kinship metrics (relatedness, Hamilton's rule, Price equation, clans and cheaters).
- `kin_dots/core/predation.py`: predators, detection, alarm calls (Numba).
- `kin_dots/viz/app.py`: single-world pygame window.
- `kin_dots/viz/compare.py`: two worlds side by side (the demo viewer).
- `kin_dots/experiments/run.py`: headless runner.
- `kin_dots/experiments/sweep.py`: many conditions and seeds in parallel, comparison table.
- `analysis/`: scripts behind the figures in `docs/`.
- `configs/`: parameter presets.
- `tests/`: determinism, energy and food balance, state consistency.
