#!/bin/zsh
# Self-sacrifice demo: two worlds side by side (see dots-that-give-their-lives.md).
#
# Usage: ./demo.sh [seed] [max steps per frame] [--set-file FILE.yaml] [--set KEY=VAL ...]
#   ./demo.sh                                   seed 0, up to 64 steps per frame
#   ./demo.sh 3 128                             seed 3, up to 128 steps per frame
#   ./demo.sh --set season_amplitude=0.7        milder winters in BOTH worlds
#   ./demo.sh 1 --set move_speed=0.2 dispersal_sd=4
#   ./demo.sh --set-file my_world.yaml          your changes from a file (both worlds),
#                                               template: configs/demo_overrides_example.yaml
# Parameters: configs/demo_sacrifice.yaml (the control differs only by share_to_random),
# names, defaults and meanings in kin_dots/core/params.py.
cd "$(dirname "$0")"
seed=0
speed=64
if [[ -n "$1" && "$1" != -* ]]; then seed=$1; shift; fi
if [[ -n "$1" && "$1" != -* ]]; then speed=$1; shift; fi
exec .venv/bin/python -m kin_dots.viz.compare \
  --left configs/demo_sacrifice.yaml --right configs/demo_sacrifice_strangers.yaml \
  --left-label "gift to a neighbour" --right-label "gift to a stranger" \
  --metric sacrifice_frac --seed "$seed" --speed "$speed" "$@"
