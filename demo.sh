#!/bin/zsh
# Self-sacrifice demo: two worlds side by side (see dots-that-give-their-lives.md).
# Usage: ./demo.sh [seed] [max steps per frame]
cd "$(dirname "$0")"
exec .venv/bin/python -m kin_dots.viz.compare \
  --left configs/demo_sacrifice.yaml --right configs/demo_sacrifice_strangers.yaml \
  --left-label "gift to a neighbour" --right-label "gift to a stranger" \
  --metric sacrifice_frac --seed "${1:-0}" --speed "${2:-64}"
