#!/bin/zsh
# Demo poświęcenia: dwa światy obok siebie (opis w DEMO.md).
# Użycie: ./demo.sh [ziarno] [kroki na klatkę]
cd "$(dirname "$0")"
exec .venv/bin/python -m kin_dots.viz.compare \
  --left configs/demo_sacrifice.yaml --right configs/demo_sacrifice_strangers.yaml \
  --left-label "dar dla sąsiada" --right-label "dar dla obcego" \
  --metric sacrifice_frac --seed "${1:-0}" --speed "${2:-64}"
