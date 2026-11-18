# Dots that give their lives

*Evolution simulation · kin-dots*

Simple creatures eat grass, reproduce and die. They can also give all their energy to a neighbour and die. Nothing tells them to help their family, yet after a few tens of thousands of steps most of them die exactly this way. This page explains how that works and why.

## The world

Grass grows on a square map. The map has no edges: walk off the right side and you come back on the left, and the same goes for top and bottom. So the world is effectively the surface of a donut (a torus). Dots wander across the map.

- **Grass regrows**, but unevenly over the year: fast in summer, almost not at all in midwinter. One year is 600 simulation steps.
- **A dot** moves slowly towards grass and eats it. Every step of life costs a little energy.
- **Death** comes from hunger (zero energy) or old age (after 300 steps).
- **Reproduction**: once a dot has collected 20 energy, it splits in two. The child appears right next to it and inherits the parent's genes with a small random change. There are no sexes and no mates.

## One extra option

At every step a dot meets a random neighbour. It can then **give that neighbour all its energy and die**. The neighbour receives 90% of that energy. How often a dot does this depends on two numbers in its genes, `a` and `b`:

```
p(sacrifice) = sigmoid(a + b · energy / 20)
```

At the start a = −9, so sacrifice is very rare: about 5% of lives end this way. Gene `b` says whether the urge rises or falls with the dot's energy.

What this model does **not** have: dots do not recognise relatives, cannot see who the neighbour is, get no reward for helping and know nothing about groups or species. No rule says "help your family".

## The experiment: two worlds

We run two worlds with identical rules and the same random seed. They differ in one thing only: who receives the energy of a dot that sacrifices itself.

| | |
|---|---|
| **Gift to a neighbour** | The energy goes to the neighbour the dot has just met. Children are born next to their parents and dots barely move, so a neighbour is usually a close relative. |
| **Gift to a stranger** | The energy goes to a random dot somewhere on the map, almost certainly a stranger. This is the control: everything else is the same. |

> **A dot that sacrifices itself dies and has no more children. After 30,000 steps, will there be more sacrifice or less?**

Intuition says a "kill yourself" gene should die out fast, because its carriers die without offspring.

## The result

| | share of deaths that are sacrifices |
|---|---|
| **Gift to a neighbour** | **~70%** (start: about 5%). The same in every repeated run. |
| **Gift to a stranger** | **5–10%**. The gene does not spread. |

![Three charts. Left: share of deaths by sacrifice. In the gift-to-a-neighbour world it rises from a few to about 70 percent; in the control world it stays below 15 percent. Middle: gene b, which in the gift-to-a-neighbour world falls to about minus 3.5. Right: mean population, stable in the gift-to-a-neighbour world and jumping year to year in the control.](docs/demo_sacrifice.png)

*Three runs of each world (thin lines) and their mean (thick). Left panel: what share of deaths are sacrifices. Middle: gene b, which drops below zero in the gift-to-a-neighbour world. Right: mean population size in each year.*

What exactly evolved: the rule **"sacrifice yourself when you are weak"**. Gene `b` falls to about −3.5, so it is mostly low-energy dots that sacrifice themselves. They are not at death's door, though: they have about 9.6 energy on average, against a population mean of 12.5. So they give up a real chance of surviving and having offspring.

The other gene, **`a`**, sets the overall urge to sacrifice. It rises throughout the run, from −9 to about −3.5, while in the control world it stays put. The dependence on energy changes in a more interesting way. For the first ~15,000 steps it is the well-fed dots that are more likely to sacrifice; only then does the curve flip. From about step 12,000 a dot with average energy sacrifices at a nearly constant rate of about 0.5% per encounter. After that, what changes is mostly *who* sacrifices. By step 30,000 a dot with energy 5 does it with a chance of about 1.5%, and one almost ready to reproduce (19) with about 0.16%: ten times less often. The process has not stopped yet.

![Five charts. Gene a rises from minus 9 to minus 3.5 in the gift-to-a-neighbour world and stays near minus 9 in the control. The curves of sacrifice chance against energy first rise with energy and later fall with it. Heatmaps show the chance growing over time in the gift-to-a-neighbour world, most for low-energy dots. The ratio of the weak dot's chance to the well-fed dot's first drops below 1, then rises to about 10.](docs/demo_sacrifice_genes.png)

*Top left: gene a. Top right: the chance to sacrifice per encounter as a function of own energy, at successive moments of evolution (log scale). Bottom maps: the same over the whole run; the white line is the dots' mean energy. Bottom right: how many times more likely a weak dot (energy 5) is to sacrifice than a well-fed one (energy 19).*

## Why this happens

Selection does not favour the dots that sacrifice themselves. Each of them dies and loses. It favours the **gene** that makes them do it, because copies of that gene sit in the relatives who receive the energy.

This is described by Hamilton's rule. A gene for helping spreads when

```
r · B > C
```

*r: relatedness of giver and recipient · B: offspring the recipient gains · C: offspring the giver loses*

| Symbol | In this simulation |
|---|---|
| **C** | The chance that the giver would still have reproduced itself. For a weak dot in winter it is small, since it probably won't live to see spring anyway. For a well-fed one it is large. That is why the weak ones sacrifice themselves. |
| **B** | The recipient gets several units of energy: a good part of the way to reproducing, or a reserve to get through the winter. |
| **r** | *Gift to a neighbour:* neighbours are close relatives; the measured genetic relatedness is about 0.75. *Gift to a stranger:* r ≈ 0, so the left side of the inequality vanishes and only the cost remains. |

The two worlds differ only in the value of r. The fact that the outcome goes in opposite directions is a direct test of this inequality. Nobody here acts "for the good of the species". Selection acts on genes, and altruism is a by-product.

## What else you can see

### The population stabilises

In the control world the population falls into a two-year rhythm: a year of plenty, then a crash, and in winter it often drops almost to zero. In the gift-to-a-neighbour world that rhythm disappears and every year looks the same. Nobody designed this.

![Population size step by step. Top: the full run. The yellow line of the gift-to-a-neighbour world has regular waves; the cyan control line is irregular. Bottom: a close-up. The three runs of the gift-to-a-neighbour world almost overlap, while in the control each run behaves differently.](docs/demo_population.png)

*Population size at every step. Each wave is one year (up in summer, down in winter). The bottom panels zoom in on the last ten years of three runs.*

### Families stay put

Each dot also has an inherited colour that changes very slowly. The dots cannot see it: it is there for us to track families. In the gift-to-a-neighbour world many families live side by side in a mosaic and survive the winter on their own patch. In the control, winter crashes wipe out most families, and each year a few survivors flood the emptied map. In numbers: a dot lives on average 11.6 units from where its ancestor lived a year earlier. In the control it is 21.1.

![Live view after 25,000 steps. Left: the gift-to-a-neighbour world, evenly covered with yellowish dots, its hue-over-time strip showing many families side by side. Right: the control world, blue dots in a few blobs surrounded by uneaten grass, its hue strip showing families repeatedly wiped out. Below: the share of sacrifices and the population for both worlds.](docs/demo_compare.png)

*The live viewer after 25,000 steps. A dot's colour is its current urge to sacrifice: blue almost never, yellow often. The narrow side panels show the family colours: their distribution now, and how it changes over time (time runs downward).*

## What this model does not show

- Reproduction is asexual, so relatives are almost clones. With sexual reproduction relatedness is lower and the effect would be weaker.
- The world's parameters were chosen so that the effect is clear. With milder winters or faster-moving dots it weakens (to 40–60% of deaths), but it still differs from the control. That is why a control that changes only one rule matters more than the number itself.
- The first winters are a bottleneck. Before sacrifice spreads, the population can drop to a few dozen dots, and with an unlucky random seed it can die out.
- Relatedness is measured directly. Benefit and cost in units of offspring (B and C) are described qualitatively here, not computed.

## How to run it

Python version (pygame), the two worlds side by side, from the repository root:

```bash
./demo.sh
```

Evolife app version (a separate port, not in this repository; path on the author's machine):

```bash
cd ~/Desktop/Evolife/Apps/KinSacrifice && ../../.venv/bin/python KinSacrifice.py
```

---

How to install, the viewer's keys and the scripts behind every figure are in the [README](README.md). Older working notes with all the runs and the variants that did not work are in [docs/DEMO.md](docs/DEMO.md) and [docs/PLAN.md](docs/PLAN.md) (in Polish).
