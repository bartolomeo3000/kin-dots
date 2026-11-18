#!/usr/bin/env python3
""" @brief  Kin sacrifice:
Dots may give ALL their energy to a neighbour and die. Nothing tells them to help relatives,
and they cannot recognise relatives. Yet self-sacrifice spreads, but only when the gift
goes to a neighbour (usually a relative), not when it goes to a random stranger.
"""

#============================================================================#
# Written for EVOLIFE  http://evolife.telecom-paris.fr                       #
# (application ported from the 'kin-dots' demo)                              #
#============================================================================#


##############################################################################
# Kin Sacrifice                                                              #
##############################################################################

# Two identical worlds are simulated side by side.
#	Left:	a sacrificing dot gives its energy to the neighbour it met (usually a relative)
#	Right:	(control) the energy goes to a random dot anywhere in the world (a stranger)
# Everything else is the same, including the random seed.
#
# Each dot carries two genes a and b. At each step it meets a random neighbour and,
# with probability  sigmoid(a + b * energy/BirthThreshold),  gives it all its energy and dies.
# At start, a = -9: sacrifice is very rare.
#
# Watch the curves: the share of deaths that are self-sacrifice rises to ~70% on the left
# and stays low on the right.
#
# The 'Field' window shows both worlds (grass in green, dots coloured according to the
# colour mode, white lines = gifts of the last step).
# Keys in the main window:	1 = colour by urge to sacrifice		2 = colour by family
#							3 = colour by energy				0 = show / hide grass
# The 'Families (hue)' window (key T) shows the distribution of the inherited family colours:
# a histogram now, and its history over the whole run (time goes down).


import sys
import os
import random
from collections import deque
from time import sleep

sys.path.append('..')
sys.path.append('../../..')
import Evolife.Scenarii.Parameters			as EPar
import Evolife.Ecology.Observer				as EO
import Evolife.Graphics.Evolife_Window	as EW

import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SacrificeWorld import World, DEFAULTS

LABELS = ('gift to a neighbour', 'gift to a stranger')
CURVE_COLOURS = ('red', 'blue')			# Evolife colour names (curves need named colours)
POP_COLOURS = ('red8', 'blue8')
TEXT_COLOURS = ((255, 110, 90), (120, 160, 255))	# same hues for the labels in the view
COLOUR_MODES = {'1': 'propensity', '2': 'family', '3': 'energy'}
MODE_TEXT = {'propensity': 'urge to sacrifice (blue = never, yellow = often)',
			 'family': 'family (inherited colour)',
			 'energy': 'energy (blue = hungry, yellow = ready to reproduce)'}
PANEL = 380			# pixels per world in the view
MARGIN, TOP = 12, 64
FIELD_TOP = 82		# header height in the Field view (one more line: last year's population)
BG = (18, 18, 22)
# 'Families (hue)' window: distribution of the inherited family colours, now and over time
HUE_BINS = 60
HUE_EVERY = 100		# steps between two rows of the hue history
HUE_W, HIST_H, STRIP_H = 300, 80, 380
HIST_MAX = 0.15		# share of dots that fills a histogram bar


def ramp(t):
	"""	values in [0, 1] -> (n, 3) uint8 colours from blue to yellow """
	t = np.clip(t, 0.0, 1.0)[:, None]
	lo, hi = np.array([40, 90, 220]), np.array([250, 210, 40])
	return (lo + (hi - lo) * t).astype(np.uint8)


def hue_of(rgb):
	"""	hue in [0, 1) of (n, 3) colours in [0, 1] """
	r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
	mx, mn = rgb.max(axis=1), rgb.min(axis=1)
	d = np.where(mx > mn, mx - mn, 1.0)
	h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
	return np.where(mx > mn, h / 6.0, 0.0)


def hsv_colours(h, s=0.85, v=1.0):
	"""	(n,) hues -> (n, 3) float RGB in [0, 255] """
	i = np.floor(h * 6).astype(int) % 6
	f = h * 6 - np.floor(h * 6)
	p, q, t, V = v * (1 - s) + 0 * h, v * (1 - s * f), v * (1 - s * (1 - f)), v + 0 * h
	table = np.stack([np.stack(c, axis=1) for c in
					  ((V, t, p), (q, V, p), (p, V, t), (p, q, V), (t, p, V), (V, p, q))])
	return table[i, np.arange(h.size)] * 255


HUE_RGB = hsv_colours((np.arange(HUE_BINS) + 0.5) / HUE_BINS)


def hue_hist(w):
	"""	share of living dots in each family-hue bin """
	if w.population == 0:	return np.zeros(HUE_BINS)
	h = np.bincount(np.minimum((hue_of(w.tag) * HUE_BINS).astype(int), HUE_BINS - 1), minlength=HUE_BINS)
	return h / h.sum()


def _font(size):
	for f in ('/System/Library/Fonts/Supplemental/Arial.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
		try:	return ImageFont.truetype(f, size)
		except OSError:	pass
	return ImageFont.load_default()


class Observer(EO.Observer):
	"""	Keeps the colour mode, changed with keys 1/2/3 """
	def __init__(self, Gbl):
		EO.Observer.__init__(self, Gbl)
		self.ColourMode = str(Gbl.get('ColourMode', 'propensity'))
		self.ShowGrass = bool(Gbl.get('ShowGrass', 1))

	def inform(self, Info):
		if Info == '0':
			self.ShowGrass = not self.ShowGrass
			print(f"grass {'shown' if self.ShowGrass else 'hidden'}")
		elif Info in COLOUR_MODES:
			self.ColourMode = COLOUR_MODES[Info]
			print(f'colour mode: {self.ColourMode}')


class Simulation:
	def __init__(self, Gbl, Obs):
		self.Gbl, self.Obs = Gbl, Obs
		params = {k: Gbl[k] for k in DEFAULTS if Gbl.get(k, None) is not None}
		params.pop('GiftToStranger', None)
		seed = int(Gbl.get('RandomSeed', 0)) or random.randrange(1, 10 ** 6)	# 0 = random (Evolife convention)
		print(f'random seed: {seed}')
		self.worlds = [World(seed, **params)]
		if Gbl.get('ShowControl', 1):
			self.worlds.append(World(seed, GiftToStranger=True, **params))
		year = max(int(self.worlds[0].P['SeasonLength']), 1)
		# sacrifices, deaths and population at each step over the last year, for each world
		self.history = [deque(maxlen=year) for _ in self.worlds]
		self.sums = [[0, 0] for _ in self.worlds]
		self.hue_rows = [[] for _ in self.worlds]		# hue histogram every HUE_EVERY steps
		self.ImageFile = os.path.join(Obs.getInfo('OutputDir', '.'), '___KinSacrifice_view.png')
		self.HueFile = os.path.join(Obs.getInfo('OutputDir', '.'), '___KinSacrifice_hue.png')
		self.font, self.small = _font(17), _font(13)

	def One_Decision(self):
		"""	Called repeatedly by Evolife: one simulation step of both worlds """
		for k, w in enumerate(self.worlds):
			if w.population == 0:	continue
			w.step()
			h, s = self.history[k], self.sums[k]
			if len(h) == h.maxlen:
				old = h[0]
				s[0] -= old[0]; s[1] -= old[1]
			h.append((w.sacrifices, w.sacrifices + w.deaths, w.population))
			s[0] += w.sacrifices; s[1] += w.sacrifices + w.deaths
			if w.step_count % HUE_EVERY == 0:	self.hue_rows[k].append(hue_hist(w))
		self.Obs.season()	# StepId = simulation step
		if self.Obs.Visible():	self.display()
		return any(w.population for w in self.worlds)	# stops when every world is extinct

	def sacrifice_share(self, k):
		sac, died = self.sums[k]
		return 100.0 * sac / died if died else 0.0

	# ---------------------------------------------------------------- display

	def colours(self, w):
		mode = self.Obs.ColourMode
		if mode == 'family':	return (40 + 215 * w.tag).astype(np.uint8)
		if mode == 'energy':	return ramp(w.energy / w.P['BirthThreshold'])
		# urge to sacrifice right now (log scale: 1e-4 -> blue, 1e-1 -> yellow)
		return ramp((np.log10(np.maximum(w.propensity(), 1e-6)) + 4.0) / 3.0)

	def panel(self, w):
		"""	one world as a PANEL x PANEL image: grass, dots, gifts of the last step """
		P = w.P
		g = int(P['GrassCells'])
		cell = np.arange(PANEL) * g // PANEL
		if self.Obs.ShowGrass:
			t = np.sqrt(w.grass / P['GrassCapacity'])[..., None]
			rgb = (np.array(BG) * (1 - t) + np.array([30, 95, 45]) * t).astype(np.uint8)
			canvas = rgb[cell][:, cell].copy()		# indexed [x, y]
		else:
			canvas = np.full((PANEL, PANEL, 3), 8, dtype=np.uint8)
		s = PANEL / P['WorldSize']
		if w.population:
			px = np.minimum((w.x * s).astype(int), PANEL - 2)
			py = np.minimum((w.y * s).astype(int), PANEL - 2)
			cols = self.colours(w)
			for dx in (0, 1):
				for dy in (0, 1):
					canvas[px + dx, py + dy] = cols
		img = Image.fromarray(np.ascontiguousarray(canvas.transpose(1, 0, 2)))		# to [y, x]
		d = ImageDraw.Draw(img)
		half = P['WorldSize'] / 2
		for xa, ya, xb, yb in w.last_gifts:
			if abs(xa - xb) < half and abs(ya - yb) < half:		# skip gifts across the torus edge
				d.line([(xa * s, ya * s), (xb * s, yb * s)], fill=(255, 255, 255), width=1)
		return img

	def display(self):
		n = len(self.worlds)
		W_ = MARGIN + n * (PANEL + MARGIN)
		H_ = FIELD_TOP + PANEL + MARGIN + 20
		img = Image.new('RGB', (W_, H_), BG)
		d = ImageDraw.Draw(img)
		w0 = self.worlds[0]
		season = 'summer' if w0.season_factor() >= 1 else 'winter'
		for k, w in enumerate(self.worlds):
			x0 = MARGIN + k * (PANEL + MARGIN)
			d.text((x0, 8), LABELS[k], font=self.font, fill=TEXT_COLOURS[k])
			d.text((x0, 30), f'population {w.population}', font=self.small, fill=(210, 210, 215))
			pops = [e[2] for e in self.history[k]]
			if pops:
				d.text((x0, 46), f'last year: mean {sum(pops) / len(pops):.0f}   min {min(pops)}   max {max(pops)}',
					   font=self.small, fill=(160, 160, 170))
			d.text((x0, 62), f'sacrifice: {self.sacrifice_share(k):.0f}% of deaths (last year)',
				   font=self.small, fill=TEXT_COLOURS[k])
			img.paste(self.panel(w), (x0, FIELD_TOP))
		d.text((MARGIN, FIELD_TOP + PANEL + 6),
			   f'step {w0.step_count}  ({season})   colour: {MODE_TEXT[self.Obs.ColourMode]}   [keys 1 2 3, 0 = grass]',
			   font=self.small, fill=(150, 150, 160))
		img.save(self.ImageFile, compress_level=1)
		self.Obs.recordInfo('Image', self.ImageFile)
		self.hue_display()
		# curves
		for k, w in enumerate(self.worlds):
			self.Obs.curve(Name=f'{LABELS[k]}: sacrifice', Value=self.sacrifice_share(k))
			self.Obs.curve(Name=f'{LABELS[k]}: population', Value=w.population / 100.0)


	def hue_display(self):
		"""	'Families (hue)' window: for each world, the hue histogram of the family colours now
			and its history over the whole run (time goes down) """
		n = len(self.worlds)
		W_ = MARGIN + n * (HUE_W + MARGIN)
		strip_y = TOP + HIST_H + 30
		H_ = strip_y + STRIP_H + MARGIN + 20
		img = Image.new('RGB', (W_, H_), BG)
		d = ImageDraw.Draw(img)
		grey = (150, 150, 160)
		for k, w in enumerate(self.worlds):
			x0 = MARGIN + k * (HUE_W + MARGIN)
			d.text((x0, 8), LABELS[k], font=self.font, fill=TEXT_COLOURS[k])
			d.text((x0, TOP - 18), 'hue now', font=self.small, fill=grey)
			d.rectangle([x0, TOP, x0 + HUE_W - 1, TOP + HIST_H - 1], fill=(26, 26, 32))
			bw = HUE_W / HUE_BINS
			for b, share in enumerate(hue_hist(w)):
				bh = int(min(share / HIST_MAX, 1.0) * (HIST_H - 2))
				if bh > 0:
					d.rectangle([x0 + int(b * bw), TOP + HIST_H - bh, x0 + int((b + 1) * bw) - 1, TOP + HIST_H - 1],
								fill=tuple(HUE_RGB[b].astype(int)))
			d.text((x0, strip_y - 18), 'hue over time', font=self.small, fill=grey)
			rows = self.hue_rows[k]
			if len(rows) >= 2:
				r = np.asarray(rows)									# (time, bins)
				bright = np.minimum(np.sqrt(r / HIST_MAX), 1.0)[..., None]
				rgb = (26 + (HUE_RGB[None] - 26) * bright).astype(np.uint8)
				strip = Image.fromarray(np.ascontiguousarray(rgb)).resize((HUE_W, STRIP_H), Image.NEAREST)
				img.paste(strip, (x0, strip_y))
				d.text((x0 + 4, strip_y + STRIP_H - 18), f'step 0 .. {len(rows) * HUE_EVERY}', font=self.small, fill=(220, 220, 225))
			else:
				d.rectangle([x0, strip_y, x0 + HUE_W - 1, strip_y + STRIP_H - 1], fill=(26, 26, 32))
		d.text((MARGIN, strip_y + STRIP_H + 6),
			   'each family has its own inherited colour: a band = a family; time runs downward',
			   font=self.small, fill=grey)
		img.save(self.HueFile, compress_level=1)
		self.Obs.recordInfo('Pattern', self.HueFile)


def declare_curves(Obs, Sim):
	for k, w in enumerate(Sim.worlds):
		Obs.curve(Name=f'{LABELS[k]}: sacrifice', Color=CURVE_COLOURS[k], Thickness=3,
				  Legend=f'{LABELS[k]}: % of deaths over the last year that are self-sacrifice')
		Obs.curve(Name=f'{LABELS[k]}: population', Color=POP_COLOURS[k], Thickness=1,
				  Legend=f'{LABELS[k]}: population size (in hundreds)')


if __name__ == "__main__":
	print(__doc__)

	#############################
	# Global objects			#
	#############################
	CfgFile = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.evo') else '_Params.evo'
	Gbl = EPar.Parameters(CfgFile)	# Loading global parameter values
	Obs = Observer(Gbl)				# Observer contains statistics
	Sim = Simulation(Gbl, Obs)
	Obs.recordInfo('DefaultViews', ['Field', 'Trajectories'])
	Obs.recordInfo('TrajectoriesTitle', 'Families (hue)')
	declare_curves(Obs, Sim)

	EW.Start(Sim.One_Decision, Obs, Capabilities='FCPT', Options={'Run': Gbl.get('Run', False)})

	print("Bye.......")
	sleep(1.0)

__author__ = 'kin-dots port'
