""" @brief  The world of the KinSacrifice application (pure NumPy, no graphics).

Dots live on a torus covered with grass. Grass regrows (slowly in winter).
Dots walk slowly towards grass, eat it, pay a metabolic cost, die of hunger or age,
and reproduce asexually when they have enough energy: the child appears next to
its parent and inherits its genes with small mutations.

One extra possibility: at each step a dot meets a random neighbour and may give it
ALL its energy and die (self-sacrifice). The probability is decided by two genes,
a and b:   p = sigmoid(a + b * energy / BirthThreshold).
Dots do not recognise their relatives.

In the control world ('GiftToStranger'), the energy of a sacrificing dot goes to a
random dot anywhere in the world instead of the neighbour it met.

This module is a NumPy port of the 'kin-dots' demo (demo_sacrifice preset).
"""

#============================================================================#
# Written for EVOLIFE  http://evolife.telecom-paris.fr                       #
#============================================================================#

import numpy as np

# The 8 neighbouring grass cells and the unit vector pointing to each
_OFFSETS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])
_UNIT = _OFFSETS / np.linalg.norm(_OFFSETS, axis=1, keepdims=True)
_BUCKET_OFFSETS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])

DEFAULTS = dict(
	WorldSize=300.0,		# side of the torus
	GrassCells=75,			# grass grid is GrassCells x GrassCells
	GrassCapacity=10.0,		# max grass per cell
	GrassGrowth=0.05,		# logistic regrowth rate per step
	GrassInflow=0.1,		# constant regrowth per cell per step
	InitialGrass=0.5,		# initial grass, as a fraction of capacity
	SeasonLength=600,		# steps per year
	WinterHarshness=0.9,	# regrowth is multiplied by 1 + harshness * sin(2 pi t / SeasonLength)
	InitialPopulation=2500,
	InitialEnergy=10.0,
	Speed=0.1,				# distance walked per step
	MoveCost=0.05,			# energy per unit of distance
	Metabolism=0.1,			# energy per step
	MaxAge=300,				# death by old age
	Bite=0.5,				# fraction of the grass of its cell a dot eats per step
	BirthThreshold=20.0,	# energy needed to reproduce
	BirthCost=1.0,			# energy lost at each birth
	ChildSpread=2.0,		# std of the child's distance from its parent
	MeetRadius=4.0,			# neighbours within this distance can be met
	GiftEfficiency=0.9,		# share of the donor's energy the recipient gets
	InitialA=-9.0,			# initial value of gene a (sacrifice is very rare at start)
	InitialB=0.0,			# initial value of gene b
	InitialFoodSeeking=1.0,	# initial value of the gene 'how strongly a dot heads towards grass'
	FounderNoise=0.5,		# std of the founders' gene values around the initial ones
	MutationSD=0.15,		# std of the mutation of a, b and food seeking at each birth
	TagMutationSD=0.02,		# std of the mutation of the family colour (RGB in [0, 1])
	GiftToStranger=False,	# control: the gift goes to a random dot anywhere
)


class World:
	"""	State as arrays (one entry per living dot) and the update rule.
	"""

	def __init__(self, seed=0, **params):
		self.P = dict(DEFAULTS)
		unknown = set(params) - set(DEFAULTS)
		if unknown:	raise ValueError(f'unknown parameter(s): {sorted(unknown)}')
		self.P.update(params)
		P = self.P
		self.rng = np.random.default_rng(seed)
		n = int(P['InitialPopulation'])
		W = P['WorldSize']
		self.x = self.rng.random(n) * W
		self.y = self.rng.random(n) * W
		self.energy = np.full(n, float(P['InitialEnergy']))
		self.age = np.zeros(n, dtype=np.int64)
		self.a = P['InitialA'] + self.rng.normal(0, P['FounderNoise'], n)
		self.b = P['InitialB'] + self.rng.normal(0, P['FounderNoise'], n)
		self.food_seeking = P['InitialFoodSeeking'] + self.rng.normal(0, P['FounderNoise'], n)
		self.tag = self.rng.random((n, 3))
		self.id = np.arange(n)
		self.next_id = n
		g = int(P['GrassCells'])
		self.grass = np.full((g, g), P['GrassCapacity'] * P['InitialGrass'])
		self.step_count = 0
		self.nb = max(3, int(W // P['MeetRadius']))	# neighbour buckets per side
		# statistics of the last step
		self.sacrifices = 0
		self.deaths = 0				# hunger + old age
		self.last_gifts = []		# (x_from, y_from, x_to, y_to) of the last step's gifts

	# ------------------------------------------------------------------ helpers

	@property
	def population(self):	return self.x.size

	def season_factor(self):
		P = self.P
		if P['SeasonLength'] <= 0:	return 1.0
		return max(0.0, 1.0 + P['WinterHarshness'] * np.sin(2 * np.pi * self.step_count / P['SeasonLength']))

	def cells(self):
		g = int(self.P['GrassCells'])
		cs = self.P['WorldSize'] / g
		return (self.x // cs).astype(np.int64) % g, (self.y // cs).astype(np.int64) % g

	def propensity(self):
		"""	per-step probability that each dot sacrifices itself if it meets someone """
		e = np.minimum(self.energy / self.P['BirthThreshold'], 2.0)
		return 1.0 / (1.0 + np.exp(-(self.a + self.b * e)))

	def _keep(self, mask):
		for name in ('x', 'y', 'energy', 'age', 'a', 'b', 'food_seeking', 'tag', 'id'):
			setattr(self, name, getattr(self, name)[mask])

	# ------------------------------------------------------------------ update

	def step(self):
		self.sacrifices = 0
		self.deaths = 0
		self.last_gifts = []
		if self.population:
			self._sacrifice()
			self._move()
			self._eat()
			self._metabolism_and_death()
			self._reproduce()
			self.age += 1
		self._regrow()
		self.step_count += 1

	def _random_neighbour(self, tries=8):
		"""	For each dot, a random neighbour within MeetRadius (or -1).
			The world is cut into buckets of side >= MeetRadius. A dot picks a random dot
			among those in the 3x3 buckets around it, and keeps it if it is within
			MeetRadius (up to `tries` attempts).
		"""
		P, rng, n, nb = self.P, self.rng, self.population, self.nb
		W = P['WorldSize']
		bs = W / nb
		bx = (self.x // bs).astype(np.int64) % nb
		by = (self.y // bs).astype(np.int64) % nb
		bucket = bx * nb + by
		order = np.argsort(bucket, kind='stable')
		counts = np.bincount(bucket, minlength=nb * nb)
		start = np.concatenate([[0], np.cumsum(counts)[:-1]])
		around = ((bx[:, None] + _BUCKET_OFFSETS[:, 0]) % nb) * nb + (by[:, None] + _BUCKET_OFFSETS[:, 1]) % nb
		c9 = counts[around]						# (n, 9) dots in each surrounding bucket
		cum = np.cumsum(c9, axis=1)
		total = cum[:, -1]
		result = np.full(n, -1)
		todo = np.flatnonzero(total > 1)
		r2 = P['MeetRadius'] ** 2
		for _ in range(tries):
			if todo.size == 0:	break
			r = (rng.random(todo.size) * total[todo]).astype(np.int64)
			k = (cum[todo] <= r[:, None]).sum(axis=1)		# which of the 9 buckets
			before = np.where(k > 0, cum[todo, np.maximum(k - 1, 0)], 0)
			j = order[start[around[todo, k]] + r - before]
			dx = np.abs(self.x[todo] - self.x[j]); dx = np.minimum(dx, W - dx)
			dy = np.abs(self.y[todo] - self.y[j]); dy = np.minimum(dy, W - dy)
			ok = (j != todo) & (dx * dx + dy * dy <= r2)
			result[todo[ok]] = j[ok]
			todo = todo[~ok]
		return result

	def _sacrifice(self):
		P, rng, n = self.P, self.rng, self.population
		partner = self._random_neighbour()
		want = (partner >= 0) & (self.energy > 0) & (rng.random(n) < self.propensity())
		donors = np.flatnonzero(want)
		if donors.size == 0:	return
		rng.shuffle(donors)
		dead = np.zeros(n, dtype=bool)
		for i in donors:
			if dead[i]:	continue
			if P['GiftToStranger']:
				j = int(rng.integers(n))
			else:
				j = int(partner[i])
			if j == i or dead[j]:	continue
			self.energy[j] += P['GiftEfficiency'] * self.energy[i]
			self.last_gifts.append((self.x[i], self.y[i], self.x[j], self.y[j]))
			self.energy[i] = 0.0
			dead[i] = True
		self.sacrifices = int(dead.sum())
		self._keep(~dead)

	def _move(self):
		P, g = self.P, int(self.P['GrassCells'])
		cx, cy = self.cells()
		own = self.grass[cx, cy]
		grad = np.zeros((self.population, 2))
		for (dx, dy), u in zip(_OFFSETS, _UNIT):
			grad += (self.grass[(cx + dx) % g, (cy + dy) % g] - own)[:, None] * u
		d = self.food_seeking[:, None] * grad + self.rng.normal(size=(self.population, 2))
		norm = np.linalg.norm(d, axis=1, keepdims=True)
		d = np.divide(d, norm, out=np.zeros_like(d), where=norm > 0)
		W = P['WorldSize']
		self.x = (self.x + d[:, 0] * P['Speed']) % W
		self.y = (self.y + d[:, 1] * P['Speed']) % W
		self.energy -= P['MoveCost'] * P['Speed']

	def _eat(self):
		g = int(self.P['GrassCells'])
		cx, cy = self.cells()
		cell = cx * g + cy
		k = np.bincount(cell, minlength=g * g)
		grass = self.grass.reshape(-1)
		eaten = grass * (1.0 - (1.0 - self.P['Bite']) ** k)	# like eating one after another
		with np.errstate(invalid='ignore', divide='ignore'):
			share = np.where(k > 0, eaten / np.maximum(k, 1), 0.0)
		grass -= eaten
		self.energy += share[cell]

	def _metabolism_and_death(self):
		self.energy -= self.P['Metabolism']
		alive = (self.energy > 0) & (self.age < self.P['MaxAge'])
		self.deaths = int((~alive).sum())
		self._keep(alive)

	def _reproduce(self):
		P, rng = self.P, self.rng
		parents = np.flatnonzero(self.energy >= P['BirthThreshold'])
		k = parents.size
		if k == 0:	return
		rest = self.energy[parents] - P['BirthCost']
		self.energy[parents] = rest / 2
		W = P['WorldSize']
		clip = lambda v: np.clip(v, -10.0, 10.0)
		new = dict(
			x=(self.x[parents] + rng.normal(0, P['ChildSpread'], k)) % W,
			y=(self.y[parents] + rng.normal(0, P['ChildSpread'], k)) % W,
			energy=rest / 2,
			age=np.zeros(k, dtype=np.int64),
			a=clip(self.a[parents] + rng.normal(0, P['MutationSD'], k)),
			b=clip(self.b[parents] + rng.normal(0, P['MutationSD'], k)),
			food_seeking=clip(self.food_seeking[parents] + rng.normal(0, P['MutationSD'], k)),
			tag=np.clip(self.tag[parents] + rng.normal(0, P['TagMutationSD'], (k, 3)), 0.0, 1.0),
			id=np.arange(self.next_id, self.next_id + k),
		)
		self.next_id += k
		for name, v in new.items():
			setattr(self, name, np.concatenate([getattr(self, name), v]))

	def _regrow(self):
		P = self.P
		f = self.grass
		f += self.season_factor() * (P['GrassGrowth'] * f * (1.0 - f / P['GrassCapacity']) + P['GrassInflow'])
		np.clip(f, 0.0, P['GrassCapacity'], out=f)


if __name__ == '__main__':
	# Headless check: both worlds side by side, share of deaths that are sacrifices per 10 years
	import sys
	steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
	seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
	harsh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
	worlds = {'neighbour': World(seed, WinterHarshness=harsh), 'stranger': World(seed, GiftToStranger=True, WinterHarshness=harsh)}
	acc = {k: [0, 0] for k in worlds}
	for t in range(1, steps + 1):
		for name, w in worlds.items():
			if w.population:
				w.step()
				acc[name][0] += w.sacrifices
				acc[name][1] += w.sacrifices + w.deaths
		if t % 6000 == 0:
			print(f'step {t:6d}  ' + '   '.join(
				f'{name}: pop {w.population:5d}  sacrifice {100 * acc[name][0] / max(acc[name][1], 1):4.0f}% of deaths'
				for name, w in worlds.items()), flush=True)
			acc = {k: [0, 0] for k in worlds}
