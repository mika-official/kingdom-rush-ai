"""v9 route facts, sampled coverage and shared layout; no combat prediction."""
import numpy as np
from light_features import LightField, FEATURE_NAMES as BASE_FEATURES, SCENE_NAMES as BASE_SCENE, mapping
from candidate_features import as_list

ROUTES = tuple(str(i) for i in range(1, 17)) + ('other',)
HORIZONS = ('current', 'next', 'second')
COMPOSITION = ('count', 'hp_total', 'armor', 'magic_armor', 'flying', 'regen_rate',
               'boss', 'summon', 'replication', 'death_spawn', 'instakill_immune', 'root_immune')
CHANNELS = ('physical_dps', 'magic_dps', 'block')
SEGMENTS = ('front', 'middle', 'rear')
FEATURE_NAMES = BASE_FEATURES + ('coverage_known',) + tuple('coverage_'+s for s in SEGMENTS) + tuple('covers_'+r for r in ROUTES)
SCENE_NAMES = BASE_SCENE + tuple(f'path_{r}_{h}_{n}' for r in ROUTES for h in HORIZONS for n in COMPOSITION)
SCENE_NAMES += tuple(f'path_{r}_{s}_{n}' for r in ROUTES for s in SEGMENTS for n in CHANNELS)
FEATURE_DIM = len(FEATURE_NAMES)
SCENE_DIM = len(SCENE_NAMES)


def route_key(value):
    key = str(value)
    return key if key in ROUTES[:-1] else 'other'


class LayoutField(LightField):
    def __init__(self, env, observation):
        super().__init__(env, observation)
        # Bucket once per observation, never once per candidate.
        groups = {h: {r: [] for r in ROUTES} for h in HORIZONS}
        for h in HORIZONS:
            for enemy in self.horizons[h]:
                groups[h][route_key(enemy.get('pi'))].append(enemy)
        self.route_values = []
        for r in ROUTES:
            for h in HORIZONS:
                # summary uses current to distinguish entities from spawn batches.
                original = self.horizons[h]
                self.horizons[h] = groups[h][r]
                # Unique cache keys avoid mixing global and route summaries.
                cache = self._summaries
                self._summaries = {}
                values = self.summary(h)
                self._summaries = cache
                self.horizons[h] = original
                self.route_values.extend(values[n] for n in COMPOSITION)
        defense = {r: np.zeros((3, 3), np.float32) for r in ROUTES}
        for tower, facts in self._tower_facts:
            if tower.get('blocked'):
                continue
            for r, bins in mapping(tower.get('defense_paths')).items():
                for segment, count in enumerate(as_list(bins)[:3]):
                    if float(count or 0) > 0:
                        defense[route_key(r)][segment] += [facts[n] for n in CHANNELS]
        self.route_values.extend(float(v) for r in ROUTES for v in defense[r].flat)

    def scene(self):
        return np.concatenate((super().scene(), np.asarray(self.route_values, np.float32)))

    def vector(self, candidate):
        base = super().vector(candidate)
        spot = candidate.get('spot') or {}
        raw = candidate.get('target_paths')
        if raw is None:
            raw = spot.get('defense_paths')
        paths = mapping(raw)
        bins = np.zeros(3, np.float32)
        memberships = np.zeros(len(ROUTES), np.float32)
        for r, samples in paths.items():
            memberships[ROUTES.index(route_key(r))] = 1
            for i, count in enumerate(as_list(samples)[:3]):
                bins[i] += float(count or 0)/100
        return np.concatenate((base, np.asarray([raw is not None], np.float32), bins, memberships))
