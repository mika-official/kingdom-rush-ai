"""Versioned tactical macros: enemy matchups, next wave and selectable sites."""
from collections import Counter
import math
import numpy as np
from gymnasium import spaces
from kr_macro import KRMacroEnv, MACROS, N_FEATURES, KINDS, ENEMY_TYPES, _opt_ok

TACTICAL_MACROS = MACROS + tuple(f"build_{kind}_site{rank}" for rank in (2, 3) for kind in KINDS)
SITE_FEATURES = 12
PROFILE_FEATURES = 10


def enemy_profile(enemies, forecast=False):
    """Damage retention is a prior, not DPS: ignores upgrades, specials and overkill."""
    total = sum(float(e.get("count", 0) if forecast else 1) for e in enemies)
    out = np.zeros(PROFILE_FEATURES, dtype=np.float32)
    out[0] = np.log1p(total) / np.log1p(300)
    for e in enemies:
        weight = float(e.get("count", 0) if forecast else 1) / max(1, total)
        flying = float(bool(e.get("flying")))
        armor = np.clip(float(e.get("armor") or 0), 0, 1)
        magic = np.clip(float(e.get("magic_armor") or 0), 0, 1)
        # Relative spawn density only; does not predict exact on-map clustering.
        density = min(1, 30 / max(1, float(e.get("interval") or 1))) if forecast else 0
        out[1:] += weight * np.array([flying, armor, magic, float(armor >= .5),
            1-armor, (1-flying)*(1-armor), 1-magic, (1-flying)*(1-armor/2), density])
    if not forecast and enemies:
        # Ground-enemy neighbours within 60 px, capped to avoid outliers.
        ground = [e for e in enemies if not e.get("flying") and "x" in e and "y" in e]
        out[-1] = max((min(1, sum((e['x']-p['x'])**2+(e['y']-p['y'])**2 <= 3600
                                 for p in ground) / 8) for e in ground), default=0)
    return out


def inside(x, y, spot, radius):
    # Game ranges are elliptical; this is a documented geometric approximation.
    return ((x-float(spot.get('x', 0)))**2 + ((y-float(spot.get('y', 0)))/.7)**2) <= radius**2


def support_at(point, spots):
    return sum(inside(point.get('x', 0), point.get('y', 0),
                      dict(x=tower.get('attack_x',tower.get('x',0)), y=tower.get('attack_y',tower.get('y',0))),
                      float(tower.get('attack_range') or tower.get('cov', {}).get(KRMacroEnv.tower_kind(tower), {}).get('r') or 0))
               for tower in spots if tower.get('state') == 'tower'
               and KRMacroEnv.tower_kind(tower) in ('archer', 'mage', 'engineer'))


class KRTacticalEnv(KRMacroEnv):
    schema = 'macro-v2-tactical'
    base_meta = len(TACTICAL_MACROS)
    n_actions = base_meta + 7

    def __init__(self, **kwargs):
        self.last_hero_move_tick = -10**9
        super().__init__(**kwargs)
        self.config['tactical'] = True
        self.observation_space = spaces.Dict({'summary': spaces.Box(-np.inf, np.inf,
            (N_FEATURES + 2*PROFILE_FEATURES + 1 + 12*SITE_FEATURES + len(ENEMY_TYPES)+1,), np.float32)})

    def _sites(self, o):
        cm = self._cmax(o)
        current = o.get('enemies') or []
        forecast = (o.get('forecast') or {}).get('spawns') or []
        paths = Counter()
        for e in forecast:
            paths[str(e.get('pi'))] += float(e.get('count') or 0)
        spots = o.get('spots') or []
        output = {}
        for k, kind in enumerate(KINDS):
            ranked = []
            for i, sp in enumerate(spots):
                if sp.get('state') != 'empty' or not self._build_allowed(o, i, kind, cm):
                    continue
                options = [op for op in sp.get('options') or [] if op.get('tpl') == 'tower_build_'+kind and not op.get('locked')]
                if not options:
                    continue
                cov = sp.get('cov', {}).get(kind, {})
                radius = float(cov.get('r') or 0)
                local = [e for e in current if inside(float(e.get('x', 0)), float(e.get('y', 0)), sp, radius)]
                route = cov.get('paths') or {}
                future = sum(paths.get(str(pi), 0) for pi, count in route.items() if count) / max(1, sum(paths.values()))
                profile = enemy_profile(local)
                nearby = support_at(sp, spots)
                quality = self._quality(sp, kind, cm)
                cost = float(options[0].get('price') or 0)
                descriptor = [float(sp.get('x') or 0)/1024, float(sp.get('y') or 0)/768,
                    quality, float(cov.get('p') or 0), future, profile[0], profile[1],
                    profile[2], profile[3], profile[-1], nearby/10, cost/1000]
                ranked.append((i, sp, descriptor))
            # Distinct choices expose coverage, route pressure and tower support.
            selected = []
            scorers = (lambda d: d[2], lambda d: d[4]+d[5]+d[2]*.1,
                       lambda d: d[10]+d[9]+d[2]*.1)
            for rank, scorer in enumerate(scorers):
                remaining = [r for r in ranked if r[0] not in {s[0] for s in selected}]
                if not remaining:
                    break
                item = max(remaining, key=lambda r: (scorer(r[2]), -r[0]))
                selected.append(item)
                action = 2+k if rank == 0 else 15+(rank-1)*4+k
                output[action] = item
        return output

    def _build_allowed(self, o, i, kind, cmax):
        # Quality ranks candidates and remains visible to the policy. It must
        # not permanently ban a legal site after the better sites are occupied.
        return not (self.freeze_last and self._final_wave(o))

    def _candidates(self, o):
        candidates = super()._candidates(o)
        for action in range(2, 6):
            candidates.pop(action, None)
        if o.get('outcome'):
            return candidates
        candidates.pop(13, None)
        hero = o.get('hero') or {}
        hero_interval = math.ceil(30 / float(self.info_env.get('tick_length') or (1/30)))
        if hero and not hero.get('dead') and self.t-self.last_hero_move_tick >= hero_interval:
            x,y = float(hero.get('x') or 0),float(hero.get('y') or 0)
            rally_distance = math.hypot(x-float(hero.get('rally_x',x)),y-float(hero.get('rally_y',y)))
            def score(point):
                nearby = [e for e in o.get('enemies') or []
                          if math.hypot(float(e.get('x') or 0)-point['x'],float(e.get('y') or 0)-point['y']) < 160]
                danger = sum(not e.get('flying') and math.hypot(float(e.get('x') or 0)-point['x'],float(e.get('y') or 0)-point['y']) < 45 for e in nearby)
                return 2*support_at(point,o.get('spots') or []) + min(4,len(nearby)) - 2*danger
            current_score = score(dict(x=x,y=y))
            best = None
            if rally_distance < 20:
                for index,target in enumerate(o.get('targets') or []):
                    legal = hero.get('move_ok') or []
                    if index >= len(legal) or not legal[index] or target.get('x') is None:
                        continue
                    distance=math.hypot(target['x']-x,target['y']-y)
                    benefit=score(target)-current_score
                    if distance < 60 or benefit < 1.5:
                        continue
                    rank=benefit-distance/1000
                    if best is None or rank>best[0]:
                        best=(rank,index)
            if best:
                candidates[13]=([('hero_move',best[1]+1)],('hero_move',None,{}),best[0])
        for action, (i, sp, descriptor) in self._sites(o).items():
            kind = KINDS[(action-2) if action < 6 else (action-15)%4]
            if _opt_ok(sp, 'tower_build_'+kind):
                candidates[action] = ([('build', sp['hid'], kind)], ('build', sp, {'kind':kind,'index':i}), descriptor[2])
        # Lua exposes the best supported legal blocking point for each barracks.
        candidates.pop(14, None)
        best = None
        for i, sp in enumerate(o.get('spots') or []):
            if self.rally_used_wave.get(i) == int(o.get('wave') or 0):
                continue
            for k, point in enumerate(sp.get('rally_points') or []):
                if not point or not (sp.get('rally_ok') or [])[k]:
                    continue
                score = support_at(point, o['spots'])
                distance = math.hypot(point['x']-float(sp.get('rally_x') or 0), point['y']-float(sp.get('rally_y') or 0))
                if distance < 5:
                    continue
                if best is None or score > best[0]:
                    best = (score, i, sp, k)
        if best:
            score, i, sp, k = best
            candidates[14] = ([('rally',sp['hid'],k+1)],('rally',sp,{'index':i}),score)
        if self.spend_before_wait and any(self.action_kind(a) in ('build','upgrade') for a in candidates):
            candidates.pop(0, None)
        return candidates

    def action_kind(self, action):
        if 15 <= action < 23:
            return 'build'
        return super().action_kind(action)

    def _start_play(self):
        self.last_hero_move_tick = -10**9
        return super()._start_play()

    def step(self, action):
        result = super().step(action)
        if action == 13 and result[4].get('agent_action_ok'):
            self.last_hero_move_tick = self.t
        return result

    def _vec(self, o):
        if 'forecast' not in o:
            raise RuntimeError('Tactical wave forecast is missing; reload the matching kr_env.lua')
        original = super()._vec(o)['summary']
        forecast = o.get('forecast') or {}
        features = [*enemy_profile(o.get('enemies') or []),
                    *enemy_profile(forecast.get('spawns') or [], True), float(forecast.get('available', False))]
        sites = self._sites(o)
        for action in (*range(2,6), *range(15,23)):
            features.extend(sites[action][2] if action in sites else [0]*SITE_FEATURES)
        counts = Counter()
        for spawn in forecast.get('spawns') or []:
            counts[spawn.get('name')] += float(spawn.get('count') or 0)
        total = max(1, sum(counts.values()))
        features.extend(counts[name]/total for name in ENEMY_TYPES)
        features.append(sum(count for name,count in counts.items() if name not in ENEMY_TYPES)/total)
        return {'summary': np.concatenate((original, np.asarray(features,dtype=np.float32)))}

    def _info(self, o):
        info = super()._info(o)
        info['forecast_available'] = bool((o.get('forecast') or {}).get('available'))
        return info
