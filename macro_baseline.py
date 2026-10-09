"""A deterministic heuristic teacher over the existing macro schema."""
from collections import Counter
import numpy as np


class MacroBaseline:
    num_timesteps = 0

    def __init__(self, env):
        self.env = env.unwrapped
        # This environment is owned by the rule-policy evaluation only. Keep
        # the learned policy's action decoder and schema completely unchanged.
        original = self.env._candidates
        def baseline_candidates(o):
            candidates = original(o)
            option = self._tesla_upgrade(o)
            if option is not None:
                candidates[9] = option
            return candidates
        self.env._candidates = baseline_candidates

    def _tesla_upgrade(self, o):
        if o.get('outcome') or o.get('replacement_pending'):
            return None
        from kr_tactical import enemy_profile
        enemies = o.get('enemies') or []
        future = (o.get('forecast') or {}).get('spawns') or []
        current, upcoming = enemy_profile(enemies), enemy_profile(future, True)
        crowded = (sum(not e.get('flying') for e in enemies) >= 3 and current[-1] >= .375) or (
            sum(float(e.get('count') or 0) for e in future if not e.get('flying')) >= 6
            and upcoming[-1] * (1-upcoming[1]) >= .5)
        if not crowded:
            return None
        best = None
        cm = self.env._cmax(o)
        for i, sp in enumerate(o.get('spots') or []):
            if sp.get('state') != 'tower' or sp.get('name') != 'tower_engineer_3':
                continue
            for op in sp.get('options') or []:
                if op.get('tpl') != 'tower_tesla' or op.get('locked') or not op.get('ok'):
                    continue
                score = self.env._quality(sp, 'engineer', cm)/max(1,float(op.get('price') or 1))
                if best is None or score > best[2]:
                    best = ([('upgrade',sp['hid'],'tower_tesla')],('upgrade',sp,{'index':i}),score)
        return best

    def predict(self, obs, action_masks, deterministic=True):
        env, mask = self.env, action_masks
        o = env.o
        towers = [s for s in o.get("spots", []) if s.get("state") == "tower"]
        counts = Counter(env.tower_kind(s) for s in towers)
        if o.get("enemies"):
            for action in (10, 12, 11):
                if mask[action]:
                    return np.int64(action), None
        kinds = ("archer", "barrack", "mage", "engineer")
        if mask[9] and self._tesla_upgrade(o) is not None:
            return np.int64(9), None
        if env.schema in ('macro-v2-tactical','macro-v3-replacement','macro-v4-strategy'):
            if env.schema in ('macro-v3-replacement','macro-v4-strategy'):
                plans=env._replacement_plans(o)
                legal=[a for a in plans if mask[a]]
                if legal:
                    return np.int64(max(legal,key=lambda a: plans[a]['features'][-1])),None
            from kr_tactical import enemy_profile
            current = enemy_profile(o.get('enemies') or [])
            future = enemy_profile((o.get('forecast') or {}).get('spawns') or [], True)
            profile = .5*(current+future) if o.get('enemies') else future
            # Reference heuristic only. PPO is not trained by imitation here.
            values = [profile[5], profile[6]*.25, profile[7], profile[8]*(.3+profile[9])]
            if mask[13]:
                return np.int64(13), None
            if mask[14]:
                return np.int64(14), None
            choices = []
            for action, (i, spot, features) in env._sites(o).items():
                if not mask[action]:
                    continue
                k = action-2 if action < 6 else (action-15)%4
                if k == 1 and counts['barrack'] >= max(1, len(towers)//4):
                    continue
                score = values[k] / (1+counts[kinds[k]]) + .1*features[2] + .1*features[4]
                if k == 1:
                    score += features[10]
                choices.append((score, action))
            if choices:
                return np.int64(max(choices)[1]), None
            if env.schema == 'macro-v4-strategy':
                skills=env._skill_plans(o)
                legal=[a for a in skills if mask[a]]
                if legal:
                    # Compatibility baseline, not a learned marginal-DPS model.
                    best=max(legal,key=lambda a: (skills[a][2][9]+skills[a][2][10]+skills[a][2][14]) /
                             max(1,float(skills[a][1].get('price') or 1)))
                    return np.int64(best),None
            for action in sorted(range(6,10), key=lambda a: values[a-6], reverse=True):
                if mask[action]:
                    return np.int64(action), None
            return np.int64(1 if mask[1] else 0), None
        preferred = [k for k in (0, 2, 3, 1) if mask[2 + k] and
                     (k != 1 or counts["barrack"] < max(1, len(towers) // 4))]
        if preferred and (len(towers) < 5 or max(counts.values(), default=0) < 3):
            k = min(preferred, key=lambda k: counts[kinds[k]])
            return np.int64(2 + k), None
        for action in (6, 8, 9, 7):
            if mask[action]:
                return np.int64(action), None
        if preferred:
            return np.int64(2 + min(preferred, key=lambda k: counts[kinds[k]])), None
        if mask[1]:
            return np.int64(1), None
        return np.int64(0 if mask[0] else np.flatnonzero(mask)[0]), None
