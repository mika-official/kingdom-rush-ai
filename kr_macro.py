"""Map-independent macro actions over every observed tower spot (schema macro-v1)."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces

from kr_gym import KRGymEnv, KINDS, _opt_ok

AUDIT = json.loads(Path(__file__).with_name("level_audit.json").read_text(encoding="utf-8"))
ENEMY_TYPES = sorted({name for row in AUDIT["combinations"] for name in row["enemies"]})
MACROS = ("noop", "wave", *("build_" + k for k in KINDS),
          *("upgrade_" + k for k in KINDS), "rain", "reinforcements", "ultimate", "hero_move", "rally")
N_FEATURES = 28 + 13 + 8 + 4 * 9 + len(ENEMY_TYPES) + 1
MODE_IDS = {"campaign": 1, "heroic": 2, "iron": 3}
REWARD2 = dict(win=10.0, win_lives=5.0, lose=5.0, timeout=5.0,
               leak=0.25, wave=2.0, kill_gold=0.0003, invalid=0.01)


class LevelSchedule:
    """Shuffled full sweeps, with optional episode-based expansion of the level pool."""
    def __init__(self, levels, modes, curriculum_every=0):
        self.levels = sorted(set(levels))
        self.modes = list(dict.fromkeys(modes))
        if not self.levels or any(x not in range(1, 27) for x in self.levels):
            raise ValueError("levels must be in 1..26")
        if not self.modes or any(x not in (1, 2, 3) for x in self.modes):
            raise ValueError("modes must be 1, 2, or 3")
        if curriculum_every < 0:
            raise ValueError("curriculum_every must be nonnegative")
        self.curriculum_every = curriculum_every
        self.episodes = 0
        self.queue = []
        self.width = 0

    @property
    def pairs(self):
        return [(level, mode) for level in self.levels for mode in self.modes]

    def next(self, rng):
        width = min(len(self.levels), 1 + self.episodes // self.curriculum_every) if self.curriculum_every else len(self.levels)
        if not self.queue or width != self.width:
            self.width = width
            pairs = [(level, mode) for level in self.levels[:width] for mode in self.modes]
            self.queue = [pairs[i] for i in rng.permutation(len(pairs))]
        self.episodes += 1
        return self.queue.pop()


class KRMacroEnv(KRGymEnv):
    base_meta = len(MACROS)
    n_actions = base_meta + 7
    schema = "macro-v1"

    def __init__(self, levels=range(1, 27), modes=(1,), difficulty=2,
                 curriculum_every=0, backup_dir="runs/save_backups", spend_before_wait=False,
                 reward_profile="balanced", loadout="original", **kwargs):
        self.loadout = loadout
        if reward_profile not in ("balanced", "legacy"):
            raise ValueError("Unknown reward profile")
        self.reward_profile = reward_profile
        self.schedule = LevelSchedule(levels, modes, curriculum_every)
        if difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2 or 3")
        self.difficulty = difficulty
        self.spend_before_wait = spend_before_wait
        self.selected_pair = None
        self.evaluation_pair = None
        self.config = {"auto_wave": kwargs.get("auto_wave", True), "wave_margin": kwargs.get("wave_margin", 1.0)}
        super().__init__(**kwargs)
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.client.enable_multilevel(Path(backup_dir) / f"slot_{stamp}.json")
        except Exception:
            self.close()
            raise
        self.observation_space = spaces.Dict({"summary": spaces.Box(-np.inf, np.inf, (N_FEATURES,), np.float32)})

    def snapshot_hash(self):
        return self.client.snapshot_sha256

    def _start_play(self):
        obs = super()._start_play()
        self.reward_totals = Counter()
        return obs

    def reset(self, *, seed=None, options=None):
        # Gym initializes the RNG before schedule selection, preserving seeded sweeps.
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        pair = (options or {}).get("level_mode", self.evaluation_pair)
        if pair is None:
            pair = self.schedule.next(self.np_random)
        if tuple(pair) not in self.schedule.pairs:
            raise ValueError("Requested level/mode is outside this environment's pool")
        self.selected_pair = tuple(pair)
        from kr_client import lua_literal
        self.client.call("AI.training_loadout(%s)" % lua_literal(self.loadout))
        self.info_env = self.client.switch_level(*self.selected_pair, difficulty=self.difficulty)
        from kr_client import lua_literal
        self.client.call("ENV.configure(%s)" % lua_literal(self.config))
        # Refresh meta-phase context after switching, even when the previous episode ended.
        self.o = self.client.obs()
        self.t = 0
        self.lives0 = max(1, int(self.o.get("lives") or 20))
        return super().reset(options=options)

    @staticmethod
    def tower_kind(sp):
        kind = sp.get("kind")
        if kind in KINDS:
            return kind
        name = sp.get("name", "")
        return next((k for k in KINDS if name.startswith("tower_" + k)), None)

    def _candidates(self, o):
        # action -> (command list, reward attribution, score). Recomputed from the
        # same observation for mask/decode/reward; indices never cross environments.
        candidates = {0: ([], ("noop", None, {}), 0)}
        if o.get("outcome") is not None:
            return candidates
        if o.get("can_call_wave") and (not self.auto_wave or int(o.get("wave", 0)) == 0):
            candidates[1] = ([("wave",)], ("wave", None, {}), 0)
        cm = self._cmax(o)

        def offer(a, command, info, score):
            if a not in candidates or score > candidates[a][2]:
                candidates[a] = ([command], info, score)

        for i, sp in enumerate(o.get("spots") or []):
            if sp.get("state") == "empty":
                for k, kind in enumerate(KINDS):
                    if _opt_ok(sp, "tower_build_" + kind) and self._build_allowed(o, i, kind, cm):
                        offer(2 + k, ("build", sp["hid"], kind), ("build", sp, {"kind": kind, "index": i}), self._quality(sp, kind, cm))
            elif sp.get("state") == "tower":
                kind = self.tower_kind(sp)
                if kind is not None:
                    for op in sp.get("options") or []:
                        if op.get("ok"):
                            # Coverage per price is a heuristic, not a damage estimate.
                            score = self._quality(sp, kind, cm) / max(1, float(op.get("price") or 1))
                            offer(6 + KINDS.index(kind), ("upgrade", sp["hid"], op["tpl"]), ("upgrade", sp, {"index": i}), score)
                if self.rally_used_wave.get(i) != int(o.get("wave", 0)):
                    for target, ok in enumerate(sp.get("rally_ok") or []):
                        if ok:
                            offer(14, ("rally", sp["hid"], target + 1), ("rally", sp, {"index": i}), self._quality(sp, "barrack", cm))
        targets = o.get("targets") or []
        cluster = int(targets[1].get("n") or 0) if len(targets) > 1 else 0
        for n, target in enumerate((2, 1, 2)):
            powers = o.get("powers") or []
            legal = (powers[n].get("ok") or []) if n < len(powers) else []
            if len(legal) >= target and legal[target - 1]:
                if self.smart and ((n == 0 and cluster < self.rain_min) or (n == 2 and cluster < 2)):
                    continue
                offer(10 + n, ("power", n + 1, target), ("power", None, {"power": n + 1, "target": target}), 0)
        hero = o.get("hero") or {}
        if (hero.get("move_ok") or [False])[0]:
            offer(13, ("hero_move", 1), ("hero_move", None, {}), 0)
        if self.spend_before_wait and any(a in candidates for a in range(2, 10)):
            candidates.pop(0)
        return candidates

    def _mask(self, o):
        if self.phase == "meta":
            return self._meta_mask()
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        mask[list(self._candidates(o))] = True
        return mask

    def _decode(self, action, o):
        return self._candidates(o).get(action, ([], None, 0))[0]

    def _action_info(self, action, o):
        return self._candidates(o).get(action, ([], ("invalid", None, {}), 0))[1]

    def action_kind(self, action):
        if action >= self.base_meta:
            return "meta"
        if 2 <= action < 6:
            return "build"
        if 6 <= action < 10:
            return "upgrade"
        return {10: "power1", 11: "power2", 12: "power3"}.get(action, MACROS[action])

    def reinforcements_available(self, mask):
        return bool(mask[11])

    def _vec(self, o):
        enemies, spots = o.get("enemies") or [], o.get("spots") or []
        n, ns = max(1, len(enemies)), max(1, len(spots))
        cm = self._cmax(o)
        log = lambda x, scale: float(np.log1p(max(0, x)) / np.log1p(scale))
        armor = [float(e.get("armor") or 0) for e in enemies]
        magic = [float(e.get("magic_armor") or 0) for e in enemies]
        f = [log(float(o.get("gold") or 0), 3500), float(o.get("lives") or 0) / max(1, self.lives0),
             float(o.get("lives") or 0) / 20, float(o.get("wave") or 0) / max(1, int(o.get("wave_total") or 1)),
             float(o.get("wave_total") or 0) / 20, self.t / max(1, self.max_ticks),
             float(bool(o.get("waves_finished"))), float(bool(o.get("can_call_wave"))),
             *[float(o.get("level_mode") == m) for m in (1, 2, 3)],
             float(o.get("difficulty") or 0) / 3, log(len(enemies), 100),
             log(sum(float(e.get("hp") or 0) for e in enemies), 10000),
             max((float(e.get("speed") or 0) for e in enemies), default=0) / 100,
             sum(bool(e.get("flying")) for e in enemies) / n,
             sum(armor) / n, max(armor, default=0), sum(magic) / n, max(magic, default=0),
             min((float(e["to_goal"]) for e in enemies if e.get("to_goal") is not None), default=200) / 200,
             len({e.get("pi") for e in enemies if e.get("pi") is not None}) / 9,
             len(spots) / 18, sum(s.get("state") == "tower" for s in spots) / ns,
             sum(s.get("state") == "empty" for s in spots) / ns,
             float(bool(o.get("locked_hero"))), float(o.get("max_upgrade_level") or 0) / 5,
             float(o.get("next_wave_left") or 0) / max(1, float(o.get("next_wave_total") or 1))]
        f += self._power_hero_feats(o) + self._meta_feats()
        locked = o.get("locked_towers") or []
        for kind in KINDS:
            towers = [s for s in spots if s.get("state") == "tower" and self.tower_kind(s) == kind]
            empty = [s for s in spots if s.get("state") == "empty"]
            good = [s for s in empty if self._quality(s, kind, cm) >= self.min_quality]
            options = [op for s in towers for op in s.get("options") or [] if not op.get("locked")]
            f += [len(towers) / ns, sum(float(s.get("level") or 0) for s in towers) / (4 * ns),
                  len(good) / ns, max((self._quality(s, kind, cm) for s in empty), default=0),
                  float(any(_opt_ok(s, "tower_build_" + kind) for s in good)),
                  float(any(op.get("ok") for op in options)),
                  log(min((float(op["price"]) for op in options if op.get("price") is not None), default=0), 1000),
                  log(sum(float(s.get("spent") or 0) for s in towers), 5000),
                  float("tower_build_" + kind in locked)]
        counts = Counter(e.get("name") for e in enemies)
        f += [counts[name] / n for name in ENEMY_TYPES]
        f += [sum(count for name, count in counts.items() if name not in ENEMY_TYPES) / n]
        return {"summary": np.asarray(f, dtype=np.float32)}

    def _info(self, o):
        info = super()._info(o)
        info.update(level_idx=o.get("level_idx"), level_mode=o.get("level_mode"),
                    difficulty=o.get("difficulty"), schema=self.schema,
                    wave_total=o.get("wave_total"), lives0=self.lives0,
                    reward_profile=self.reward_profile)
        return info

    def step(self, action):
        old_wave = int((self.o or {}).get("wave") or 0)
        obs, reward, terminated, truncated, info = super().step(action)
        if self.reward_profile == "balanced" and "d_lives" in info:
            o, r = self.o, REWARD2
            total = max(1, int(o.get("wave_total") or 1))
            ratio = float(np.clip(float(o.get("lives") or 0) / max(1, self.lives0), 0, 1))
            # Reaching a later wave is progress, not proof that earlier enemies
            # were cleared. Never pay for starting wave 1 (especially iron).
            later_waves = max(0, min(int(o.get("wave") or 0), total) - max(1, old_wave))
            comp = dict(leak=-r["leak"] * max(0, -int(info["d_lives"])),
                        wave=r["wave"] * later_waves / total * ratio,
                        kill=r["kill_gold"] * max(0, int(info.get("kill_gold", 0))))
            if not info.get("agent_action_ok", True):
                comp["invalid"] = -r["invalid"]
            if o.get("outcome") == "victory":
                comp["win"] = r["win"] + r["win_lives"] * ratio
            elif o.get("outcome") == "defeat":
                comp["lose"] = -r["lose"]
            elif truncated:
                comp["timeout"] = -r["timeout"]
            self.reward_totals.update(comp)
            info["reward_components"] = comp
            if "ep_stats" in info:
                info["ep_stats"]["reward_parts"] = {k: round(v, 6) for k, v in self.reward_totals.items()}
            return obs, float(sum(comp.values())), terminated, truncated, info
        # Keep total wave shaping comparable: campaign 7..20, heroic 6, iron 1.
        parts = info.get("reward_components", {})
        if "wave" in parts:
            old = parts["wave"]
            parts["wave"] = old * 10 / max(1, int(self.o.get("wave_total") or 1))
            reward += parts["wave"] - old
            self.ep["comp"]["wave"] += parts["wave"] - old
            if "ep_stats" in info:
                info["ep_stats"]["reward_parts"]["wave"] = round(self.ep["comp"]["wave"], 2)
        return obs, float(reward), terminated, truncated, info
