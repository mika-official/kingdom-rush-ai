"""Gymnasium environment for the injected Kingdom Rush game.

Built on the user's reward-shaped version. Changes in this revision (Lua side: kr_env.lua v5):

* Human-knowledge action masks (switch off with smart_masks=False):
    - towers only on spots above the static path-coverage quality threshold;
    - selling disabled to prevent destructive sell/rebuild loops;
    - each barracks rally point can be changed at most once per wave;
    - rain of fire only on a sufficiently dense enemy cluster, reinforcements only on the blocking
      point just ahead of the front-most enemy.
* The next wave is called automatically just before its countdown runs out (countdown = group.interval / FPS),
  so waves never pile up and the fixed early-wave bonus is still collected. The agent only calls wave 1
  (which has no countdown), with an automatic call after first_wave_delay_ticks.
* Build/upgrade shaping uses the STATIC spot quality instead of distance to the current enemies. The old
  "pressure" term paid most for building next to enemies, which pulls construction towards the rear of the path.
* Out-of-level upgrades (stars) are part of the episode: meta="rl" starts every episode with a purchase phase
  (buy a level of one of 6 classes, or START). The levels are applied in memory only; the save is never written.
"""
import argparse
from collections import Counter

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from kr_client import KRClient, lua_literal

KINDS = ["archer", "barrack", "mage", "engineer"]
MAX_SPOTS = 12
MAX_ENEMIES = 32
MAX_ANCHORS = 12
T = 2 + MAX_ANCHORS
BLOCK = len(KINDS) + 1 + 2
BASE_POW = 2 + MAX_SPOTS * BLOCK
BASE_HERO = BASE_POW + 3 * T
BASE_RALLY = BASE_HERO + T
N_PLAY = BASE_RALLY + MAX_SPOTS * 3
N_META = 7                          # buy one level of each of 6 upgrade classes + START
BASE_META = N_PLAY
N_ACT = N_PLAY + N_META

UP_CLASSES = ["archers", "barracks", "mages", "engineers", "rain", "reinforcements"]
DEFAULT_COSTS = {
    "archers": [1, 1, 2, 2, 3], "barracks": [1, 1, 2, 2, 3], "mages": [1, 1, 2, 2, 3],
    "engineers": [1, 1, 2, 3, 3], "rain": [2, 2, 3, 3, 3], "reinforcements": [2, 3, 3, 3, 4],
}

# 6 base + 6 power + 7 hero + 1 waves_finished + 2 wave timer + 8 meta (phase, stars left, 6 class levels).
N_GLOBAL = 30
# 26 (position + local enemy features) + 8 static path coverage (c, p per kind) + best quality + any-build-allowed.
N_SPOT_F = 36
N_ENEMY_F = 9

# Reward values are intentionally moderate.  The important change is not just
# the absolute size, but that progress during an episode produces signal.
DEFAULT_REWARD = dict(
    # Main objectives: wave progress, victory, and lives remaining at victory.
    # Kills and action shaping provide only small auxiliary learning signals.
    kill=0.02,
    leak=1.0,
    wave=2.0,

    # Economy is deliberately NOT rewarded directly.  Otherwise the agent can
    # learn to farm gold / call waves instead of building a defense.
    gold=0.0,
    early=0.001,

    # Action shaping.  Build/upgrade rewards depend on local enemy pressure.
    build=0.01,
    build_position=0.04,
    upgrade=0.01,
    upgrade_position=0.025,
    sell=0.75,
    sell_loss=0.005,
    call_wave_busy=0.10,

    # Power placement shaping (only paid when the power lands where a human would put it).
    power_hit=0.01,
    power_block=0.01,

    win=80.0,
    win_lives=100.0,
    lose=10.0,
    timeout=5.0,
    step=0.0005,
    invalid=0.02,
)

MAP_W = 1024.0
MAP_H = 768.0
MAP_DIAG = float(np.hypot(MAP_W, MAP_H))
LOCAL_ENEMY_RADIUS = 180.0


def _opt_ok(sp, tpl):
    for op in sp.get("options", []):
        if op.get("tpl") == tpl:
            return bool(op.get("ok"))
    return False


def _upg_options(sp):
    return sp.get("options", []) if sp.get("state") == "tower" else []


class KRGymEnv(gym.Env):
    """Single-game Gymnasium environment with a masked discrete action space."""

    metadata = {"render_modes": []}
    n_actions = N_ACT
    base_meta = BASE_META

    def __init__(
        self,
        port=9001,
        ticks_per_step=30,
        max_ticks=80000,
        settle_ticks=10,
        reward=None,
        smart_masks=True,
        auto_wave=True,
        first_wave_delay_ticks=1200,
        wave_margin=1.0,
        min_quality=0.4,
        rear_progress=0.7,
        rain_min_enemies=3,
        freeze_build_last_wave=True,
        meta="off",
        meta_budget=None,
        meta_levels=None,
    ):
        super().__init__()
        self.ticks_per_step = int(ticks_per_step)
        self.max_ticks = int(max_ticks)
        self.settle_ticks = int(settle_ticks)
        self.rw = dict(DEFAULT_REWARD, **(reward or {}))
        self.smart = bool(smart_masks)
        self.auto_wave = bool(auto_wave)
        self.first_wave_delay = int(first_wave_delay_ticks)
        self.min_quality = float(min_quality)
        self.rear_progress = float(rear_progress)
        self.rain_min = int(rain_min_enemies)
        self.freeze_last = bool(freeze_build_last_wave)
        self.ep = {}
        self.rally_used_wave = {}  # spot index -> last wave in which rally was changed
        if meta not in ("off", "rl", "fixed"):
            raise ValueError("meta must be 'off', 'rl' or 'fixed'")
        self.meta = meta

        self.client = KRClient(port=port)
        self.info_env = self.client.load_env()
        self.client.manual(True)
        self.client.call("ENV.configure(%s)" % lua_literal({"auto_wave": self.auto_wave, "wave_margin": float(wave_margin)}))

        # ---- out-of-level upgrades ----
        self.meta_info = self.client.call("ENV.meta_info()")
        self.costs = dict(DEFAULT_COSTS)
        if self.meta != "off":
            if not self.meta_info.get("available"):
                raise RuntimeError("upgrade modules are not available in this game build (meta=%r)" % meta)
            self.costs.update({c: list(v) for c, v in (self.meta_info.get("costs") or {}).items() if v})
        if meta_budget is None:
            stars = self.meta_info.get("stars")
            meta_budget = min(int(stars), 65) if stars else 30
        self.meta_budget = max(1, int(meta_budget))
        self.fixed_levels = {c: int((meta_levels or {}).get(c, 0)) for c in UP_CLASSES}
        self.meta_levels = {c: 0 for c in UP_CLASSES}
        self.stars_left = self.meta_budget
        self.phase = "play"
        self.probe = None

        self.action_space = spaces.Discrete(self.n_actions)
        inf = np.inf
        self.observation_space = spaces.Dict(
            {
                "globals": spaces.Box(-inf, inf, (N_GLOBAL,), np.float32),
                "spots": spaces.Box(-inf, inf, (MAX_SPOTS, N_SPOT_F), np.float32),
                "enemies": spaces.Box(-inf, inf, (MAX_ENEMIES, N_ENEMY_F), np.float32),
            }
        )

        self.o = None
        self.prev_o = None
        self.t = 0
        self.lives0 = 20

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _power_hero_feats(self, o):
        f = []
        powers = o.get("powers") or []
        for n in range(3):
            p = powers[n] if n < len(powers) else {}
            f += [
                1.0 if p.get("ready") else 0.0,
                min(float(p.get("cooldown_left", 0.0)), 60.0) / 60.0,
            ]

        h = o.get("hero")
        if h:
            hp_max = max(1.0, float(h.get("hp_max", 1.0)))
            ult_ready = bool(
                h.get("ult_level", 0) > 0
                and len(powers) > 2
                and powers[2].get("ready")
            )
            f += [
                1.0,
                0.0 if h.get("dead") else 1.0,
                float(h.get("hp", 0.0)) / hp_max,
                float(h.get("level", 0.0)) / 10.0,
                1.0 if ult_ready else 0.0,
                float(h.get("x", 0.0)) / MAP_W,
                float(h.get("y", 0.0)) / MAP_H,
            ]
        else:
            f += [0.0] * 7
        return f

    @staticmethod
    def _nearest_enemy_features(sp, enemies):
        """Return spatial features that are useful to a tower-placement policy.

        [nearest distance, enemies within radius, nearest hp ratio,
         nearest enemy to_goal]
        """
        sx = float(sp.get("x", 0.0))
        sy = float(sp.get("y", 0.0))
        nearest = None
        nearest_d2 = None
        nearby = 0

        for e in enemies:
            ex = float(e.get("x", 0.0))
            ey = float(e.get("y", 0.0))
            dx = ex - sx
            dy = ey - sy
            d2 = dx * dx + dy * dy
            if d2 <= LOCAL_ENEMY_RADIUS * LOCAL_ENEMY_RADIUS:
                nearby += 1
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest = e

        if nearest is None:
            return [1.0, 0.0, 0.0, 1.0]

        dist = np.sqrt(max(0.0, nearest_d2)) / MAP_DIAG
        hp = float(nearest.get("hp", 0.0))
        hp_max = max(1.0, float(nearest.get("hp_max", 1.0)))
        to_goal = float(nearest.get("to_goal") or 0.0) / 200.0
        return [
            min(float(dist), 1.0),
            min(float(nearby), 16.0) / 16.0,
            np.clip(hp / hp_max, 0.0, 1.0),
            np.clip(to_goal, 0.0, 1.0),
        ]

    def _vec(self, o):
        wave_total = max(1, int(o.get("wave_total", 1)))
        globals_ = [
            float(o.get("gold", 0.0)) / 500.0,
            float(o.get("lives", 0.0)) / max(1, self.lives0),
            float(o.get("wave", 0.0)) / wave_total,
            1.0 if o.get("can_call_wave") else 0.0,
            float(o.get("early_reward") or 0.0) / 50.0,
            min(float(self.t) / max(1, self.max_ticks), 1.0),
        ]
        globals_ += self._power_hero_feats(o)
        globals_.append(1.0 if o.get("waves_finished") else 0.0)
        left, total = o.get("next_wave_left"), o.get("next_wave_total")
        if left is not None and total:
            globals_ += [1.0, float(np.clip(left / max(1e-6, total), 0.0, 1.0))]
        else:
            globals_ += [0.0, 0.0]
        globals_ += self._meta_feats()
        g = np.asarray(globals_, dtype=np.float32)

        enemies = o.get("enemies") or []
        spots = np.zeros((MAX_SPOTS, N_SPOT_F), dtype=np.float32)
        cmax = self._cmax(o)

        for i, sp in enumerate((o.get("spots") or [])[:MAX_SPOTS]):
            st = sp.get("state", "missing")
            name = sp.get("name", "") or ""
            fam = [
                1.0 if (st == "tower" and k in name) else 0.0
                for k in KINDS
            ]
            fam.append(1.0 if (st == "tower" and not any(fam)) else 0.0)
            ups = _upg_options(sp)
            local = self._nearest_enemy_features(sp, enemies)

            row = (
                [
                    1.0,
                    1.0 if st == "empty" else 0.0,
                    1.0 if st == "building" else 0.0,
                    1.0 if st == "tower" else 0.0,
                    1.0 if st == "blocked" else 0.0,
                ]
                + fam
                + [
                    float(sp.get("level") or 0.0) / 4.0,
                    float(sp.get("spent") or 0.0) / 500.0,
                ]
                + [
                    1.0 if _opt_ok(sp, "tower_build_" + k) else 0.0
                    for k in KINDS
                ]
                + [
                    1.0 if (j < len(ups) and ups[j].get("ok")) else 0.0
                    for j in range(2)
                ]
                + [
                    float(sp.get("sell") or 0.0) / 100.0,
                    1.0 if any(sp.get("rally_ok") or []) else 0.0,
                ]
                + [
                    np.clip(float(sp.get("x", 0.0)) / MAP_W, 0.0, 1.0),
                    np.clip(float(sp.get("y", 0.0)) / MAP_H, 0.0, 1.0),
                ]
                + local
                + self._cover_feats(sp, cmax)
            )
            # Defensive assertion catches accidental feature-count changes.
            if len(row) != N_SPOT_F:
                raise RuntimeError(
                    "tower spot feature size mismatch: got %d, expected %d"
                    % (len(row), N_SPOT_F)
                )
            spots[i] = np.asarray(row, dtype=np.float32)

        en = np.zeros((MAX_ENEMIES, N_ENEMY_F), dtype=np.float32)
        for i, e in enumerate(enemies[:MAX_ENEMIES]):
            hp_max = max(1.0, float(e.get("hp_max", 1.0)))
            en[i] = [
                1.0,
                float(e.get("x", 0.0)) / MAP_W,
                float(e.get("y", 0.0)) / MAP_H,
                float(e.get("hp", 0.0)) / hp_max,
                float(e.get("hp_max", 0.0)) / 100.0,
                float(e.get("to_goal") or 0.0) / 200.0,
                float(e.get("speed") or 0.0) / 100.0,
                float(e.get("lives_cost") or 0.0),
                float(e.get("gold") or 0.0) / 10.0,
            ]

        return {"globals": g, "spots": spots, "enemies": en}

    # ------------------------------------------------------------------
    # Static tower-spot quality (path coverage), meta features
    # ------------------------------------------------------------------
    def _meta_feats(self):
        return [1.0 if self.phase == "meta" else 0.0, self.stars_left / float(self.meta_budget)] + [
            self.meta_levels[c] / 5.0 for c in UP_CLASSES
        ]

    @staticmethod
    def _cov(sp, kind):
        return ((sp.get("cov") or {}).get(kind)) or {}

    def _cmax(self, o):
        cm = {k: 1e-9 for k in KINDS}
        for sp in o.get("spots") or []:
            for k in KINDS:
                cm[k] = max(cm[k], float(self._cov(sp, k).get("c", 0.0)))
        return cm

    def _quality(self, sp, kind, cmax):
        """0..1: how much of the path a tower of this kind built here covers (relative to the best spot),
        discounted when the covered part is the rear of the path. Independent of where enemies are right now."""
        cv = self._cov(sp, kind)
        if not cv:                      # older Lua without coverage data: neutral, never blocks anything
            return 1.0
        if cmax[kind] < 1e-6:           # no spot covers any path sample (bad data): do not block everything
            return 1.0
        c = float(np.clip(float(cv.get("c", 0.0)) / cmax[kind], 0.0, 1.0))
        p = float(cv.get("p", 0.0))
        rp = self.rear_progress
        w = 1.0 if p <= rp else max(0.0, 1.0 - (p - rp) / max(1e-6, 1.0 - rp))
        return c * w

    def _cover_feats(self, sp, cmax):
        f = []
        for k in KINDS:
            cv = self._cov(sp, k)
            f += [float(np.clip(float(cv.get("c", 0.0)) / cmax[k], 0.0, 1.0)), float(cv.get("p", 0.0))]
        qs = [self._quality(sp, k, cmax) for k in KINDS]
        f += [max(qs), 1.0 if max(qs) >= self.min_quality else 0.0]
        return f

    def _final_wave(self, o):
        return int(o.get("wave", 0)) >= int(o.get("wave_total", 1)) or bool(o.get("waves_finished"))

    def _build_allowed(self, o, i, kind, cmax):
        if not self.smart:
            return True
        if self.freeze_last and self._final_wave(o):   # a tower started during the last wave is wasted money
            return False
        return self._quality(o["spots"][i], kind, cmax) >= self.min_quality

    @staticmethod
    def _action_info(action, o):
        """Return (kind, spot, extra) for action-dependent reward shaping."""
        if action >= BASE_META:
            return "meta", None, {"index": action - BASE_META}
        if action == 0:
            return "noop", None, {}
        if action == 1:
            return "wave", None, {}
        if action >= BASE_RALLY:
            i, k = divmod(action - BASE_RALLY, 3)
            if i < len(o.get("spots", [])):
                return "rally", o["spots"][i], {"index": i, "rally": k + 1}
            return "rally", None, {}
        if action >= BASE_HERO:
            return "hero_move", None, {}
        if action >= BASE_POW:
            n, t = divmod(action - BASE_POW, T)
            return "power", None, {"power": n + 1, "target": t + 1}

        i, k = divmod(action - 2, BLOCK)
        if i >= len(o.get("spots", [])):
            return "invalid", None, {}
        sp = o["spots"][i]
        if k < len(KINDS):
            return "build", sp, {"kind": KINDS[k], "index": i}
        if k == len(KINDS):
            return "sell", sp, {"index": i}
        return "upgrade", sp, {"index": i, "upgrade": k - len(KINDS)}

    # ------------------------------------------------------------------
    # Action masking / decoding
    # ------------------------------------------------------------------
    def _meta_mask(self):
        m = np.zeros(self.n_actions, dtype=np.bool_)
        for j, c in enumerate(UP_CLASSES):
            lvl = self.meta_levels[c]
            if lvl < 5 and self.costs[c][lvl] <= self.stars_left:
                m[self.base_meta + j] = True
        m[self.base_meta + len(UP_CLASSES)] = True          # START is always possible
        return m

    def _mask(self, o):
        if self.phase == "meta":
            return self._meta_mask()
        m = np.zeros(N_ACT, dtype=np.bool_)
        m[0] = True
        can_wave = bool(o.get("can_call_wave")) and o.get("outcome") is None
        # With auto_wave only wave 1 (no countdown) is called by the agent.
        m[1] = can_wave and (not self.auto_wave or int(o.get("wave", 0)) == 0)

        cmax = self._cmax(o)
        for i, sp in enumerate((o.get("spots") or [])[:MAX_SPOTS]):
            base = 2 + i * BLOCK
            state = sp.get("state")
            if state == "empty":
                for k, kind in enumerate(KINDS):
                    m[base + k] = _opt_ok(sp, "tower_build_" + kind) and self._build_allowed(o, i, kind, cmax)
            elif state == "tower":
                # Selling is disabled: previous runs showed destructive sell/rebuild loops.
                m[base + len(KINDS)] = False
                for j, op in enumerate(_upg_options(sp)[:2]):
                    m[base + len(KINDS) + 1 + j] = bool(op.get("ok"))
                # A barracks may change its rally point at most once per wave.
                rally_available = self.rally_used_wave.get(i) != int(o.get("wave", 0))
                for k, ok in enumerate((sp.get("rally_ok") or [])[:3]):
                    m[BASE_RALLY + i * 3 + k] = bool(ok) and rally_available

        if o.get("outcome") is None:
            tg = o.get("targets") or []

            def tn(i):
                return int((tg[i] or {}).get("n") or 0) if i < len(tg) else 0

            for n, p in enumerate((o.get("powers") or [])[:3]):
                for t, ok in enumerate((p.get("ok") or [])[:T]):
                    if not ok:
                        continue
                    if self.smart:
                        if n == 0 and not (t == 1 and tn(1) >= self.rain_min):    # rain: densest cluster only
                            continue
                        if n == 1 and t != 0:                                       # reinforcements: block ahead of the front
                            continue
                        if n == 2 and not (t == 1 and tn(1) >= 2):                  # hero ultimate: on a cluster
                            continue
                    m[BASE_POW + n * T + t] = True
            h = o.get("hero")
            if h:
                for t, ok in enumerate((h.get("move_ok") or [])[:T]):
                    m[BASE_HERO + t] = bool(ok)
        return m

    def action_masks(self):
        if self.o is None:
            # MaskablePPO normally asks after reset, but keeping a safe mask
            # here makes diagnostics and wrappers less fragile.
            m = np.zeros(self.n_actions, dtype=np.bool_)
            m[0] = True
            return m
        return self._mask(self.o)

    def _decode(self, a, o):
        if a >= BASE_META:
            return []
        if a == 0:
            return []
        if a == 1:
            return [("wave",)]
        if a >= BASE_RALLY:
            i, k = divmod(a - BASE_RALLY, 3)
            return (
                [("rally", o["spots"][i]["hid"], k + 1)]
                if i < len(o.get("spots", []))
                else []
            )
        if a >= BASE_HERO:
            return [("hero_move", a - BASE_HERO + 1)]
        if a >= BASE_POW:
            n, t = divmod(a - BASE_POW, T)
            return [("power", n + 1, t + 1)]

        i, k = divmod(a - 2, BLOCK)
        if i >= len(o.get("spots", [])):
            return []
        sp = o["spots"][i]
        if k < len(KINDS):
            return [("build", sp["hid"], KINDS[k])]
        if k == len(KINDS):
            return [("sell", sp["hid"])]
        ups = _upg_options(sp)
        j = k - len(KINDS) - 1
        return [("upgrade", sp["hid"], ups[j]["tpl"])] if j < len(ups) else []

    def _info(self, o):
        return {
            "action_mask": self._mask(o),
            "outcome": o.get("outcome"),
            "gold": o.get("gold"),
            "lives": o.get("lives"),
            "wave": o.get("wave"),
            "tick": o.get("tick"),
            "waves_finished": o.get("waves_finished", False),
            "level_run_complete": o.get("level_run_complete", False),
            "enemies_alive": o.get("enemies_alive", len(o.get("enemies", []))),
            "active_spawners": o.get("active_spawners", 0),
            "wave_spawn_pending": o.get("wave_spawn_pending", False),
            "phase": self.phase,
            "meta_levels": dict(self.meta_levels),
            "stars_left": self.stars_left,
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _apply_levels(self, levels):
        res = self.client.raw("ENV.apply_upgrades(%s)" % lua_literal(levels))
        if res != "ok":
            raise RuntimeError("ENV.apply_upgrades failed: %s" % res)
        self.probe = self.client.call("ENV.probe_upgrade()")

    def _start_play(self):
        """(Re)start the level; applies out-of-level upgrades first so the new level uses patched templates."""
        if self.meta == "rl":
            self._apply_levels(self.meta_levels)
        elif self.meta == "fixed":
            self._apply_levels(self.fixed_levels)
            self.meta_levels = dict(self.fixed_levels)
        o = self.client.reset(self.settle_ticks)
        if (
            o.get("wave") != 0
            or o.get("outcome") is not None
            or not o.get("spots")
        ):
            raise RuntimeError(
                "reset did not return the level to its start state: "
                "wave=%s outcome=%s spots=%d"
                % (o.get("wave"), o.get("outcome"), len(o.get("spots", [])))
            )
        self.phase = "play"
        self.rally_used_wave = {}
        self.o = o
        self.prev_o = o
        self.t = 0
        self.lives0 = max(1, int(o.get("lives", 20)))
        cmax = self._cmax(o)
        allowed = {
            k: sum(1 for i in range(len(o["spots"])) if self._build_allowed(o, i, k, cmax))
            for k in KINDS
        }
        self.ep = {
            "act": Counter(), "build_kind": Counter(), "kills": 0, "leaks": 0, "leaks_by_wave": Counter(),
            "gold_max": int(o.get("gold", 0)), "spots": len(o["spots"]), "allowed_spots": max(allowed.values()) if allowed else 0,
            "comp": Counter(),
        }
        return o

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.meta == "rl":
            # Purchase phase first; the level is (re)started when the agent chooses START.
            self.meta_levels = {c: 0 for c in UP_CLASSES}
            self.stars_left = self.meta_budget
            self.phase = "meta"
            if self.o is None:
                self.o = self.client.obs()
            self.prev_o = self.o
            return self._vec(self.o), self._info(self.o)
        o = self._start_play()
        return self._vec(o), self._info(o)

    def _meta_step(self, action):
        j = action - self.base_meta
        if not self._meta_mask()[action]:
            info = self._info(self.o)
            info.update({"results": [], "reward_components": {}})
            return self._vec(self.o), -self.rw["invalid"], False, False, info
        if j < len(UP_CLASSES):
            c = UP_CLASSES[j]
            self.stars_left -= self.costs[c][self.meta_levels[c]]
            self.meta_levels[c] += 1
        else:                                           # START
            self._start_play()
        info = self._info(self.o)
        info.update({"results": [], "reward_components": {}, "upgrade_probe": self.probe})
        return self._vec(self.o), 0.0, False, False, info

    def step(self, action):
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError("action %d is outside [0, %d)" % (action, self.n_actions))
        if self.phase == "meta" or action >= self.base_meta:
            if self.phase != "meta":                    # meta action in the play phase: masked, treated as invalid
                info = self._info(self.o)
                info.update({"results": [], "reward_components": {}})
                return self._vec(self.o), -self.rw["invalid"], False, False, info
            return self._meta_step(action)

        old = self.o
        mask = self._mask(old)
        valid_action = bool(mask[action])
        acts = self._decode(action, old) if valid_action else []
        invalid = action != 0 and not acts

        agent_acts = list(acts)
        # The first wave has no countdown: call it automatically after a while so an episode can never stall.
        if (
            self.auto_wave
            and int(old.get("wave", 0)) == 0
            and old.get("can_call_wave")
            and self.t >= self.first_wave_delay
            and not any(a[0] == "wave" for a in acts)
        ):
            acts = list(acts) + [("wave",)]

        r = self.client.turn(acts, self.ticks_per_step)
        o, st = r["obs"], r["step"]
        self.prev_o = old
        self.o = o
        self.t += int(st.get("ticks", 0))
        self.ep['unblock_clicks'] = self.ep.get('unblock_clicks', 0) + int(st.get('unblock_clicks', 0))
        self.ep['boss_finish_clicks'] = self.ep.get('boss_finish_clicks', 0) + int(st.get('boss_finish_clicks', 0))
        for key in ('leaks_by_path','leak_counts_by_path'):
            values=st.get(key) or {}
            if isinstance(values,dict):
                self.ep.setdefault(key,Counter()).update(values)
        self.ep['leaks_path_partial']=self.ep.get('leaks_path_partial',False) or bool(st.get('leaks_path_partial'))
        for key in ('leak_cohorts','leak_life_cohorts'):
            for wave,values in (st.get(key) or {}).items():
                self.ep.setdefault(key,{}).setdefault(wave,Counter()).update(values)
        self.ep['birth_attribution_partial']=self.ep.get('birth_attribution_partial',False) or any(
            not e.get('birth_verified') for e in (st.get('goal_events') or []))
        cohort_status=st.get('route_cohort_status') or {}
        if o.get('tower_damage') is not None:self.ep['tower_damage']=o['tower_damage']
        if isinstance(cohort_status.get('born'),dict):
            self.ep['birth_cohorts']=cohort_status['born']
            self.ep['birth_attribution_partial'] |= any(v.get('unverified',0)>0
                for paths in cohort_status['born'].values() for v in paths.values())

        kills = max(0, int(st.get("kills", 0)))
        d_lives = int(st.get("d_lives", 0))
        d_gold = int(st.get("d_gold", 0))
        old_wave = int(old.get("wave", 0))
        new_wave = int(o.get("wave", 0))
        wave_delta = max(0, new_wave - old_wave)

        old_early = float(old.get("early_reward") or 0.0)
        new_early = float(o.get("early_reward") or 0.0)
        early_delta = max(0.0, new_early - old_early)

        action_kind, action_spot, action_extra = self._action_info(action, old)
        # Shaping rewards are only paid for actions the game really executed (masked/rejected ones get nothing).
        results = r.get("results", [])
        agent_ok = bool(agent_acts) and bool(results) and bool(results[0].get("ok", False))
        if action != 0 and not agent_ok:
            action_kind, action_spot, action_extra = "invalid", None, {}
        elif action_kind == "rally" and agent_ok:
            self.rally_used_wave[int(action_extra["index"])] = int(old.get("wave", 0))
        cmax = self._cmax(old)
        quality = 0.0
        if action_spot is not None and action_kind == "build":
            quality = self._quality(action_spot, action_extra["kind"], cmax)
        elif action_spot is not None and action_kind == "upgrade":
            quality = max(self._quality(action_spot, k, cmax) for k in KINDS)

        reward = 0.0
        comp = {}

        # 1) Kills provide a small auxiliary learning signal.
        comp["kill"] = self.rw["kill"] * kills
        # 2) Penalize each lost life in addition to rewarding lives at victory.
        comp["leak"] = -self.rw["leak"] * max(0, -d_lives)
        # 3) Reward wave-number progress with the existing timing and delta cap.
        comp["wave"] = self.rw["wave"] * min(wave_delta, 2)
        # 4) Gold itself is not rewarded. The early-wave bonus is tiny (harvested automatically with auto_wave).
        comp["early"] = self.rw["early"] * min(early_delta, 10.0)
        comp["call_wave_busy"] = -self.rw["call_wave_busy"] if (action_kind == "wave" and old.get("enemies")) else 0.0

        # 5) Construction shaping uses the STATIC spot quality (path coverage, not the rear), never the
        #    instantaneous distance to enemies (that rewarded building next to enemies = at the rear).
        comp["build"] = (self.rw["build"] + self.rw["build_position"] * quality) if action_kind == "build" else 0.0
        comp["upgrade"] = (self.rw["upgrade"] + self.rw["upgrade_position"] * quality) if action_kind == "upgrade" else 0.0

        if action_kind == "sell" and action_spot is not None:
            spent = max(0.0, float(action_spot.get("spent") or 0.0))
            refund = max(0.0, float(action_spot.get("sell") or 0.0))
            lost = max(0.0, spent - refund)
            comp["sell"] = -(self.rw["sell"] + self.rw["sell_loss"] * min(lost, 500.0))
        else:
            comp["sell"] = 0.0

        # 6) Power placement: rain on a dense cluster, reinforcements on the blocking point.
        comp["power"] = 0.0
        if action_kind == "power":
            tg = old.get("targets") or []
            n, t = action_extra["power"], action_extra["target"]
            cnt = int((tg[t - 1] or {}).get("n") or 0) if t - 1 < len(tg) else 0
            if n == 1 and t == 2:
                comp["power"] = self.rw["power_hit"] * min(cnt, 8) / 8.0
            elif n == 2 and t == 1 and cnt >= 1:
                comp["power"] = self.rw["power_block"]

        ep = self.ep
        if ep:
            kind = action_kind
            if kind == "power":
                kind = {1: "rain", 2: "reinf", 3: "ult"}.get(action_extra.get("power"), "power")
            if kind not in ("noop", "invalid"):
                ep["act"][kind] += 1
            if action_kind == "build":
                ep["build_kind"][action_extra.get("kind")] += 1
            ep["kills"] += kills
            leaked = max(0, -d_lives)
            ep["leaks"] += leaked
            if leaked:
                ep["leaks_by_wave"][int(o.get("wave", 0))] += leaked
            ep["gold_max"] = max(ep["gold_max"], int(o.get("gold", 0)))
            for k2, v2 in comp.items():
                ep["comp"][k2] += v2

        reward += sum(comp.values())
        reward -= self.rw["step"]
        if invalid or any(not x.get("ok", False) for x in r.get("results", [])):
            reward -= self.rw["invalid"]

        terminated = o.get("outcome") is not None
        if o.get("outcome") == "victory":
            life_ratio = float(o.get("lives", 0)) / max(1, self.lives0)
            reward += self.rw["win"] + self.rw["win_lives"] * np.clip(life_ratio, 0.0, 1.0)
        elif o.get("outcome") == "defeat":
            reward -= self.rw["lose"]

        truncated = (not terminated) and self.t >= self.max_ticks
        if truncated:
            reward -= self.rw["timeout"]

        info = self._info(o)
        info["end_reason"] = o.get("outcome") if terminated else ("timeout" if truncated else None)
        info.update(
            {
                "results": list(zip(acts, r.get("results", []))),
                "kills": kills,
                "kill_gold": int(st.get("kill_gold", 0)),
                "unblock_clicks": int(st.get("unblock_clicks", 0)),
                "boss_finish_clicks": int(st.get("boss_finish_clicks", 0)),
                "agent_action_valid": valid_action,
                "agent_action_ok": agent_ok if action != 0 else valid_action,
                "d_lives": d_lives,
                "d_gold": d_gold,
                "wave_delta": wave_delta,
                "leak_cohorts": st.get("leak_cohorts") or {},
                "leak_life_cohorts": st.get("leak_life_cohorts") or {},
                "goal_events": st.get("goal_events") or [],
                "birth_events": st.get("birth_events") or [],
                "route_cohort_status": st.get("route_cohort_status") or {},
                "tower_damage":o.get('tower_damage'),
                "early_reward_delta": early_delta,
                "quality": quality,
                "reward_components": comp,
            }
        )
        if terminated or truncated:
            ep = self.ep
            info["ep_stats"] = {
                "actions": dict(ep.get("act", {})), "build_kinds": dict(ep.get("build_kind", {})),
                "kills": ep.get("kills", 0), "leaks": ep.get("leaks", 0),
                "unblock_clicks": ep.get("unblock_clicks", 0),
                "boss_finish_clicks": ep.get("boss_finish_clicks", 0),
                "leaks_by_wave": dict(ep.get("leaks_by_wave", {})), "gold_max": ep.get("gold_max", 0),
                "leaks_by_path":dict(ep.get("leaks_by_path",{})),
                "leak_counts_by_path":dict(ep.get("leak_counts_by_path",{})),
                "leaks_path_partial":ep.get('leaks_path_partial',False),
                "leak_cohorts":{w:dict(v) for w,v in ep.get('leak_cohorts',{}).items()},
                "leak_life_cohorts":{w:dict(v) for w,v in ep.get('leak_life_cohorts',{}).items()},
                "birth_attribution_partial":ep.get('birth_attribution_partial',False),
                "birth_cohorts":ep.get('birth_cohorts',{}),
                "tower_damage":ep.get('tower_damage'),
                "gold_end": o.get("gold"), "towers_end": sum(1 for sp in o.get("spots", []) if sp.get("state") == "tower"),
                "spots": ep.get("spots", 0), "allowed_spots": ep.get("allowed_spots", 0),
                "reward_parts": {k: round(v, 2) for k, v in ep.get("comp", {}).items() if abs(v) > 1e-9},
            }
        return self._vec(o), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.client.manual(False)
        finally:
            self.client.close()


def _action_type(a):
    if a >= BASE_META:
        return "meta"
    if a == 0:
        return "noop"
    if a == 1:
        return "wave"
    if a >= BASE_RALLY:
        return "rally"
    if a >= BASE_HERO:
        return "hero_move"
    if a >= BASE_POW:
        return "power%d" % ((a - BASE_POW) // T + 1)
    k = (a - 2) % BLOCK
    return "build" if k < len(KINDS) else ("sell" if k == len(KINDS) else "upgrade")


def _random_agent(port, episodes, ticks_per_step):
    env = KRGymEnv(port=port, ticks_per_step=ticks_per_step)
    print("env info:", env.info_env)
    rng = np.random.default_rng(0)
    ok, bad = Counter(), Counter()
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            total, steps, done = 0.0, 0, False
            while not done:
                valid = np.flatnonzero(info["action_mask"])
                if info["action_mask"][1] and rng.random() < 0.5:
                    a = 1
                else:
                    a = int(rng.choice(valid))
                obs, r, term, trunc, info = env.step(a)
                for act, res in info["results"]:
                    key = _action_type(a)
                    (ok if res.get("ok") else bad)[key] += 1
                total += r
                steps += 1
                done = term or trunc
            print(
                "episode %d: steps=%d return=%.2f outcome=%s lives=%s wave=%s tick=%s"
                % (
                    ep,
                    steps,
                    total,
                    info.get("outcome"),
                    info.get("lives"),
                    info.get("wave"),
                    info.get("tick"),
                )
            )
        print("actions accepted by the game:", dict(ok))
        print("actions rejected by the game:", dict(bad))
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a random masked agent (needs a running game)")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--step", type=int, default=30)
    a = ap.parse_args()
    _random_agent(a.port, a.episodes, a.step)
