"""Evaluation diagnostics; never changes action masks or rewards."""
import json
import os
import random
from collections import Counter
from contextlib import contextmanager


@contextmanager
def evaluation_rng(seed):
    """Keep evaluation sampling from advancing the training RNG streams."""
    import numpy as np
    import torch
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def action_kind(action):
    from kr_gym import _action_type
    return _action_type(action)


def wave_progress(wave, total, victory):
    """Reached-wave proxy, NOT a count of cleared waves (waves can overlap)."""
    if victory:
        return 1.0
    total = max(1, int(total or 1))
    return min(max(0, int(wave or 0) - 1), total - 1) / total


def record_step(rows, base, masks, action, info, wave, gold, phase):
    if phase != "play":
        return
    from kr_gym import BASE_POW, T
    classify = getattr(base, "action_kind", action_kind)
    previous_kinds=info.get('candidate_action_kinds')
    if previous_kinds is not None:
        classify=lambda a: previous_kinds[a] if a<len(previous_kinds) else 'meta' if a>=base.base_meta else 'invalid'
    row = rows.setdefault(wave, Counter())
    spending = [i for i, allowed in enumerate(masks)
                if allowed and classify(i) in ("build", "upgrade", "tower_skill")]
    reinf_available = (any(allowed and classify(i)=='power2' for i,allowed in enumerate(masks)) if previous_kinds is not None else
                      base.reinforcements_available(masks) if hasattr(base, "reinforcements_available")
                       else any(masks[BASE_POW + T:BASE_POW + 2 * T]))
    kind = classify(action)
    # The first result belongs to the agent; auto-wave results follow it.
    results = info.get("results") or []
    succeeded = bool(results and results[0][1].get("ok", False))
    row["steps"] += 1
    row["gold_sum"] += gold
    row["gold_max"] = max(row["gold_max"], gold)
    row["gold_last"] = gold
    row["selected_" + kind] += 1
    if spending:
        row["spend_available_steps"] += 1
        row["spend_legal_actions_sum"] += len(spending)
        row["spend_available_" + ("spend" if kind in ("build", "upgrade", "tower_skill") else
                                  "wait" if kind == "noop" else "other")] += 1
    if kind in ("build", "upgrade", "tower_skill"):
        row["spend_attempts"] += 1
        row["spend_success"] += int(succeeded)
    row["reinf_available_steps"] += int(reinf_available)
    if kind == "power2":
        row["reinf_attempts"] += 1
        row["reinf_success"] += int(succeeded)


def evaluate(model, env, episodes, deterministic=True, diagnostic_wave=8,
             output=None, seed=42, label="", record_context=None):
    mode = "deterministic" if deterministic else "stochastic"
    records = []
    base = env.unwrapped
    with evaluation_rng(seed):
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=seed + ep)
            total, steps = 0.0, 0
            rows = {}
            advanced_peak=0;advanced_seconds=0.;advanced_seen=set()
            while True:
                masks = base.action_masks()
                wave = int(base.o.get("wave", 0))
                gold = int(base.o.get("gold", 0))
                phase = base.phase
                action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
                action = int(action)
                investment_diagnostics=None
                policy=getattr(model,'policy',None)
                if policy is not None and hasattr(policy,'investment_state'):
                    import torch
                    with torch.no_grad():
                        tensor,_=policy.obs_to_tensor(obs)
                        investment=policy.investment_state(tensor,masks)
                        from support_features import F
                        waited=bool(tensor['candidates'][0,action,F['type_wait']]>.5)
                        investment_diagnostics=dict(investment_ready=int(investment['ready'][0]),
                            investment_ready_wait=int(bool(investment['ready'][0]) and waited),
                            investment_ready_progress=int(investment['match'][0,action]),
                            investment_bootstrap_wait=int(bool(investment['bootstrap'][0]) and waited))
                previous_obs=obs
                tick_before=getattr(base,'t',0)
                obs, reward, term, trunc, info = env.step(action)
                advanced_names={'tower_tesla','tower_bfg','tower_ranger','tower_musketeer',
                                'tower_sorcerer','tower_arcane_wizard','tower_paladin','tower_barbarian'}
                towers=[s for s in base.o.get('spots') or [] if s.get('name') in advanced_names]
                advanced_peak=max(advanced_peak,len(towers));advanced_seen.update(s['hid'] for s in towers)
                advanced_seconds+=len(towers)*float(getattr(base,'tick_seconds',1/30))*max(0,getattr(base,'t',tick_before)-tick_before)
                if output and info.get('candidate_catalog'):
                    from candidate_diagnostics import evaluation_event
                    event=evaluation_event(model,previous_obs,masks,info,action)
                    for key in ('route_predictions','route_prediction_waves'):
                        if key in info['candidate_event']:
                            event[key]=info['candidate_event'][key]
                    with open(str(output)+'.candidates.jsonl','a',encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(label=label,timesteps=model.num_timesteps,**event))+'\n')
                record_step(rows, base, masks, action, info, wave, gold, phase)
                if investment_diagnostics is not None:
                    for key,value in investment_diagnostics.items():rows[wave][key]+=value
                total += float(reward)
                steps += 1
                if term or trunc:
                    break
            won = info.get("outcome") == "victory"
            lives = int(info.get("lives") or 0)
            record = dict(label=label, mode=mode, timesteps=model.num_timesteps,
                          episode=ep, outcome=info.get("outcome"), end=info.get("end_reason"),
                          wave=info.get("wave"), lives=lives, victory=won,
                          return_value=total, steps=steps, stats=info.get("ep_stats", {}),
                          waves={str(w): dict(row) for w, row in sorted(rows.items())})
            record.update(advanced_towers_peak=advanced_peak,advanced_towers_distinct=len(advanced_seen),
                          advanced_tower_seconds=advanced_seconds,advanced_skill_purchases=len(record['stats'].get('tower_skills',[])))
            record.update({key: info[key] for key in ("level_idx", "level_mode", "difficulty", "schema", "game_seed") if key in info})
            record.update(record_context or {})
            record["wave_total"] = info.get("wave_total", base.o.get("wave_total", 1))
            record["wave_progress"] = wave_progress(record["wave"], record["wave_total"], won)
            record["life_ratio"] = min(1.0, max(0.0, lives / max(1, info.get("lives0", getattr(base, "lives0", 20)))))
            records.append(record)
            print("eval %s episode %d/%d: steps=%d return=%+.2f outcome=%s lives=%d wave=%s end=%s" %
                  (mode, ep, episodes, steps, total, info.get("outcome"), lives,
                   info.get("wave"), info.get("end_reason")), flush=True)
            stats = record["stats"]
            print("    actions=%s builds=%s gold_end=%s leaks_by_wave=%s" %
                  (stats.get("actions", {}), stats.get("build_kinds", {}),
                   stats.get("gold_end"), stats.get("leaks_by_wave", {})), flush=True)
            for w, row in sorted(rows.items()):
                if w < diagnostic_wave:
                    continue
                print("    diag w%d: gold_mean=%.1f gold_max=%d steps=%d spend_available=%d "
                      "spend/wait/other=%d/%d/%d spend_success=%d/%d reinf_available=%d reinf_success=%d/%d" %
                      (w, row["gold_sum"] / row["steps"], row["gold_max"], row["steps"],
                       row["spend_available_steps"], row["spend_available_spend"],
                       row["spend_available_wait"], row["spend_available_other"],
                       row["spend_success"], row["spend_attempts"], row["reinf_available_steps"],
                       row["reinf_success"], row["reinf_attempts"]), flush=True)
            if output:
                os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    winners = [r for r in records if r["victory"]]
    summary = dict(mode=mode, episodes=episodes, victories=len(winners),
                   win_rate=len(winners) / episodes,
                   mean_win_lives=sum(r["lives"] for r in winners) / len(winners) if winners else 0.0,
                   mean_wave_progress=sum(r["wave_progress"] for r in records) / episodes,
                   mean_life_ratio=sum(r["life_ratio"] for r in records) / episodes,
                   mean_return=sum(r["return_value"] for r in records) / episodes)
    print("eval %s summary: victories=%d/%d win_rate=%.1f%% mean_win_lives=%.2f mean_return=%+.2f" %
          (mode, len(winners), episodes, summary["win_rate"] * 100,
           summary["mean_win_lives"], summary["mean_return"]), flush=True)
    return summary


def evaluate_modes(model, env, episodes, mode="both", **kwargs):
    modes = (True, False) if mode == "both" else (mode == "deterministic",)
    return {("deterministic" if deterministic else "stochastic"):
            evaluate(model, env, episodes, deterministic=deterministic, **kwargs)
            for deterministic in modes}


def best_score(summary):
    return summary["win_rate"], summary["mean_win_lives"]


def train_in_segments(model, steps, interval, callbacks, assess, reset_after_evaluation=True):
    target = model.num_timesteps + steps
    rollout = model.n_steps * model.n_envs
    chunk = ((interval + rollout - 1) // rollout) * rollout
    while model.num_timesteps < target:
        model.learn(total_timesteps=min(chunk, target - model.num_timesteps),
                    callback=callbacks, progress_bar=False, reset_num_timesteps=False)
        assess("periodic")
        if reset_after_evaluation:
            model.set_env(model.get_env(), force_reset=True)
