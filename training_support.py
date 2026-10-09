"""Shared training recipe, normalization, diagnostics and checkpoint contract."""
from collections import Counter, defaultdict, deque
import json
from pathlib import Path


def selection_score(summary):
    # Win rate stays primary. Progress resolves all-zero-win ties without
    # pretending that starting a wave means its enemies have been cleared.
    return (summary["win_rate"], summary.get("mean_wave_progress", 0), summary.get("mean_life_ratio", 0))


class LinearRate:
    def __init__(self, initial, final):
        self.initial, self.final, self.remaining = initial, final, 1.0

    def __call__(self, _):
        return self.final + (self.initial - self.final) * self.remaining


def normalize_env(env, args):
    from stable_baselines3.common.vec_env import VecNormalize
    if args.resume:
        state = Path(args.resume).with_suffix(".vecnormalize.pkl")
        if args.norm_reward:
            if not state.is_file():
                raise ValueError("Missing matching reward statistics: " + str(state))
            wrapped = VecNormalize.load(state, env)
            if wrapped.norm_obs or not wrapped.norm_reward:
                raise ValueError("Expected reward-only normalization")
            if wrapped.gamma != args.gamma:
                raise ValueError("Resume gamma differs from reward normalization statistics")
            wrapped.training = True
            return wrapped
    if args.norm_reward:
        return VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10, gamma=args.gamma)
    return env


def build_model(args, env):
    from sb3_contrib import MaskablePPO
    algorithm=MaskablePPO
    if getattr(args,'imitation_arm',None)=='anchored':
        from candidate_bc import BCPPO
        algorithm=BCPPO
    schedule = LinearRate(args.lr, args.lr_final)
    options = dict(n_steps=args.rollout_steps, batch_size=min(args.batch_size, args.rollout_steps * env.num_envs),
                   n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
                   learning_rate=schedule, ent_coef=args.ent_coef, target_kl=args.target_kl)
    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device=args.device, **options)
        if args.env_version.startswith('candidate-'):
            model.policy.type_balance_alpha=args.type_balance_alpha
            model.policy_kwargs['type_balance_alpha']=args.type_balance_alpha
            for name in ('advanced_tower_bias','tesla_route_bias','tower_skill_bias','tower_focus_bias','core_investment_bias'):
                value=getattr(args,name,0.)
                setattr(model.policy,name,value);model.policy_kwargs[name]=value
        if args.net_arch is not None and model.policy_kwargs.get("net_arch") != dict(pi=args.net_arch, vf=args.net_arch):
            raise ValueError("Cannot resize a resumed model; start a fresh run to compare architectures")
    else:
        candidate=args.env_version.startswith('candidate-')
        arch = args.net_arch or ([128,128] if candidate else [256, 256])
        from candidate_policy import CandidatePolicy
        if args.env_version in ('candidate-v9','candidate-v10'):
            from candidate_policy import LayoutCandidatePolicy
            CandidatePolicy=LayoutCandidatePolicy
        if getattr(args,'policy_version','layout')=='route-v1':
            from route_policy import RouteCandidatePolicy
            CandidatePolicy=RouteCandidatePolicy
        if getattr(args,'policy_version','layout')=='route-v2-invest':
            from investment_policy import StepwiseRoutePolicy
            CandidatePolicy=StepwiseRoutePolicy
        policy_options=dict(net_arch=dict(pi=arch,vf=arch))
        if candidate: policy_options['type_balance_alpha']=args.type_balance_alpha
        if args.env_version in ('candidate-v9','candidate-v10'):
            for name in ('advanced_tower_bias','tesla_route_bias','tower_skill_bias','tower_focus_bias','core_investment_bias'):
                policy_options[name]=getattr(args,name,0.)
        model = algorithm(CandidatePolicy if candidate else "MultiInputPolicy", env, **options,
                            policy_kwargs=policy_options,
                            seed=args.seed, device=args.device, verbose=1)
        if getattr(args,'bc_init',None):
            # Load parameters into the newly configured algorithm. Keep fresh
            # rollout counters, reward statistics and the requested logit bias.
            model.set_parameters(args.bc_init,exact_match=True,device=model.device)
    return model, schedule


def contract(args):
    return dict(reward_profile=args.reward_profile, norm_reward=args.norm_reward, gamma=args.gamma)


def validate_checkpoint(args, schema):
    for path in (args.resume, args.eval,getattr(args,'bc_init',None)):
        if not path:
            continue
        path = Path(path)
        metadata = path.with_suffix(".json")
        if not metadata.exists():
            raise ValueError("Model metadata is missing: " + str(metadata))
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if getattr(args,'policy_version','layout')!=saved.get('config',{}).get('policy_version','layout'):
            raise ValueError('Checkpoint policy architecture differs; start a fresh run')
        if saved.get("schema") != schema:
            raise ValueError("Model observation/action schema does not match")
        if args.loadout != saved.get('config', {}).get('loadout', 'original'):
            raise ValueError('Checkpoint training loadout differs; start a new run for full upgrades and Elora')
        if path==getattr(args,'bc_init',None):
            if saved.get('timesteps')!=0 or saved.get('config',{}).get('bc_pretrain_epochs',0)<=0:
                raise ValueError('--bc-init requires a BC-pretrained checkpoint at step 0')
            if args.net_arch is not None and saved.get('architecture')!=dict(pi=args.net_arch,vf=args.net_arch):
                raise ValueError('BC initialization architecture differs')
        if args.resume:
            if saved.get('config',{}).get('imitation_arm')=='anchored':
                raise ValueError('Generic --resume would drop the continuous BC loss; anchored comparison checkpoints are evaluation-only here')
            previous = saved.get("training_contract", dict(reward_profile="legacy", norm_reward=False,
                                                           gamma=saved.get("config", {}).get("gamma", 0.999)))
            if previous != contract(args):
                raise ValueError("Resume reward/normalization/gamma must match the checkpoint: " + str(previous))


def save_checkpoint(model, env, path, args, schema, config, summary=None):
    path = Path(path)
    model.save(path)
    if args.norm_reward:
        env.save(path.with_suffix(".vecnormalize.pkl"))
    metadata = dict(schema=schema, config=config, training_contract=contract(args),
                    architecture=model.policy_kwargs.get("net_arch"), summary=summary,
                    timesteps=model.num_timesteps)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def verify_snapshot_hashes(hashes, expected=None):
    if not hashes or any(not value for value in hashes) or len(set(hashes)) != 1:
        raise ValueError("Training/evaluation save snapshots differ; use the same unchanged slot for all instances")
    if expected is not None and hashes[0] != expected:
        raise ValueError("Save snapshot differs from the checkpoint")
    return hashes[0]


def make_callback(args, run, schedule, start_step):
    from stable_baselines3.common.callbacks import BaseCallback
    from ppo_evaluation import wave_progress

    class Diagnostics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = defaultdict(lambda: deque(maxlen=100))
            self.sample_steps = Counter()
            self.decision_totals=Counter()
            self.decision_batch=None
            self.decision_updates=0

        def _on_rollout_start(self):
            # The previous buffer is still available after its PPO updates.
            # Measure updates only where the old policy had a real choice.
            if args.env_version.startswith('candidate-') and self.decision_batch is not None:
                import torch
                batch=self.decision_batch
                choices=batch.action_masks.sum(-1)>1
                if choices.any():
                    with torch.no_grad():
                        _,log_prob,_=self.model.policy.evaluate_actions(batch.observations,batch.actions.long().flatten(),batch.action_masks)
                        difference=log_prob[choices]-batch.old_log_prob[choices]
                        kl=((difference.exp()-1)-difference).mean()
                        self.logger.record('decision/approx_kl',float(kl))
                        clip=self.model.clip_range(self.model._current_progress_remaining)
                        self.logger.record('decision/clip_fraction',float(((difference.exp()-1).abs()>clip).float().mean()))
                        self.decision_updates+=1
            self.decision_totals.clear()
            remaining = max(0, 1 - (self.num_timesteps - start_step) / args.steps)
            self.model.ent_coef = args.ent_final + (args.ent_coef - args.ent_final) * remaining

        def _on_step(self):
            candidate_scores=None
            if args.env_version.startswith('candidate-'):
                import torch
                with torch.no_grad():
                    candidate_scores=self.model.policy.candidate_logits(self.locals['obs_tensor'],self.locals.get('action_masks')).cpu().numpy()
                    distribution=self.model.policy.get_distribution(self.locals['obs_tensor'],self.locals.get('action_masks'))
                    import numpy as np
                    masks=np.asarray(self.locals['action_masks'],dtype=bool)
                    decision=masks.sum(-1)>1
                    if hasattr(self.model.policy,'investment_state'):
                        investment=self.model.policy.investment_state(self.locals['obs_tensor'],self.locals['action_masks'])
                        selected=torch.as_tensor(self.locals['actions'],device=self.model.device,dtype=torch.long).flatten()
                        from support_features import F
                        c=self.locals['obs_tensor']['candidates']
                        waited=c[torch.arange(len(selected),device=c.device),selected,F['type_wait']]>.5
                        advanced=investment['match'].gather(1,selected[:,None]).squeeze(1)
                        for name,flag in (('ready',investment['ready']),('ready_wait',investment['ready']&waited),
                                          ('ready_progress',investment['ready']&advanced),
                                          ('bootstrap',investment['bootstrap']),('bootstrap_wait',investment['bootstrap']&waited)):
                            self.decision_totals['investment/'+name]+=int(flag.sum())
                    self.decision_totals['samples']+=len(decision)
                    self.decision_totals['choices']+=int(decision.sum())
                    if decision.any():
                        entropy=distribution.entropy().cpu().numpy()
                        self.decision_totals['entropy']+=float(entropy[decision].sum())
                        from candidate_features import TYPES
                        probabilities=distribution.distribution.probs.cpu().numpy()
                        types=self.locals['obs_tensor']['candidates'][:,:,1:1+len(TYPES)].cpu().numpy()
                        for j,typ in enumerate(TYPES):
                            self.decision_totals['mass/'+typ]+=float((probabilities*types[:,:,j])[decision].sum())
            for rank, info in enumerate(self.locals.get("infos", [])):
                archive=info.get('success_replay_manifest')
                pool=getattr(getattr(self.model,'demo_data',None),'success_pool',None)
                if archive and pool is not None and pool.add(archive):
                    memory=dict(timesteps=self.num_timesteps,archive=archive,**pool.summary())
                    (run/'success_memory.json').write_text(json.dumps(memory,indent=2),encoding='utf8')
                    with (run/'success_memory.jsonl').open('a',encoding='utf8') as handle:handle.write(json.dumps(memory)+'\n')
                    print('SUCCESS_REPLAY_ADDED',archive,flush=True)
                if info.get('candidate_catalog') and candidate_scores is not None:
                    from candidate_diagnostics import ranked_event
                    catalog=info['candidate_catalog']
                    record=ranked_event(info['candidate_event'],catalog,candidate_scores[rank],
                                        int(self.locals['actions'][rank]))
                    for key in ('route_predictions','route_prediction_waves'):
                        if key in info['candidate_event']:
                            record[key]=info['candidate_event'][key]
                    record.update(worker=rank,timesteps=self.num_timesteps,level=info.get('level_idx'))
                    with (run/'candidate_events.jsonl').open('a',encoding='utf-8') as handle:
                        handle.write(json.dumps(record)+'\n')
                level = info.get("level_idx")
                if level is not None:
                    self.sample_steps[level] += 1
                if "episode" not in info:
                    continue
                won = info.get("outcome") == "victory"
                progress = wave_progress(info.get("wave"), info.get("wave_total"), won)
                self.history[level].append((float(won), progress))
                stats = info.get("ep_stats", {})
                row = dict(worker=rank, timesteps=self.num_timesteps, wave_progress=progress,
                           **{key: info.get(key) for key in ("episode", "level_idx", "level_mode", "outcome", "lives", "lives0", "wave", "wave_total")},
                           gold_end=stats.get("gold_end"), actions=stats.get("actions"),
                           build_kinds=stats.get("build_kinds"), unblock_clicks=stats.get('unblock_clicks',0),
                           boss_finish_clicks=stats.get('boss_finish_clicks',0),
                           replacements=stats.get('replacements',[]),
                           tower_skills=stats.get('tower_skills',[]),
                           rain_cooldown_leaks=stats.get('rain_cooldown_leaks',0),
                           planning=stats.get('planning'),rain_outcomes=stats.get('rain_outcomes',[]),
                           game_seed=info.get('game_seed'),
                           reward_parts=stats.get("reward_parts"))
                for key in ('leaks_by_wave','leaks_by_path','leak_counts_by_path','leaks_path_partial','route_development',
                            'leak_cohorts','leak_life_cohorts','birth_attribution_partial','birth_cohorts','tower_damage'):
                    if key in stats:
                        row[key]=stats[key]
                for key in ('opening_bonus', 'experiment_stage'):
                    if key in info:
                        row[key] = info[key]
                print(row, flush=True)
                with (run / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            return True

        def _on_rollout_end(self):
            if args.env_version.startswith('candidate-'):
                import numpy as np
                buffer=self.model.rollout_buffer
                state=np.random.get_state()
                try: self.decision_batch=next(buffer.get(batch_size=min(512,buffer.buffer_size*buffer.n_envs)))
                finally: np.random.set_state(state)
            schedule.remaining = max(0, 1 - (self.num_timesteps - start_step) / args.steps)
            total = max(1, sum(self.sample_steps.values()))
            for level, count in self.sample_steps.items():
                self.logger.record(f"level/{level:02d}/step_share", count / total)
            for level, values in self.history.items():
                self.logger.record(f"level/{level:02d}/win_rate", sum(x[0] for x in values) / len(values))
                self.logger.record(f"level/{level:02d}/wave_progress", sum(x[1] for x in values) / len(values))
            self.logger.record("train/entropy_coefficient", self.model.ent_coef)
            n=max(1,self.decision_totals['choices'])
            self.logger.record('decision/fraction',self.decision_totals['choices']/max(1,self.decision_totals['samples']))
            self.logger.record('decision/entropy',self.decision_totals['entropy']/n)
            for key,value in self.decision_totals.items():
                if key.startswith('mass/'): self.logger.record('decision/'+key,value/n)
            if hasattr(self.model.policy,'investment_state'):
                ready=max(1,self.decision_totals['investment/ready'])
                bootstrap=max(1,self.decision_totals['investment/bootstrap'])
                self.logger.record('investment/ready_steps',self.decision_totals['investment/ready'])
                self.logger.record('investment/ready_wait_fraction',self.decision_totals['investment/ready_wait']/ready)
                self.logger.record('investment/ready_progress_fraction',self.decision_totals['investment/ready_progress']/ready)
                self.logger.record('investment/bootstrap_wait_fraction',self.decision_totals['investment/bootstrap_wait']/bootstrap)

    return Diagnostics()
