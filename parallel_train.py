"""Spawn one Gym worker per game; all workers contribute to one PPO policy."""
from functools import partial
import json


def env_options(args, modes):
    return dict(levels=args.levels, modes=modes, difficulty=args.difficulty,
                ticks_per_step=args.ticks_per_step, max_ticks=args.max_ticks,
                curriculum_every=args.curriculum_every,
                reward_profile=args.reward_profile, env_version=args.env_version, loadout=args.loadout,
                spend_before_wait=args.spend_before_wait, freeze_build_last_wave=False,
                core_investment_bias=getattr(args,'core_investment_bias',0.),
                fixed_game_seed=getattr(args,'fixed_game_seed',None),
                demo_start_dataset=getattr(args,'demo_start_dataset',None),
                demo_start_probability=getattr(args,'demo_start_probability',0.),
                demo_start_min_wave=getattr(args,'demo_start_min_wave',4),
                demo_start_max_wave=getattr(args,'demo_start_max_wave',10),
                success_replay_dir=getattr(args,'success_replay_dir',None) if getattr(args,'success_replay',False) else None,
                success_replay_source='training',success_replay_levels=args.levels)


def make_worker(port, options, directory, rank):
    import multiprocessing
    import signal
    if multiprocessing.current_process().name != "MainProcess":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    from kr_macro import KRMacroEnv
    from stable_baselines3.common.monitor import Monitor
    directory.mkdir(parents=True, exist_ok=True)
    if multiprocessing.current_process().name != 'MainProcess':
        import sys
        sys.stdout=sys.stderr=(directory/'worker.log').open('a',encoding='utf8',buffering=1)
    env = make_environment(port=port, backup_dir=directory / "save_backups", **options)
    return Monitor(env, filename=str(directory / "monitor"),
                   info_keywords=("level_idx", "level_mode", "outcome"))


def make_environment(env_version='macro-v1', **options):
    from kr_macro import KRMacroEnv
    from kr_tactical import KRTacticalEnv
    from kr_replacement import KRReplacementEnv
    from kr_candidates import KRCandidateEnv
    from kr_planning import KRPlanningEnv
    from kr_route_planning import KRRoutePlanningEnv
    from kr_light import KRLightEnv
    from kr_layout import KRLayoutEnv
    from kr_support import KRSupportEnv
    from kr_strategy import KRStrategyEnv
    cls = {'macro-v1':KRMacroEnv,'macro-v2-tactical':KRTacticalEnv,'macro-v3-replacement':KRReplacementEnv,
           'macro-v4-strategy':KRStrategyEnv,'candidate-v5':KRCandidateEnv,'candidate-v6':KRPlanningEnv,
           'candidate-v7':KRRoutePlanningEnv,'candidate-v8':KRLightEnv,'candidate-v9':KRLayoutEnv,'candidate-v10':KRSupportEnv}[env_version]
    core_bias=options.pop('core_investment_bias',0.)
    dataset=options.pop('demo_start_dataset',None);probability=options.pop('demo_start_probability',0.)
    min_wave=options.pop('demo_start_min_wave',4);max_wave=options.pop('demo_start_max_wave',10)
    fixed_seed=options.pop('fixed_game_seed',None)
    replay_dir=options.pop('success_replay_dir',None);replay_source=options.pop('success_replay_source','training')
    replay_levels=options.pop('success_replay_levels',None)
    if env_version in ('candidate-v6','candidate-v7','candidate-v8','candidate-v9','candidate-v10'):options['fixed_game_seed']=fixed_seed
    if env_version in ('candidate-v9','candidate-v10'):options['core_investment']=bool(core_bias)
    env=cls(**options)
    if probability:
        from demo_start import DemonstrationStart
        try:env=DemonstrationStart(env,dataset,probability,min_wave,max_wave)
        except BaseException:env.close();raise
    if replay_dir:
        from success_memory import VictoryRecorder
        env=VictoryRecorder(env,replay_dir,source=replay_source,allowed_levels=replay_levels)
    return env


def run_parallel(args, modes, run, schema, config):
    config.setdefault('loadout', args.loadout)
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from safe_vec_env import SafeSubprocVecEnv
    from stable_baselines3.common.logger import configure
    from kr_macro import KRMacroEnv
    from train_all import evaluate_suite
    from ppo_evaluation import train_in_segments
    from training_support import (normalize_env, build_model, save_checkpoint,
                                  make_callback, selection_score, verify_snapshot_hashes)
    import torch
    torch.set_num_threads(1)
    options = env_options(args, modes)
    if args.env_version.startswith('candidate-'): options['focus_every']=args.focus_every
    ports = args.ports or [args.port]
    env, evaluation, training_logger = None, None, None
    try:
        if args.smoke:
            # Verify every process and all requested cases before long training.
            for rank, port in enumerate(ports):
                worker = make_worker(port, dict(options, levels=args.eval_levels), run / f"worker_{rank}", rank)
                try:
                    rows = []
                    for pair in worker.unwrapped.schedule.pairs:
                        obs, info = worker.reset(seed=args.seed + rank, options={"level_mode": pair})
                        if args.loadout == 'full_elora':
                            base = worker.unwrapped
                            levels = base.client.call('ENV.meta_info()').get('levels', {})
                            assert all(levels.get(k) == 5 for k in ('archers','barracks','mages','engineers','rain','reinforcements')), levels
                            if not base.o.get('locked_hero'):
                                assert (base.o.get('hero') or {}).get('name') == 'hero_elora', base.o.get('hero')
                        assert worker.observation_space.contains(obs)
                        mask = worker.unwrapped.action_masks()
                        action = next((i for i in range(2, 10) if mask[i]), 0)
                        obs, _, _, _, info = worker.step(action)
                        assert worker.observation_space.contains(obs)
                        if any(not result.get("ok") for _, result in info.get("results", [])):
                            raise RuntimeError(f"Smoke action rejected on port {port}")
                        rows.append(dict(port=port, level=pair[0], mode=pair[1],
                                         game_seed=info.get('game_seed'),demonstration_start=info.get('demonstration_start')))
                        (run / f"smoke_{port}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                        print("smoke", rows[-1], flush=True)
                finally:
                    worker.close()
            return
        if args.eval_port is not None or args.eval or args.baseline:
            evaluation = make_environment(port=args.eval_port or args.port, backup_dir=run / "eval_backups",
                                    **dict(options, levels=args.eval_levels, curriculum_every=0,demo_start_probability=0.,success_replay_source='evaluation'))
        if not (args.eval or args.baseline):
            factories = [partial(make_worker, port, options, run / f"worker_{rank}", rank)
                         for rank, port in enumerate(ports)]
            env = SafeSubprocVecEnv(factories, start_method="spawn") if args.ports else DummyVecEnv(factories)
            env.seed(args.seed)
        hashes = env.env_method("snapshot_hash") if env is not None else []
        if evaluation is not None:
            hashes.append(evaluation.snapshot_hash())
        expected = None
        if args.resume or args.eval or getattr(args,'bc_init',None):
            previous = json.loads((args.resume or args.eval or args.bc_init).with_suffix(".json").read_text(encoding="utf-8"))
            expected = previous.get("config", {}).get("snapshot_sha256")
        config["snapshot_sha256"] = verify_snapshot_hashes(hashes, expected)
        (run / "config.json").write_text(json.dumps(dict(config=config, schema=schema), indent=2), encoding="utf-8")
        if args.baseline:
            if args.env_version.startswith('candidate-'):
                from candidate_baseline import CandidateBaseline
                from planning_baseline import PlanningBaseline
                model=(PlanningBaseline if args.env_version in ('candidate-v6','candidate-v7') else CandidateBaseline)(evaluation)
            else:
                from macro_baseline import MacroBaseline
                model = MacroBaseline(evaluation)
        elif args.eval:
            model = MaskablePPO.load(args.eval, device=args.device)
            if args.env_version.startswith('candidate-'):
                model.policy.type_balance_alpha=args.type_balance_alpha
                model.policy_kwargs['type_balance_alpha']=args.type_balance_alpha
                for name in ('advanced_tower_bias','tesla_route_bias','tower_skill_bias','tower_focus_bias','core_investment_bias'):
                    value=getattr(args,name,0.)
                    setattr(model.policy,name,value);model.policy_kwargs[name]=value
        else:
            env = normalize_env(env, args)
            model, rate_schedule = build_model(args, env)
            config["effective_net_arch"] = model.policy_kwargs["net_arch"]
            (run / "config.json").write_text(json.dumps(dict(config=config, schema=schema), indent=2), encoding="utf-8")
            training_logger = configure(str(run / "metrics"), ["stdout", "csv"] + (["tensorboard"] if args.tensorboard else []))
            model.set_logger(training_logger)

        demonstrations=None;success_pool=None
        if getattr(args,'imitation_arm',None):
            from candidate_bc import Demonstrations,pretrain
            from difficulty_ablation import policy_hash
            from ppo_evaluation import evaluation_rng
            config['initial_policy_sha256']=policy_hash(model)
            demonstrations=Demonstrations(args.bc_dataset,args.seed,expected_schema=args.env_version,sampling=args.bc_sampling,split=args.bc_split)
            if args.env_version=='candidate-v10' and args.imitation_arm in ('anchored','pretrained'):
                data_maps={r['level'] for r in demonstrations.train+demonstrations.validation}
                train_maps={r['level'] for r in demonstrations.train if not r['correction']}
                if set(args.levels)-train_maps:
                    raise ValueError('BC data is missing training maps: '+str(sorted(set(args.levels)-train_maps)))
                if data_maps & set(args.holdout_levels):
                    raise ValueError('BC data contains held-out evaluation maps')
            if args.core_investment_bias and not (demonstrations.metadata['train_core_goal_samples'] and demonstrations.metadata['validation_core_goal_samples']):
                raise ValueError('Core investment requires demonstrations collected with --core-investment in both seed splits')
            config.update(demonstrations=demonstrations.metadata,
                          bc_coefficient=args.bc_coef if args.imitation_arm=='anchored' else 0.,
                          bc_pretrain_epochs=previous.get('config',{}).get('bc_pretrain_epochs',0) if args.bc_init else (args.bc_epochs if args.imitation_arm in ('anchored','pretrained') else 0),
                          bc_initialization=str(args.bc_init.resolve()) if args.bc_init else None)
            if args.imitation_arm in ('anchored','pretrained'):
                if args.bc_init:
                    print('BC_INIT_LOADED',str(args.bc_init.resolve()),flush=True)
                else:
                    with evaluation_rng(args.seed):
                        pretrain(model,demonstrations,args.bc_epochs,args.bc_lr,run,args.bc_batch_size)
                if args.imitation_arm=='anchored':
                    model.demo_data=demonstrations;model.bc_coef=args.bc_coef;model.bc_batch_size=args.bc_batch_size
                    if args.success_replay:
                        from success_memory import VictoryPool
                        success_pool=VictoryPool(args.env_version,args.success_replay_dir,args.levels,args.difficulty,
                            args.loadout,args.fixed_game_seed,seed=args.seed)
                        success_pool.scan()
                        demonstrations.success_pool=success_pool;demonstrations.success_ratio=args.success_replay_ratio
                        config['success_memory']=success_pool.summary()
            (run/'config.json').write_text(json.dumps(dict(config=config,schema=schema),indent=2),encoding='utf8')

        def save(name, summary=None):
            save_checkpoint(model, env, run / name, args, schema, config, summary)

        best = {}

        def evaluate_all():
            results = {}
            policies = ("deterministic", "stochastic") if args.eval_mode == "both" else (args.eval_mode,)
            for policy in policies:
                results[policy] = evaluate_suite(model, evaluation, run / "evaluation.jsonl", args.eval_episodes,
                                                args.seed+100000 if args.imitation_arm else args.seed, deterministic=policy == "deterministic",
                                                holdout_levels=args.holdout_levels)
                print("evaluation", policy, results[policy]["splits"], flush=True)
            return results

        def assess(label):
            save("latest")
            if demonstrations is not None:
                metrics=demonstrations.measure(model.policy,args.bc_batch_size)
                row=dict(timesteps=model.num_timesteps,arm=args.imitation_arm,metrics=metrics)
                with (run/'bc_retention.jsonl').open('a',encoding='utf8') as handle:handle.write(json.dumps(row)+'\n')
                print('BC_RETENTION',json.dumps(row),flush=True)
            result = evaluate_all()
            if success_pool is not None:
                success_pool.scan();config['success_memory']=success_pool.summary()
                (run/'success_memory.json').write_text(json.dumps(config['success_memory'],indent=2),encoding='utf8')
            (run / f"evaluation_{model.num_timesteps}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            for policy, report in result.items():
                score = selection_score(report["splits"]["train"])
                if policy not in best or score > best[policy]:
                    best[policy] = score
                    save("best_" + policy, report)
                    if policy == ("stochastic" if args.eval_mode == "both" else args.eval_mode):
                        save("best", report)

        if args.eval or args.baseline:
            result = evaluate_all()
            (run / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            try:
                callback = make_callback(args, run, rate_schedule, model.num_timesteps)
                if args.imitation_arm:
                    save('bc_pretrained' if args.imitation_arm in ('anchored','pretrained') else 'initial')
                    assess('before_ppo')
                    if args.bc_verify_opening and args.imitation_arm in ('anchored','pretrained'):
                        records=[json.loads(line) for line in (run/'evaluation.jsonl').read_text(encoding='utf8').splitlines()]
                        initial=[r for r in records if r['timesteps']==0 and r['level_idx'] in args.levels]
                        from bc_opening_gate import opening_gate
                        revision='behavior-v2' if args.env_version=='candidate-v10' and len(args.levels)>1 else 'strict-v1'
                        gate=opening_gate(initial,demonstrations.metadata['opening_profiles'],revision=revision);passed=gate['passed']
                        (run/'bc_opening_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf8')
                        if not passed:raise ValueError('BC opening gate failed; PPO was not started. See bc_opening_gate.json')
                if args.eval_every:
                    train_in_segments(model, args.steps, args.eval_every, callback, assess, reset_after_evaluation=False)
                else:
                    model.learn(args.steps, callback=callback, reset_num_timesteps=False)
                save("final")
            except KeyboardInterrupt:
                save("interrupted")
                raise
            except Exception as exc:
                (run/'failure.json').write_text(json.dumps(dict(error=repr(exc),timesteps=model.num_timesteps),indent=2),encoding='utf8')
                try: save('failed')
                except Exception as save_error:
                    print('Failed to save emergency checkpoint:',repr(save_error),flush=True)
                raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if evaluation is not None:
                    evaluation.close()
            finally:
                if training_logger is not None:
                    training_logger.close()
