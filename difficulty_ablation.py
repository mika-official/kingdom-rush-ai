"""Matched single-level training with opening-gold assistance and native evaluation."""
from functools import partial
from pathlib import Path
import hashlib
import json

from kr_planning import KRPlanningEnv


class AssistedPlanningEnv(KRPlanningEnv):
    def __init__(self, **kwargs):
        self.opening_bonus = 0
        self.experiment_stage = 0
        super().__init__(**kwargs)

    def set_assistance(self, bonus, stage):
        if not isinstance(bonus, int) or bonus < 0:
            raise ValueError('Opening bonus must be a nonnegative integer')
        self.opening_bonus, self.experiment_stage = bonus, stage

    def _start_play(self):
        obs = super()._start_play()
        if self.opening_bonus:
            before = obs['gold']
            obs = self.client.call('(function() game.store.player_gold = game.store.player_gold + %d; return ENV.obs_json() end)()' % self.opening_bonus)
            if obs['gold'] != before + self.opening_bonus or obs['wave'] != 0:
                raise RuntimeError('Opening assistance did not apply exactly at reset')
            self.o = self.prev_o = obs
            self.ep['gold_max'] = obs['gold']
            self.invalidate_candidates()
        return obs

    def _info(self, obs):
        info = super()._info(obs)
        info.update(opening_bonus=self.opening_bonus, experiment_stage=self.experiment_stage)
        return info


def worker(port, options, directory):
    from stable_baselines3.common.monitor import Monitor
    directory.mkdir(parents=True, exist_ok=True)
    return Monitor(AssistedPlanningEnv(port=port, backup_dir=directory/'save_backups', **options),
                   filename=str(directory/'monitor'),
                   info_keywords=('level_idx', 'level_mode', 'outcome', 'opening_bonus', 'experiment_stage'))


def policy_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_ablation(args, modes, run, schema, config):
    import torch
    from stable_baselines3.common.logger import configure
    from safe_vec_env import SafeSubprocVecEnv
    from parallel_train import env_options
    from training_support import normalize_env, build_model, make_callback, save_checkpoint, verify_snapshot_hashes
    from ppo_evaluation import evaluate
    torch.set_num_threads(1)
    options = env_options(args, modes)
    options.pop('env_version')
    options['focus_every'] = 0
    bonuses = [4000, 2000, 1000, 0] if args.ablation_arm == 'assisted' else [0]*4
    config.update(opening_bonuses=bonuses, stage_steps=args.steps//4,
                  evaluation_seed=100042, stage_boundary_reset=True)
    env = evaluation = logger = model = None
    results = []
    try:
        ports = args.ports or [args.port]
        env = SafeSubprocVecEnv([partial(worker, port, options, run/f'worker_{i}') for i, port in enumerate(ports)], start_method='spawn')
        evaluation = AssistedPlanningEnv(port=args.eval_port, backup_dir=run/'eval_backups', **options)
        config['snapshot_sha256'] = verify_snapshot_hashes(env.env_method('snapshot_hash') + [evaluation.snapshot_hash()])
        env.seed(args.seed)
        env = normalize_env(env, args)
        model, rate = build_model(args, env)
        config['initial_policy_sha256'] = policy_hash(model)
        config['effective_net_arch'] = model.policy_kwargs['net_arch']
        (run/'config.json').write_text(json.dumps(dict(config=config, schema=schema), indent=2), encoding='utf8')
        logger = configure(str(run/'metrics'), ['stdout', 'csv'])
        model.set_logger(logger)
        callback = make_callback(args, run, rate, 0)

        def assess(stage, episodes):
            # This environment never receives assistance. evaluation_rng restores
            # all Python/NumPy/Torch RNG state so evaluation cannot steer training.
            summary = evaluate(model, evaluation, episodes, deterministic=False,
                               diagnostic_wave=100, seed=config['evaluation_seed'],
                               output=str(run/'evaluation.jsonl'), label=f'stage={stage}',
                               record_context=dict(stage=stage, arm=args.ablation_arm, opening_bonus=0))
            results.append(dict(stage=stage, timesteps=model.num_timesteps, summary=summary))
            (run/'results.json').write_text(json.dumps(dict(config=config, evaluations=results), indent=2), encoding='utf8')

        assess(0, args.eval_episodes)
        for stage, bonus in enumerate(bonuses, 1):
            env.env_method('set_assistance', bonus, stage)
            # Both arms reset at the same step boundaries; no episode carries
            # bonus gold into the stage explicitly designated unassisted.
            model.set_env(env, force_reset=True)
            print(f'ABLATION stage={stage} opening_bonus={bonus} target_steps={stage*args.steps//4}', flush=True)
            model.learn(args.steps//4, callback=callback, reset_num_timesteps=False)
            save_checkpoint(model, env, run/f'stage_{stage}', args, schema, config)
            assess(stage, max(10, args.eval_episodes) if stage == 4 else args.eval_episodes)
        save_checkpoint(model, env, run/'final', args, schema, config)
    except BaseException as exc:
        (run/'failure.json').write_text(json.dumps(dict(error=repr(exc), timesteps=getattr(model, 'num_timesteps', 0)), indent=2), encoding='utf8')
        if model is not None:
            save_checkpoint(model, env, run/'interrupted', args, schema, config)
        raise
    finally:
        if env is not None:
            env.close()
        if evaluation is not None:
            evaluation.close()
        if logger is not None:
            logger.close()
