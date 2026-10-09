"""Train map-independent PPO; campaign-only with held-out levels by default."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from kr_macro import KRMacroEnv, MODE_IDS, MACROS, ENEMY_TYPES


def parse_levels(value):
    result = set()
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            result.update(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise argparse.ArgumentTypeError("Use levels like 1-26 or 1,4,9-12")
    if not result or not result <= set(range(1, 27)):
        raise argparse.ArgumentTypeError("Levels must be in 1..26")
    return sorted(result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--ports", help="comma-separated ports for parallel training workers")
    parser.add_argument("--eval-port", type=int, help="dedicated game for evaluation")
    parser.add_argument("--levels", type=parse_levels, default=list(range(1, 27)))
    parser.add_argument("--holdout-levels", type=lambda s: [] if s == "none" else parse_levels(s), default=None)
    parser.add_argument("--env-version", choices=("macro-v1", "macro-v2-tactical", "macro-v3-replacement", "macro-v4-strategy", "candidate-v5", "candidate-v6", "candidate-v7", "candidate-v8", "candidate-v9", "candidate-v10"), default=None)
    parser.add_argument('--policy-version',choices=('layout','route-v1','route-v2-invest'),default=None,help='versioned v10 route and incremental-investment policies; fresh BC weights required')
    parser.add_argument("--type-balance-alpha",type=float,default=None,help="candidate policy: subtract alpha*log(legal candidates per type), 0..1")
    parser.add_argument('--advanced-tower-bias',type=float,default=None,help='v9 upgrade/advanced-skill logit prior, 0..4')
    parser.add_argument('--tesla-route-bias',type=float,default=None,help='v9 Tesla route-coverage logit prior, 0..4')
    parser.add_argument('--tower-skill-bias',type=float,default=None,help='v9 skill purchase logit prior, 0..4')
    parser.add_argument('--tower-focus-bias',type=float,default=None,help='v9 concentrate upgrades/skills on covered important routes, 0..4')
    parser.add_argument('--core-investment-bias',type=float,default=None,help='v9 persistent Tesla goal and saving logit prior, 0..4')
    parser.add_argument("--loadout", choices=("original", "full_elora", "full_ignus"), default=None)
    parser.add_argument("--modes", default="campaign")
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--focus-every", type=int, default=None, help="candidate: one hard training-level revisit every N resets; 0 disables")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--ticks-per-step", type=int, default=30)
    parser.add_argument("--max-ticks", type=int, default=200_000)
    parser.add_argument("--curriculum-every", type=int, default=0)
    parser.add_argument("--spend-before-wait", action="store_true")
    parser.add_argument("--reward-profile", choices=("balanced", "legacy"), default="balanced")
    parser.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--net-arch", nargs="+", type=int, help="hidden widths; candidate default: 128 128; macro default: 256 256; resume preserves architecture")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-final", type=float, default=5e-5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-final", type=float, default=0.003)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/campaign"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--fixed-game-seed',type=int,help='repeat this native combat seed in every training/evaluation episode')
    parser.add_argument('--demo-start-dataset',type=Path,help='verified v9 demonstrations for optional native prefix starts')
    parser.add_argument('--demo-start-probability',type=float,default=0.)
    parser.add_argument('--demo-start-min-wave',type=int,default=4)
    parser.add_argument('--demo-start-max-wave',type=int,default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=1, help="episodes PER level and evaluation policy mode")
    parser.add_argument("--eval-mode", choices=("stochastic", "deterministic", "both"), default="stochastic")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--baseline", action="store_true", help="evaluate the macro rule policy without training")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ablation-arm", choices=("direct", "assisted"),
                        help="single-level paired experiment with four equal training stages")
    parser.add_argument('--imitation-arm',choices=('pure','anchored','pretrained'))
    parser.add_argument('--bc-sampling',choices=('balanced','strategic'),default='balanced')
    parser.add_argument('--bc-verify-opening',action='store_true',help='stop before PPO if the deterministic BC policy does not build and upgrade')
    parser.add_argument('--bc-dataset',type=Path)
    parser.add_argument('--bc-split',choices=('native_seed','trajectory'),default='native_seed',help='trajectory split explicitly supports single fixed native seed datasets')
    parser.add_argument('--bc-init',type=Path,help='start fresh PPO from a saved step-0 BC checkpoint; skip BC pretraining')
    parser.add_argument('--bc-coef',type=float,default=.1)
    parser.add_argument('--bc-epochs',type=int,default=40)
    parser.add_argument('--bc-batch-size',type=int,default=32)
    parser.add_argument('--bc-lr',type=float,default=3e-4)
    parser.add_argument('--success-replay',action=argparse.BooleanOptionalAction,default=None,help='archive native victories; default on for anchored v10')
    parser.add_argument('--success-replay-dir',type=Path,default=Path('runs/success_memory'))
    parser.add_argument('--success-replay-ratio',type=float,default=.2,help='fraction of continuous BC batches from new victories, maximum .25')
    args = parser.parse_args(argv)
    if args.ports:
        try:
            args.ports = [int(p) for p in args.ports.split(",")]
        except ValueError:
            parser.error("ports must be comma-separated integers")
    ports = args.ports or [args.port]
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        parser.error("ports must be unique and in 1..65535")
    if args.eval_port is not None and (not 1 <= args.eval_port <= 65535 or args.eval_port in ports):
        parser.error("eval-port must be valid and separate from training ports")
    training = not (args.smoke or args.eval or args.baseline)
    if training and args.eval_every and args.eval_port is None:
        parser.error("periodic evaluation requires --eval-port; launch_training.py supplies it automatically")
    try:
        modes = [MODE_IDS[name] for name in args.modes.split(",")]
    except KeyError:
        parser.error("modes must be campaign,heroic,iron")
    if min(args.steps, args.ticks_per_step, args.max_ticks, args.eval_episodes, args.batch_size, args.n_epochs) < 1:
        parser.error("steps/ticks/episodes/batch size/epochs must be positive")
    if min(args.eval_every, args.curriculum_every, args.ent_coef, args.ent_final) < 0:
        parser.error("intervals and entropy coefficients must be nonnegative")
    if not (0 < args.gamma <= 1 and 0 < args.gae_lambda <= 1 and min(args.lr, args.lr_final, args.target_kl) > 0):
        parser.error("invalid discount, GAE, learning rate or target KL")
    if (args.rollout_steps is not None and args.rollout_steps < 2) or (args.net_arch is not None and any(n < 1 for n in args.net_arch)):
        parser.error("rollout-steps must be >=2; layer widths must be positive")
    if sum(bool(x) for x in (args.resume, args.eval, args.smoke, args.baseline)) > 1:
        parser.error("resume, eval, smoke and baseline are mutually exclusive")
    args.eval_levels = list(args.levels)
    checkpoint = args.resume or args.eval or args.bc_init
    previous = json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8')) if checkpoint else {}
    if args.fixed_game_seed is None:args.fixed_game_seed=previous.get('config',{}).get('fixed_game_seed')
    if args.fixed_game_seed is not None and not 1<=args.fixed_game_seed<2**31-1:
        parser.error('fixed-game-seed must be in 1..2147483646')
    if args.bc_init and args.net_arch is None:
        args.net_arch=previous.get('architecture',{}).get('pi')
    args.env_version = args.env_version or previous.get('schema', {}).get('schema', 'candidate-v6')
    saved_policy=previous.get('config',{}).get('policy_version','layout')
    args.policy_version=args.policy_version or saved_policy
    if checkpoint and args.policy_version!=saved_policy:parser.error('Checkpoint policy version differs; start fresh BC for the requested policy')
    if args.policy_version in ('route-v1','route-v2-invest') and args.env_version!='candidate-v10':parser.error('Route policies require candidate-v10 observations')
    if args.success_replay is None:args.success_replay=args.env_version=='candidate-v10' and args.imitation_arm=='anchored'
    if args.success_replay and args.env_version not in ('candidate-v9','candidate-v10'):parser.error('Success replay requires candidate-v9/v10')
    if not 0<=args.success_replay_ratio<=.25:parser.error('success-replay-ratio must be in 0..0.25')
    if not 0<=args.demo_start_probability<=1 or not 1<=args.demo_start_min_wave<=args.demo_start_max_wave<=16:
        parser.error('Invalid demonstration start probability/wave range')
    if args.demo_start_probability and (args.env_version not in ('candidate-v9','candidate-v10') or not args.demo_start_dataset or not args.demo_start_dataset.is_file()):
        parser.error('Demonstration starts require candidate-v9/v10 and a valid dataset')
    if args.fixed_game_seed is not None and args.env_version not in ('candidate-v6','candidate-v7','candidate-v8','candidate-v9','candidate-v10'):
        parser.error('fixed-game-seed requires candidate-v6 or newer')
    if args.baseline and args.env_version in ('candidate-v8','candidate-v9','candidate-v10'):parser.error('fact schemas do not support old rule baselines')
    candidate=args.env_version.startswith('candidate-')
    if args.type_balance_alpha is None:
        args.type_balance_alpha=previous.get('config',{}).get('type_balance_alpha',0.0) if checkpoint else (1.0 if candidate else 0.0)
    if not 0<=args.type_balance_alpha<=1 or (not candidate and args.type_balance_alpha):
        parser.error('type-balance-alpha must be in 0..1 and requires a candidate environment')
    for name in ('advanced_tower_bias','tesla_route_bias','tower_skill_bias','tower_focus_bias','core_investment_bias'):
        value=getattr(args,name)
        if value is None:value=previous.get('config',{}).get(name,0.0)
        if not 0<=value<=4 or (value and args.env_version not in ('candidate-v9','candidate-v10')):
            parser.error(name.replace('_','-')+' must be in 0..4 and requires candidate-v9/v10')
        setattr(args,name,value)
    args.rollout_steps = args.rollout_steps or previous.get('config',{}).get('rollout_steps',64 if args.env_version=='candidate-v7' else (256 if candidate else 1024))
    if args.focus_every is None:
        args.focus_every=previous.get('config',{}).get('focus_every',3 if candidate else 0)
    if args.focus_every<0: parser.error('focus-every must be nonnegative')
    args.loadout = args.loadout or (previous.get('config', {}).get('loadout', 'original') if checkpoint else 'full_elora')
    if args.holdout_levels is None:
        args.holdout_levels = previous.get('config', {}).get('holdout_levels', [3, 5, 8, 11, 18, 23])
    args.holdout_levels = sorted(set(args.holdout_levels) & set(args.eval_levels))
    args.levels = [n for n in args.eval_levels if n not in args.holdout_levels]
    if training and not args.levels:
        parser.error("No training levels remain after excluding holdouts")
    if args.ablation_arm:
        if not training or args.env_version != 'candidate-v6' or len(args.levels) != 1 or len(modes) != 1 or args.eval_port is None:
            parser.error("ablation requires fresh candidate-v6 training, one level/mode and an eval port")
        if args.steps % (4 * args.rollout_steps * len(ports)):
            parser.error("ablation steps must be divisible by four complete vector rollouts")
    if args.imitation_arm:
        if (not training or args.resume or args.ablation_arm or args.env_version not in ('candidate-v8','candidate-v9','candidate-v10')
                or (args.env_version!='candidate-v10' and args.levels!=[21]) or modes!=[1] or args.difficulty!=2 or args.loadout!='full_elora'
                or args.eval_port is None or not args.bc_dataset or not args.bc_dataset.is_file()):
            parser.error('imitation requires fresh candidate-v8/v9/v10 campaign training (v8/v9: level 21 only), normal difficulty, full_elora, BC data and evaluation port')
        if args.bc_coef<=0 or min(args.bc_epochs,args.bc_batch_size,args.bc_lr)<=0:
            parser.error('BC coefficient, epochs, batch size and learning rate must be positive')
        rollout=args.rollout_steps*len(ports)
        if args.steps%rollout or not args.eval_every or args.eval_every%rollout:
            parser.error('imitation steps/evaluation interval must be divisible by a complete vector rollout')
    if args.bc_init and (args.resume or args.eval or args.smoke or args.baseline
            or args.imitation_arm not in ('anchored','pretrained') or not args.bc_init.is_file()):
        parser.error('--bc-init requires an existing BC checkpoint and a fresh anchored/pretrained run')
    if args.bc_dataset and not args.imitation_arm:
        parser.error('--bc-dataset requires --imitation-arm')
    return args, modes


def evaluate_suite(model, env, output, episodes=1, seed=42, deterministic=True, holdout_levels=()):
    from ppo_evaluation import evaluate
    from copy import deepcopy
    base = env.unwrapped
    rng_state = deepcopy(base.np_random.bit_generator.state)
    summaries = {}
    try:
        for level, mode in base.schedule.pairs:
            base.evaluation_pair = (level, mode)
            split = "holdout" if level in holdout_levels else "train"
            summaries[f"{level}:{mode}"] = evaluate(
                model, env, episodes, deterministic=deterministic, seed=seed, output=str(output),
                label=f"split={split},level={level},mode={mode}", record_context={"split": split})
    finally:
        base.evaluation_pair = None
        base.np_random.bit_generator.state = rng_state

    def aggregate(values):
        if not values:
            return None
        return {key: sum(s.get(key, 0) for s in values) / len(values)
                for key in ("win_rate", "mean_wave_progress", "mean_life_ratio", "mean_return")}

    result = aggregate(list(summaries.values()))
    result.update(by_level_mode=summaries, splits={
        split: aggregate([s for pair, s in summaries.items() if
                          (int(pair.split(":")[0]) in holdout_levels) == (split == "holdout")])
        for split in ("train", "holdout")})
    result['by_stage'] = {stage: {
        split: aggregate([s for pair, s in summaries.items()
                          if low <= int(pair.split(':')[0]) <= high
                          and (int(pair.split(':')[0]) in holdout_levels) == (split == 'holdout')])
        for split in ('train', 'holdout')}
        for stage, low, high in (('early',1,6),('middle',7,12),('late',13,26))}
    return result


def environment_schema(args):
    if args.env_version=='candidate-v10':
        from kr_support import SCHEMA
        from kr_candidates import MAX_CANDIDATES
        from support_features import FEATURE_NAMES,SCENE_NAMES,ROUTES,HORIZONS
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),
                    scene=list(SCENE_NAMES),routes=list(ROUTES),horizons=list(HORIZONS),actor_context='scene64_mean_max16',
                    support_mechanisms_version=1)
    if args.env_version=='candidate-v9':
        from kr_layout import SCHEMA
        from kr_candidates import MAX_CANDIDATES
        from layout_features import FEATURE_NAMES,SCENE_NAMES,ROUTES,HORIZONS
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),
                    scene=list(SCENE_NAMES),routes=list(ROUTES),horizons=list(HORIZONS),actor_context='scene64_mean_max16')
    if args.env_version=='candidate-v8':
        from kr_light import SCHEMA
        from kr_candidates import MAX_CANDIDATES
        from light_features import FEATURE_NAMES,SCENE_NAMES
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),scene=list(SCENE_NAMES))
    if args.env_version == 'candidate-v7':
        from kr_route_planning import SCHEMA
        from kr_candidates import MAX_CANDIDATES
        from route_features import FEATURE_NAMES,SCENE_NAMES,ROUTES,HORIZONS
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),
                    scene=list(SCENE_NAMES),routes=list(ROUTES),horizons=list(HORIZONS))
    if args.env_version == 'candidate-v6':
        from kr_planning import SCHEMA, MAX_CANDIDATES
        from planning_features import FEATURE_NAMES, SCENE_NAMES
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),scene=list(SCENE_NAMES))
    if args.env_version == 'candidate-v5':
        from kr_candidates import SCHEMA, MAX_CANDIDATES
        from candidate_features import FEATURE_NAMES, SCENE_NAMES
        return dict(schema=SCHEMA,max_candidates=MAX_CANDIDATES,features=list(FEATURE_NAMES),scene=list(SCENE_NAMES))
    if args.env_version == 'macro-v4-strategy':
        from kr_strategy import KRStrategyEnv, STRATEGY_MACROS, SKILLS, ROUTES, ROUTE_FEATURES
        return dict(schema=KRStrategyEnv.schema, macros=list(STRATEGY_MACROS), enemy_types=ENEMY_TYPES,
                    tower_skills=list(SKILLS), routes=list(ROUTES), route_features=ROUTE_FEATURES)
    if args.env_version == 'macro-v3-replacement':
        from kr_replacement import KRReplacementEnv, REPLACEMENT_MACROS
        return dict(schema=KRReplacementEnv.schema, macros=list(REPLACEMENT_MACROS), enemy_types=ENEMY_TYPES)
    if args.env_version == 'macro-v2-tactical':
        from kr_tactical import KRTacticalEnv, TACTICAL_MACROS
        return dict(schema=KRTacticalEnv.schema, macros=list(TACTICAL_MACROS), enemy_types=ENEMY_TYPES)
    return dict(schema=KRMacroEnv.schema, macros=list(MACROS), enemy_types=ENEMY_TYPES)


def main():
    args, modes = parse_args()
    from training_support import validate_checkpoint
    schema = environment_schema(args)
    validate_checkpoint(args, schema)
    args.out.mkdir(parents=True, exist_ok=True)
    run = args.out / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.env_version in ('candidate-v6','candidate-v7','candidate-v8','candidate-v9','candidate-v10'):
        from kr_planning import PLANNING_REVISION
        config.update(planning_revision=PLANNING_REVISION,late_replacements_per_wave=1,
                      late_rich_replacements_per_wave=2,late_rich_gold_threshold=1500)
        if args.env_version=='candidate-v7':
            from kr_route_planning import ROUTE_PLANNING_REVISION
            config.update(route_planning_revision=ROUTE_PLANNING_REVISION,
                          route_estimator='continuous_leak_proxy',blocking_exposure_modeled=False)
        if args.env_version in ('candidate-v8','candidate-v9','candidate-v10'):config.update(observation_mode='local_facts',combat_prediction=False,upgrade_search=False)
    (run / "config.json").write_text(json.dumps(dict(config=config, schema=schema), indent=2), encoding="utf-8")
    if args.ablation_arm:
        from difficulty_ablation import run_ablation
        run_ablation(args, modes, run, schema, config)
    else:
        from parallel_train import run_parallel
        run_parallel(args, modes, run, schema, config)


if __name__ == "__main__":
    main()
