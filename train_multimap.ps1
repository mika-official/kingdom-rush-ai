param([string]$BcInit = '', [switch]$DryRun)
$launchOptions = @()
if ($BcInit) { $launchOptions += @('--bc-init', $BcInit) }
if ($DryRun) { $launchOptions += '--dry-run' }
python -u launch_training.py @launchOptions `
  --workers 2 --base-port 9861 --device cpu `
  --env-version candidate-v10 --policy-version route-v2-invest --modes campaign --difficulty 2 --loadout full_elora `
  --levels 13-20,23-26 --holdout-levels 19,26 `
  --seed 42 --fixed-game-seed 191664964 `
  --steps 100000 --rollout-steps 100 --batch-size 200 --n-epochs 2 `
  --lr 0.00003 --lr-final 0.00001 --target-kl 0.005 `
  --eval-every 25000 --eval-episodes 1 --eval-mode both `
  --focus-every 0 --curriculum-every 0 `
  --imitation-arm anchored `
  --bc-dataset runs/walkthrough_v10_expanded/corrected_data.json `
  --bc-split trajectory --bc-sampling strategic `
  --bc-epochs 60 --bc-batch-size 32 --bc-lr 0.0003 --bc-coef 0.1 `
  --bc-verify-opening `
  --demo-start-dataset runs/walkthrough_v10_expanded/prefix_data.json `
  --demo-start-probability 0.5 --demo-start-min-wave 4 --demo-start-max-wave 10 `
  --advanced-tower-bias 1 --tesla-route-bias 1 `
  --tower-skill-bias 1.2 --tower-focus-bias 1 --core-investment-bias 3 `
  --success-replay --success-replay-dir runs/success_memory_multimap `
  --success-replay-ratio 0.2 `
  --tensorboard --out runs/v10_multimap_expanded_no_pandaemonium
