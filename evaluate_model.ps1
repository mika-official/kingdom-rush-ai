param(
    [string]$Game = 'E:/SteamLibrary/steamapps/common/Kingdom Rush/Kingdom Rush.exe',
    [int]$Episodes = 1,
    [int]$Port = 9961,
    [switch]$DryRun
)
$launchOptions = @()
if ($DryRun) { $launchOptions += '--dry-run' }
Push-Location -LiteralPath $PSScriptRoot
try {
    python launch_training.py @launchOptions --game $Game --workers 1 --base-port $Port `
      --eval runs/v10_multimap_expanded_no_pandaemonium/20261006_201129_090211/best_deterministic.zip `
      --levels 1-26 --holdout-levels 1-12,19,21,22,26 `
      --env-version candidate-v10 --policy-version route-v2-invest `
      --difficulty 2 --modes campaign --loadout full_elora `
      --fixed-game-seed 191664964 --seed 100042 `
      --eval-mode deterministic --eval-episodes $Episodes --out runs/github_evaluation
    if ($LASTEXITCODE -ne 0) { throw ('Evaluation launcher failed: ' + $LASTEXITCODE) }
}
finally { Pop-Location }
