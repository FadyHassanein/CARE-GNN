# Sweep lambda_policy for the per-element advantage actor-critic.
#
# Skips 0.3 since we already have that result (test AUC 0.7838).
# Total runtime: ~75 min on this hardware (3 runs x 25 min).
#
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\sweep_lambda_policy.ps1

$ErrorActionPreference = 'Stop'

$python = 'C:\Users\fadyh\miniconda3\envs\torch_cuda\python.exe'
$logdir = 'logs\sweep_lambda_policy'
New-Item -ItemType Directory -Path $logdir -Force | Out-Null

$lambdas = @('0.1', '0.15', '0.2')

foreach ($lp in $lambdas) {
    Write-Host "=== $(Get-Date -Format 'HH:mm:ss') :: lambda_policy=$lp ==="
    $logout = Join-Path $logdir "lp_${lp}.out"
    $logerr = Join-Path $logdir "lp_${lp}.err"
    $logfin = Join-Path $logdir "lp_${lp}.log"

    $argList = @(
        'train.py',
        '--data', 'yelp', '--model', 'CARE',
        '--use-capn', '--use-actor-critic',
        '--gnn-warmup-epochs', '5', '--lambda-policy-ramp-epochs', '5',
        '--llm-priors-file', 'data/llm_priors/yelp_priors.json',
        '--text-enrichment', 'gate', '--text-state-enrichment',
        '--seed', '72',
        '--lambda-policy', $lp
    )

    Start-Process -FilePath $python -ArgumentList $argList `
        -RedirectStandardOutput $logout -RedirectStandardError $logerr `
        -NoNewWindow -Wait

    Get-Content $logout, $logerr | Set-Content $logfin -Encoding utf8
    Remove-Item $logout, $logerr -ErrorAction SilentlyContinue

    Write-Host "    done -> $logfin"
}

Write-Host ""
Write-Host "=== Sweep complete. Best-checkpoint AUCs: ==="
foreach ($lp in $lambdas) {
    $log = Join-Path $logdir "lp_${lp}.log"
    $line = Select-String -Path $log -Pattern 'New best model saved' | Select-Object -Last 1
    if ($line -and $line.Line -match 'AUC: (0\.\d+)') {
        Write-Host "  lambda_policy=$lp  best val AUC=$($matches[1])  ($log)"
    } else {
        Write-Host "  lambda_policy=$lp  (no best-model line found in $log)"
    }
}
