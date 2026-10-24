# Run the G3 gate until it has every prompt and the ppl pass, resuming from completed dumps.
# The shared rig kills a model process now and then (another task benchmarks the same box), so a single
# g3_sparsa.py pass is not enough: this loop restarts it and the runner reuses any complete dump.
#   powershell -File tools/minimax/g3_until_done.ps1 -Model <part1.gguf> -Out <dir> -Tier <label> -TierArgs "--kv host ..."
param(
    [Parameter(Mandatory = $true)][string] $Model,
    [Parameter(Mandatory = $true)][string] $Out,
    [Parameter(Mandatory = $true)][string] $Tier,
    [string] $TierArgs = "--memory-limit-mib 40000 --prefill-batch 0 --cpu-threads 16 --max-context 8192 --vram-slots 0 --dense-layers-gpu 0 --kv host --gpus 0,1 --layer-split auto",
    [string] $Ref = "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out\logits",
    [string] $Ids = "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\g3_iq2xs\ids",
    [string] $ValidateWith = "",
    [string] $ValidatePpl = "",
    [int] $Attempts = 40,
    [int] $WaitSeconds = 120
)
$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\venv-ref\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Have-All($outDir) {
    $j = Join-Path $outDir "results.json"
    if (-not (Test-Path $j)) { return $false }
    try { $r = Get-Content $j -Raw | ConvertFrom-Json } catch { return $false }
    if (-not $r.ppl) { return $false }
    return (@($r.per_prompt).Count -eq 8)
}

for ($i = 1; $i -le $Attempts; $i++) {
    if (Have-All $Out) { break }
    Write-Output "=== attempt $i $(Get-Date -Format HH:mm:ss)"
    $argsList = @(
        (Join-Path $repo "tools\minimax\g3_sparsa.py"),
        "--model", $Model, "--out", $Out, "--tier", $Tier,
        "--ref", $Ref, "--ids", $Ids, "--retries", "1"
    )
    if ($ValidateWith) { $argsList += @("--validate-with", $ValidateWith) }
    if ($ValidatePpl) { $argsList += @("--validate-ppl", $ValidatePpl) }
    $argsList += ($TierArgs -split "\s+")
    & $py @argsList
    if (-not (Have-All $Out)) { Start-Sleep -Seconds $WaitSeconds }
}
if (Have-All $Out) { Write-Output "DONE $(Get-Date -Format HH:mm:ss)" } else { Write-Output "GAVE UP $(Get-Date -Format HH:mm:ss)" }
