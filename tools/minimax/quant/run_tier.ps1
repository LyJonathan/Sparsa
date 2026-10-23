# Run or prefetch the MM-IQ2_XS split job.
#   powershell -File tools\minimax\quant\run_tier.ps1 -Plan
#   powershell -File tools\minimax\quant\run_tier.ps1 -Tier MM-IQ2_XS -Machine pc -Imatrix E:\mm-quant-work\imatrix.gguf -Model D:\model -Exe .\strata-mm-quant.exe -OutDir E:\mm-quant-work\iq2xs
param(
    [string]$Tier = "MM-IQ2_XS",
    [string]$Machine = "",
    [string]$Model = "",
    [string]$Exe = "",
    [string]$Imatrix = "",
    [string]$OutDir = "",
    [switch]$Plan,
    [switch]$Prefetch,
    [switch]$DryRun,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }
$script = Join-Path $PSScriptRoot "run_tier.py"
$arg = @($script, "--tier", $Tier)
if ($Plan) { $arg += "--plan" }
if ($Machine) { $arg += @("--machine", $Machine) }
if ($Model) { $arg += @("--model", $Model) }
if ($Exe) { $arg += @("--exe", $Exe) }
if ($Imatrix) { $arg += @("--imatrix", $Imatrix) }
if ($OutDir) { $arg += @("--out-dir", $OutDir) }
if ($Prefetch) { $arg += "--prefetch" }
if ($DryRun) { $arg += "--dry-run" }
if ($Force) { $arg += "--force" }
& $py @arg
exit $LASTEXITCODE
