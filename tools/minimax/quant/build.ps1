# Build strata-mm-quant (Release, MSVC, Ninja) against the patched llama.cpp ggml.
# nvcc breaks on the apostrophe in "Jon's Projects". This script does not call nvcc,
# but maps both trees to drive letters so the same layout works if a later step does.
# Usage: powershell -File tools\minimax\quant\build.ps1
$ErrorActionPreference = "Stop"
$wt = "E:\Jon's Projects\AI\Strata Fork Minimax\wt-quant"
$llama = "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_baseline\llama.cpp"
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars)) { throw "vcvars64.bat not found: $vcvars" }
if (-not (Test-Path -LiteralPath "$llama\ggml\include\ggml.h")) { throw "llama.cpp ggml.h missing: $llama" }

# subst fails when the letter is already mapped. Reuse an existing mapping.
function Ensure-Subst([string]$Letter, [string]$Target) {
    $cur = (subst) | Where-Object { $_ -like "${Letter}:\*" }
    if ($cur) {
        Write-Host $cur
        return
    }
    & subst "${Letter}:" $Target
    if ($LASTEXITCODE -ne 0) { throw "subst ${Letter}: failed" }
}
Ensure-Subst "Q" $wt
Ensure-Subst "R" $llama

# A one-line `cmd /c "call vcvars && set PATH=...%PATH%"` expands %PATH% before
# vcvars runs, which drops cl and lets CMake pick MSYS gcc. The .cmd file
# expands PATH after vcvars.
& cmd /c "Q:\tools\minimax\quant\build-msvc.cmd"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
