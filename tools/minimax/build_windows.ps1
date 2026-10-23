# Build the MiniMax engine on Windows (VS 2026 + CUDA 13.3, sm_120) from the worktree, through a subst drive.
#
# nvcc drops include paths that contain the apostrophe in "Jon's Projects" (C1083: ggml.h / cuda_runtime.h not
# found).  This script maps the worktree to M: and configures/builds from there; a CMake cache written against the
# apostrophe path is deleted first (it would keep the old path in every rule).
#
#   pwsh -File tools/minimax/build_windows.ps1                # configure + build strata and strata-minimax
#   pwsh -File tools/minimax/build_windows.ps1 -Targets strata-minimax
#   pwsh -File tools/minimax/build_windows.ps1 -BuildDir build-minimax -Clean
#
# The environment's CUDA_PATH points at 13.1; this pins 13.3.  Tokens/keys are never read or printed.
[CmdletBinding()]
param(
    [string[]] $Targets = @("strata", "strata-minimax"),
    [string] $BuildDir = "build",
    [string] $SubstDrive = "M:",
    [string] $GgmlSubstDrive = "L:",
    [switch] $Clean,
    [switch] $ConfigureOnly,
    [int] $Jobs = 0
)
$ErrorActionPreference = "Stop"

$RootReal = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$Cuda   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$Python = "C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe"
$Cmake  = "C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts\cmake.exe"
$Ninja  = "C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts\ninja.exe"

# llama.cpp checkout whose ggml `src/prefill` and the CPU expert pool are built from.  It must match the pinned
# commit in third_party/ggml/VERSION.txt: the MiniMax-patched _baseline/llama.cpp is a different ggml (its MMQ API
# differs) and is not a build dependency of Strata.  Set STRATA_GGML_DIR, or leave it empty and CMake fetches the
# pinned commit itself.
$Ggml = $env:STRATA_GGML_DIR
if (-not $Ggml) {
    $local = Join-Path $RootReal "third_party\llama.cpp"
    if (Test-Path (Join-Path $local "ggml")) { $Ggml = $local }
}
if ($Ggml -and -not (Test-Path (Join-Path $Ggml "ggml"))) {
    throw "STRATA_GGML_DIR does not look like a llama.cpp checkout: $Ggml"
}
if (-not $Ggml) {
    Write-Host "no STRATA_GGML_DIR: CMake will fetch the pinned ggml commit (network needed)"
}

foreach ($p in @($Vcvars, (Join-Path $Cuda "bin\nvcc.exe"), $Cmake, $Ninja)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}

# --- subst: nvcc and the apostrophe ---------------------------------------------------------------
# Both the worktree and the ggml checkout can sit under "Jon's Projects"; nvcc drops an include directory whose path
# contains that apostrophe (C1083), so both are reached through a drive letter while CMake runs.
$Src = $RootReal
$usedSubst = $false
if ($RootReal -match "'") {
    $existing = (& subst) | Out-String
    if ($existing -match ("(?m)^" + [regex]::Escape($SubstDrive) + "\s")) {
        & subst $SubstDrive /d | Out-Null
    }
    & subst $SubstDrive $RootReal
    if ($LASTEXITCODE -ne 0) { throw "subst $SubstDrive $RootReal failed" }
    $Src = "$SubstDrive\"
    $usedSubst = $true
}
$GgmlReal = $null
$usedGgmlSubst = $false
if ($Ggml) {
    $GgmlReal = (Resolve-Path $Ggml).Path
    if ($GgmlReal -match "'") {
        $existing = (& subst) | Out-String
        if ($existing -match ("(?m)^" + [regex]::Escape($GgmlSubstDrive) + "\s")) {
            & subst $GgmlSubstDrive /d | Out-Null
        }
        # Map the checkout's parent, not the checkout itself: CMakeLists appends "/ggml", and a drive root
        # ("L:/") would become "L://ggml" - two spellings of one header, which ninja refuses.
        $GgmlParent = Split-Path $GgmlReal -Parent
        $GgmlLeaf = Split-Path $GgmlReal -Leaf
        & subst $GgmlSubstDrive $GgmlParent
        if ($LASTEXITCODE -ne 0) { throw "subst $GgmlSubstDrive $GgmlParent failed" }
        $Ggml = "$GgmlSubstDrive/$GgmlLeaf"
        $usedGgmlSubst = $true
    }
}
$Build = Join-Path $Src $BuildDir
$BuildReal = Join-Path $RootReal $BuildDir
$Cache = Join-Path $BuildReal "CMakeCache.txt"

try {
    if ($Clean -and (Test-Path $BuildReal)) { Remove-Item -Recurse -Force $BuildReal }
    elseif (Test-Path $Cache) {
        $cacheText = Get-Content -Raw -ErrorAction SilentlyContinue $Cache
        # Only the project's own source path matters: reconfiguring updates STRATA_GGML_DIR in place and keeps the
        # objects already compiled.  A source path with the apostrophe must be rebuilt from scratch.
        if ($cacheText -match "(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=.*Jon'") {
            Write-Host "build cache uses the apostrophe path; recreating $BuildReal"
            Remove-Item -Recurse -Force $BuildReal
        }
    }

    $env:CUDA_PATH = $Cuda
    $env:CUDAToolkit_ROOT = $Cuda
    $env:CUDACXX = "$Cuda\bin\nvcc.exe"
    $vcenv = & cmd.exe /c "call `"$Vcvars`" >nul && set"
    foreach ($line in $vcenv) {
        if ($line -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
    }
    $env:CUDA_PATH = $Cuda
    $env:CUDAToolkit_ROOT = $Cuda
    $env:CUDACXX = "$Cuda\bin\nvcc.exe"
    $env:PATH = "$Cuda\bin;" + (Split-Path $Cmake) + ";" + $env:PATH

    $cfgArgs = @(
        "-S", $Src,
        "-B", $Build,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=cl",
        "-DCMAKE_CXX_COMPILER=cl",
        "-DCMAKE_CUDA_COMPILER=$Cuda\bin\nvcc.exe",
        "-DCUDAToolkit_ROOT=$Cuda",
        "-DCMAKE_MAKE_PROGRAM=$Ninja",
        "-DSTRATA_ENABLE_CUDA=ON",
        "-DSTRATA_BUILD_TESTS=OFF",
        "-DSTRATA_Q6K_EXPERTS=ON",
        "-DSTRATA_MMQ_KQUANTS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=120"
    )
    if ($Ggml) { $cfgArgs += "-DSTRATA_GGML_DIR=$Ggml" }
    & $Cmake @cfgArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }

    if (-not $ConfigureOnly) {
        if ($Jobs -le 0) { $Jobs = [Math]::Max(2, [Environment]::ProcessorCount - 4) }
        & $Cmake --build $Build --target @Targets -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw "cmake build failed: $LASTEXITCODE" }
        Write-Host "built: $(($Targets | ForEach-Object { Join-Path $BuildReal ($_ + '.exe') }) -join ', ')"
    }
} finally {
    if ($usedSubst) { & subst $SubstDrive /d | Out-Null }
    if ($usedGgmlSubst) { & subst $GgmlSubstDrive /d | Out-Null }
}
