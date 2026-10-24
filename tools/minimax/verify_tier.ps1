# Verify a downloaded GGUF tier byte-for-byte against the Hub's own content hashes.
#
#   powershell -File tools/minimax/verify_tier.ps1 -Dir D:\hf-iq3k -Tier MiniMax-M3.1-Flash-IQ3_K
#
# Two sources are tried, in order: the Hub's blob sha256 for each file (always present), then the
# repo's SHA256SUMS file (published only once a tier is finished uploading).  A tier that matches
# every blob hash is the bytes the uploader published, so a G3 number measured on it cannot be a
# corrupt-file artefact.
param(
    [Parameter(Mandatory = $true)][string] $Dir,      # parent folder that holds the tier folder
    [Parameter(Mandatory = $true)][string] $Tier,     # e.g. MiniMax-M3.1-Flash-IQ3_K
    [string] $Repo = "LyJonathan/MiniMax-M3.1-Flash-GGUF"
)
$ErrorActionPreference = "Continue"
$j = curl.exe -s "https://huggingface.co/api/models/$Repo`?blobs=true" | ConvertFrom-Json
$bad = 0; $n = 0
foreach ($s in ($j.siblings | Where-Object { $_.rfilename -like "$Tier/*" })) {
    $f = Join-Path (Join-Path $Dir $Tier) (Split-Path $s.rfilename -Leaf)
    $want = if ($s.lfs.sha256) { $s.lfs.sha256 } else { $s.lfs.oid }
    if (-not (Test-Path $f)) { "MISSING $($s.rfilename)"; $bad++; continue }
    $got = (Get-FileHash -Algorithm SHA256 -Path $f).Hash.ToLower()
    $n++
    if ($want -and $got -eq $want) { "OK   $($s.rfilename)" }
    else { "FAIL $($s.rfilename) want=$want got=$got"; $bad++ }
}
"checked=$n bad=$bad"
if ($bad -gt 0) { exit 1 }
