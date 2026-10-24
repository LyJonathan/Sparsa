# Download one HF tier folder, resumable and size-aware.
#
#   powershell -File tools/minimax/dl_tier2.ps1 -Tier MiniMax-M3.1-Flash-IQ4_KS -Dest D:\hf-iq4ks
#
# Every shard is compared against the Hub's blob size before any byte is fetched, so a shard that is
# already complete is never opened again.  That matters: an unconditional `curl -C -` against a
# complete file can get a 200 instead of a 416 and truncate it, which silently re-downloads 15 GB.
# A partial shard resumes from its current length; one longer than the Hub blob is truncated first.
param(
    [Parameter(Mandatory = $true)][string] $Tier,     # e.g. MiniMax-M3.1-Flash-IQ4_KS
    [Parameter(Mandatory = $true)][string] $Dest,     # e.g. D:\hf-iq4ks
    [string] $Repo = "LyJonathan/MiniMax-M3.1-Flash-GGUF",
    [int] $MaxPasses = 4
)
$ErrorActionPreference = "Continue"
$dir = Join-Path $Dest $Tier
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$logPath = Join-Path $Dest "dl_$Tier.log"

function Say($m) { "$(Get-Date -Format HH:mm:ss) $m" | Tee-Object -FilePath $logPath -Append }

for ($pass = 1; $pass -le $MaxPasses; $pass++) {
    $j = curl.exe -s "https://huggingface.co/api/models/$Repo`?blobs=true" | ConvertFrom-Json
    $parts = @($j.siblings | Where-Object { $_.rfilename -like "$Tier/*" })
    if ($parts.Count -eq 0) { Say "no shards for $Tier yet"; Start-Sleep 120; continue }
    $missing = 0
    foreach ($p in $parts) {
        $name = Split-Path $p.rfilename -Leaf
        $out = Join-Path $dir $name
        $want = [int64]$p.size
        $have = if (Test-Path $out) { (Get-Item $out).Length } else { [int64]0 }
        if ($have -eq $want) { continue }
        if ($have -gt $want) { Say "truncating oversize $name ($have > $want)"; Remove-Item $out -Force; $have = 0 }
        $missing++
        Say "GET $name have=$([math]::Round($have/1MB))/$([math]::Round($want/1MB)) MB"
        $url = "https://huggingface.co/$Repo/resolve/main/$($p.rfilename)"
        curl.exe -L -C - --retry 8 --retry-delay 5 -o $out $url 2>&1 | Select-Object -Last 1 | ForEach-Object { Say $_ }
        if ($LASTEXITCODE -ne 0) { Say "curl rc=$LASTEXITCODE on $name" }
    }
    if ($missing -eq 0) { Say "complete: $($parts.Count) parts of $Tier"; exit 0 }
    Start-Sleep 20
}
Say "GAVE UP after $MaxPasses passes"
exit 1
