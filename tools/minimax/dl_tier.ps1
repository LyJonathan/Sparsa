# Download one HF tier folder with curl (resumable), polling for new shards until all parts exist.
# Usage: pwsh -File tools/minimax/dl_tier.ps1 -Tier MiniMax-M3.1-Flash-IQ3_K -Dest D:\hf-iq3k
param(
    [Parameter(Mandatory = $true)][string] $Tier,
    [Parameter(Mandatory = $true)][string] $Dest,
    [string] $Repo = "LyJonathan/MiniMax-M3.1-Flash-GGUF",
    [int] $PollSeconds = 180
)
$ErrorActionPreference = "Stop"
$dir = Join-Path $Dest $Tier
New-Item -ItemType Directory -Force -Path $dir | Out-Null

function Get-Parts {
    $api = "https://huggingface.co/api/models/$Repo"
    $j = curl.exe -s $api | ConvertFrom-Json
    $j.siblings | Where-Object { $_.rfilename -like "$Tier/*" } |
        ForEach-Object {
            $p = [pscustomobject]@{
                name = [System.IO.Path]::GetFileName($_.rfilename)
                path = $_.rfilename
                size = if ($_.lfs) { [int64]$_.lfs.size } else { [int64]0 }
            }
            $p
        }
}

$deadline = (Get-Date).AddHours(12)
while ((Get-Date) -lt $deadline) {
    $parts = @(Get-Parts)
    if ($parts.Count -eq 0) {
        Write-Output "$(Get-Date -Format HH:mm:ss) no shards for $Tier yet"
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    $missing = 0
    foreach ($p in $parts) {
        $out = Join-Path $dir $p.name
        $have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
        if ($p.size -gt 0 -and $have -ge $p.size) { continue }
        $missing++
        $url = "https://huggingface.co/$Repo/resolve/main/$($p.path)"
        Write-Output "$(Get-Date -Format HH:mm:ss) GET $($p.name) have=$([math]::Round($have/1MB))/$([math]::Round($p.size/1MB)) MB"
        # -C - resumes a partial file; --retry keeps a flaky link from ending the shard early.
        curl.exe -L -C - --retry 5 --retry-delay 3 -o $out $url
        if ($LASTEXITCODE -ne 0) { Write-Output "$(Get-Date -Format HH:mm:ss) curl rc=$LASTEXITCODE on $($p.name)" }
    }
    if ($missing -eq 0) {
        Write-Output "$(Get-Date -Format HH:mm:ss) complete: $($parts.Count) parts of $Tier"
        break
    }
    Start-Sleep -Seconds 15
}
