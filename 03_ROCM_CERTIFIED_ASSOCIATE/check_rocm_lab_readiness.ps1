<#
.SYNOPSIS
    Checks Docker Desktop, WSL2 backend, and NVIDIA RTX 3050 isolation readiness
    for AMD ROCm Certified Associate Labs (Modules 1-5) and saves an audit log
    to docker_lab_logs\readiness_audit_latest.log.
#>

$ErrorActionPreference = 'Continue'

$logDir = Join-Path $PSScriptRoot 'docker_lab_logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$latestLog = Join-Path $logDir 'readiness_audit_latest.log'

$inv = [System.Globalization.CultureInfo]::InvariantCulture
$utcStamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ', $inv)
$buffer = New-Object System.Collections.Generic.List[string]
$buffer.Add("ROCm Certified Associate Readiness Audit (UTC: $utcStamp)")
$buffer.Add("Host: $env:COMPUTERNAME")
$buffer.Add("=" * 68)

function Emit-Line([string]$Line) {
    Write-Host $Line
    $buffer.Add($Line)
}

Emit-Line "`n=== 1. WSL2 Status ==="
$wslList = (& wsl -l -v 2>&1) | ForEach-Object { ($_ -replace "`0", "").TrimEnd() } | Where-Object { $_ -ne "" }
foreach ($l in $wslList) { Emit-Line $l }

Emit-Line "`n=== 2. WSL2 /dev/accel Check ==="
$wslAccel = (& wsl -d Ubuntu -- bash -c "ls -la /dev/accel 2>&1 || echo 'NO_DEV_ACCEL (expected on WSL2)'; uname -r" 2>&1)
foreach ($l in $wslAccel) { Emit-Line "$l" }

Emit-Line "`n=== 3. Docker Desktop / CLI Readiness ==="
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
    $fallbackDocker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
    if (Test-Path $fallbackDocker) {
        $dockerCmd = $fallbackDocker
    }
}

if ($dockerCmd) {
    $dVer = (& $dockerCmd --version 2>&1) -join "`n"
    $dInfo = (& $dockerCmd info --format "ServerVersion={{.ServerVersion}} OSType={{.OSType}} Driver={{.Driver}}" 2>&1) -join "`n"
    Emit-Line $dVer
    Emit-Line $dInfo
} else {
    Emit-Line "WARNING: Docker CLI is NOT installed on Windows PATH or WSL2."
    Emit-Line "Action needed for Docker-based ROCm labs:"
    Emit-Line "  winget install -e --id Docker.DockerDesktop"
    Emit-Line "  Enable 'Use the WSL 2 based engine' in Docker Desktop Settings -> Resources -> WSL Integration."
}

Emit-Line "`n=== 4. NVIDIA RTX 3050 Isolation Check ==="
$nvOut = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1)
foreach ($l in $nvOut) { Emit-Line "$l" }

[System.IO.File]::WriteAllText($latestLog, ($buffer -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Host "`nSaved readiness audit log to: $latestLog" -ForegroundColor Green
