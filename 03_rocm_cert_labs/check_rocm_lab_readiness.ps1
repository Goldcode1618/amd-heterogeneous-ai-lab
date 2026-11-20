<#
.SYNOPSIS
    Checks Docker Desktop, WSL2 backend, and NVIDIA RTX 3050 isolation readiness
    for AMD ROCm Certified Associate Labs (Modules 1-5).
#>

$ErrorActionPreference = 'Continue'

Write-Host "=== 1. WSL2 Status ===" -ForegroundColor Cyan
wsl -l -v

Write-Host "`n=== 2. WSL2 /dev/accel Check ===" -ForegroundColor Cyan
wsl -d Ubuntu -- bash -c "ls -la /dev/accel 2>&1 || echo 'NO_DEV_ACCEL (expected on WSL2)'; uname -r"

Write-Host "`n=== 3. Docker Desktop / CLI Readiness ===" -ForegroundColor Cyan
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
    $fallbackDocker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
    if (Test-Path $fallbackDocker) {
        $dockerCmd = $fallbackDocker
    }
}

if ($dockerCmd) {
    & $dockerCmd --version
    & $dockerCmd info --format "ServerVersion={{.ServerVersion}} OSType={{.OSType}} Driver={{.Driver}}" 2>&1
} else {
    Write-Warning "Docker CLI is NOT installed on Windows PATH or WSL2."
    Write-Host "Action needed for Docker-based ROCm labs:" -ForegroundColor Yellow
    Write-Host "  winget install -e --id Docker.DockerDesktop"
    Write-Host "  Enable 'Use the WSL 2 based engine' in Docker Desktop Settings -> Resources -> WSL Integration."
}

Write-Host "`n=== 4. NVIDIA RTX 3050 Isolation Check ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
