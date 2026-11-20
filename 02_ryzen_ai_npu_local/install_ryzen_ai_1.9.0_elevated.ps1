<#
.SYNOPSIS
    Elevated installer helper for AMD Ryzen AI Software 1.9.0 on GOLDENRATIO1618.
.DESCRIPTION
    1. Ensures Administrator elevation (re-launches via UAC if needed).
    2. Registers C:\Windows\System32\AMD in Machine PATH so xrt-smi works globally.
    3. Reports any prior Ryzen AI installation (e.g. 1.7.1) before proceeding.
    4. Runs msiexec.exe on C:\Temp\RyzenAI_1.9.0_Stage\ryzen-ai-1.9.0.msi with isolated
       target paths:
         INSTALLDIR = C:\Program Files\RyzenAI\1.9.0
         VENV_DIR   = C:\Program Files\RyzenAI\ryzen-ai-1.9.0
    5. Verifies Scripts\activate.bat presence post-install.
#>

[CmdletBinding()]
param(
    [string]$MsiPath = 'C:\Temp\RyzenAI_1.9.0_Stage\ryzen-ai-1.9.0.msi',
    [string]$InstallDir = 'C:\Program Files\RyzenAI\1.9.0',
    [string]$VenvDir = 'C:\Program Files\RyzenAI\ryzen-ai-1.9.0',
    [string]$LogPath = 'C:\Temp\RyzenAI_1.9.0_Stage\ryzen_ai_1.9.0_install.log',
    [switch]$Silent
)

$ErrorActionPreference = 'Stop'

# Check Administrator privilege
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "Administrator privileges are required to install Ryzen AI 1.9.0 MSI and update Machine PATH."
    $argList = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-MsiPath', "`"$MsiPath`"",
        '-InstallDir', "`"$InstallDir`"",
        '-VenvDir', "`"$VenvDir`"",
        '-LogPath', "`"$LogPath`""
    )
    if ($Silent) { $argList += '-Silent' }
    Write-Host "Requesting UAC elevation..."
    Start-Process -FilePath 'powershell.exe' -ArgumentList $argList -Verb RunAs -Wait
    exit $LASTEXITCODE
}

if (-not (Test-Path $MsiPath)) {
    throw "Staged MSI not found at: $MsiPath"
}

Write-Host "=== Pre-Install Audit ===" -ForegroundColor Cyan
$existing171 = 'C:\Program Files\RyzenAI\1.7.1'
if (Test-Path $existing171) {
    Write-Host "Detected existing Ryzen AI 1.7.1 directory at: $existing171" -ForegroundColor Yellow
    Write-Host "Note: Ryzen AI 1.9.0 will install isolated into '$InstallDir' and venv '$VenvDir'."
}

# Ensure C:\Windows\System32\AMD is in Machine PATH
$amdSysPath = 'C:\Windows\System32\AMD'
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$machineParts = @()
if ($machinePath) {
    $machineParts = $machinePath -split ';' | Where-Object { $_ -and $_.Trim() -ne '' }
}
if ($machineParts -notcontains $amdSysPath) {
    $newMachinePath = ($machineParts + $amdSysPath) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $newMachinePath, 'Machine')
    Write-Host "Registered $amdSysPath in System (Machine) PATH." -ForegroundColor Green
} else {
    Write-Host "$amdSysPath is already registered in System (Machine) PATH." -ForegroundColor Green
}

# Build msiexec arguments
$msiArgs = @(
    '/i', "`"$MsiPath`"",
    "INSTALLDIR=`"$InstallDir`"",
    "VENV_DIR=`"$VenvDir`"",
    '/L*v', "`"$LogPath`""
)
if ($Silent) {
    $msiArgs += '/qn'
}

Write-Host "=== Launching Ryzen AI 1.9.0 MSI Installer ===" -ForegroundColor Cyan
Write-Host "Command: msiexec.exe $($msiArgs -join ' ')"
$proc = Start-Process -FilePath 'msiexec.exe' -ArgumentList $msiArgs -Wait -PassThru
Write-Host "msiexec exit code: $($proc.ExitCode)"

if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
    Write-Error "MSI installation failed with exit code $($proc.ExitCode). Inspect log: $LogPath"
    exit $proc.ExitCode
}

$activateBat = Join-Path $VenvDir 'Scripts\activate.bat'
if (Test-Path $activateBat) {
    Write-Host "SUCCESS: Isolated Ryzen AI 1.9.0 environment ready at $activateBat" -ForegroundColor Green
} else {
    Write-Warning "MSI completed, but '$activateBat' was not found directly. Check '$InstallDir' and '$LogPath' for actual conda/venv location."
    Get-ChildItem 'C:\Program Files\RyzenAI' -ErrorAction SilentlyContinue | Select-Object Name, FullName
}

if ($proc.ExitCode -eq 3010) {
    Write-Warning "Installer returned 3010 (reboot recommended). Please reboot manually when convenient."
}
