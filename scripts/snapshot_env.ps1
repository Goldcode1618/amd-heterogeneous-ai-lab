<#
.SYNOPSIS
    Captures a publication-grade hardware, driver, and Python environment snapshot
    for GOLDENRATIO1618 into 00_RESEARCH_LIBRARY\ENVIRONMENT_SNAPSHOTS\.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $workspaceRoot '00_RESEARCH_LIBRARY\ENVIRONMENT_SNAPSHOTS'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$utcNow = (Get-Date).ToUniversalTime()
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$stamp = $utcNow.ToString('yyyyMMdd_HHmmss', $inv)
$isoUtc = $utcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $inv)

$osInfo = Get-CimInstance Win32_OperatingSystem
$csInfo = Get-CimInstance Win32_ComputerSystem
$biosInfo = Get-CimInstance Win32_BIOS
$cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
$gpus = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM, VideoProcessor

$xrtExe = 'C:\Windows\System32\AMD\xrt-smi.exe'
$xrtExamine = if (Test-Path $xrtExe) {
    (& $xrtExe examine 2>&1) -join "`n"
} else {
    'xrt-smi.exe not found'
}

$nvidiaSmi = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1) -join "`n"
} else {
    'nvidia-smi not found'
}

$wslKernel = (wsl -d Ubuntu -- uname -r 2>&1) -join "`n"

$ryzenAiPy = 'C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\python.exe'
$ryzenAiPackages = if (Test-Path $ryzenAiPy) {
    @(& $ryzenAiPy -m pip list --format=freeze 2>$null)
} else {
    @()
}

$tokenLabPy = Join-Path $workspaceRoot '04_CLOUD_AND_API_LABS\token_lab\.venv\Scripts\python.exe'
$tokenLabPackages = if (Test-Path $tokenLabPy) {
    @(& uv pip list --python $tokenLabPy --format=freeze 2>$null)
} else {
    @()
}

$gitCommit = (git -C $workspaceRoot rev-parse --short HEAD 2>$null)
if (-not $gitCommit) { $gitCommit = 'uninitialized' }

$snapshot = [ordered]@{
    timestamp_utc   = $isoUtc
    host_node       = $env:COMPUTERNAME
    git_commit      = $gitCommit
    os              = [ordered]@{
        caption      = $osInfo.Caption
        version      = $osInfo.Version
        build_number = $osInfo.BuildNumber
        arch         = $osInfo.OSArchitecture
    }
    hardware        = [ordered]@{
        manufacturer = $csInfo.Manufacturer
        model        = $csInfo.Model
        total_ram_gb = [math]::Round($csInfo.TotalPhysicalMemory / 1GB, 2)
        bios_version = $biosInfo.SMBIOSBIOSVersion
        cpu_name     = $cpuInfo.Name.Trim()
        cpu_cores    = $cpuInfo.NumberOfCores
        cpu_threads  = $cpuInfo.NumberOfLogicalProcessors
        gpus         = @($gpus | ForEach-Object {
            [ordered]@{
                name           = $_.Name
                driver_version = $_.DriverVersion
                adapter_ram_mb = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM / 1MB, 0) } else { $null }
            }
        })
    }
    xrt_smi_examine = $xrtExamine
    nvidia_smi      = $nvidiaSmi
    wsl2_kernel     = $wslKernel.Trim()
    environments    = [ordered]@{
        ryzen_ai_1_9_0_pkg_count = $ryzenAiPackages.Count
        ryzen_ai_1_9_0_packages  = $ryzenAiPackages
        token_lab_pkg_count      = $tokenLabPackages.Count
        token_lab_packages       = $tokenLabPackages
    }
}

$jsonPath = Join-Path $outDir "snapshot_$stamp.json"
$jsonText = $snapshot | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText($jsonPath, $jsonText, (New-Object System.Text.UTF8Encoding($false)))
$jsonHash = (Get-FileHash -Path $jsonPath -Algorithm SHA256).Hash.ToLowerInvariant()

$mdLines = @(
    "# Environment Snapshot — $($snapshot.host_node)",
    "",
    "- **Captured (UTC):** ``$isoUtc``",
    "- **Git Commit:** ``$gitCommit``",
    "- **JSON Artifact:** ``snapshot_$stamp.json``",
    "- **JSON SHA-256:** ``$jsonHash``",
    "",
    "## Host Platform",
    "- **OS:** $($osInfo.Caption) (Version $($osInfo.Version), Build $($osInfo.BuildNumber))",
    "- **CPU:** $($cpuInfo.Name.Trim()) ($($cpuInfo.NumberOfCores)C / $($cpuInfo.NumberOfLogicalProcessors)T)",
    "- **RAM:** $([math]::Round($csInfo.TotalPhysicalMemory / 1GB, 2)) GB",
    "- **BIOS:** $($biosInfo.SMBIOSBIOSVersion)",
    "- **WSL2 Ubuntu Kernel:** ``$($wslKernel.Trim())``",
    "",
    "## Display & Compute Adapters"
)
foreach ($g in $gpus) {
    $mdLines += "- **$($g.Name)** — Driver ``$($g.DriverVersion)``"
}
$fence = '```'
$mdLines += @(
    "",
    "## NVIDIA dGPU Isolation Check",
    "${fence}text",
    $nvidiaSmi,
    $fence,
    "",
    "## XRT NPU Diagnostics (``xrt-smi examine``)",
    "${fence}text",
    $xrtExamine,
    $fence,
    "",
    "## Python Environment Inventory",
    "- ``ryzen-ai-1.9.0``: $($ryzenAiPackages.Count) packages captured",
    "- ``04_CLOUD_AND_API_LABS/token_lab/.venv``: $($tokenLabPackages.Count) packages captured"
)

$mdPath = Join-Path $outDir 'latest_snapshot.md'
[System.IO.File]::WriteAllText($mdPath, ($mdLines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Snapshot JSON : $jsonPath" -ForegroundColor Green
Write-Host "Snapshot SHA256: $jsonHash" -ForegroundColor Cyan
Write-Host "Latest Summary: $mdPath" -ForegroundColor Green
