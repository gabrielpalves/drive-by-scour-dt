<#
.SYNOPSIS
  Sequences one MATLAB generation run end to end: capability/physics smokes,
  A00_Run, a raw-parity gate on the first generated state, then waits for
  full completion. Every provenance/hash check already inside A00_Run.m and
  check_raw_parity.py runs unchanged -- this script only removes the need to
  type the sequence by hand on each host.

.PARAMETER BundleRoot
  Absolute path to the extracted generation bundle root (the folder that
  contains scour_MATLAB\). Defaults to this script's own folder, since the
  script ships inside the bundle.

.PARAMETER MatlabExe
  Path to matlab.exe. Defaults to "matlab" resolved from PATH. Pass an
  explicit path if more than one MATLAB release is installed on this host.

.PARAMETER PythonExe
  Path to the python executable used for check_raw_parity.py. Defaults to
  "python" resolved from PATH.

.EXAMPLE
  .\run_generation_host.ps1
  .\run_generation_host.ps1 -MatlabExe 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
#>
param(
    [string]$BundleRoot = $PSScriptRoot,
    [string]$MatlabExe = "matlab",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ScourDir = Join-Path $BundleRoot "scour_MATLAB"
if (-not (Test-Path $ScourDir)) {
    throw "scour_MATLAB not found under '$BundleRoot' -- pass -BundleRoot explicitly."
}
$ResultsDir = Join-Path $ScourDir "Results"

function Invoke-MatlabBatch {
    param([string]$Command)
    & $MatlabExe -batch $Command
    if ($LASTEXITCODE -ne 0) {
        throw "MATLAB command failed (exit $LASTEXITCODE): $Command"
    }
}

Write-Host "== Capability/physics smokes ==" -ForegroundColor Cyan
$Smokes = @(
    'smoke_audit', 'smoke_geometry', 'smoke_stage3',
    'smoke_familytable', 'smoke_b54_overlap_parity', 'smoke_generation_worker'
)
foreach ($smoke in $Smokes) {
    Write-Host "-- $smoke"
    Invoke-MatlabBatch "cd('$ScourDir'); $smoke"
}
Write-Host "All smokes passed." -ForegroundColor Green

Write-Host "== A00_Run (production generation) ==" -ForegroundColor Cyan
# A leftover Results/ from an earlier attempt must never be mistaken for this
# run's first state: the parity gate would then check stale data.
$PreExistingStates = @(
    Get-ChildItem -Path $ResultsDir -Filter "0001.mat" -Recurse `
        -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
)
if ($PreExistingStates.Count -gt 0) {
    Write-Warning "$($PreExistingStates.Count) pre-existing 0001.mat under $ResultsDir; only a newly written state will be gated."
}
$LogOut = Join-Path $BundleRoot "a00_run.log"
$LogErr = Join-Path $BundleRoot "a00_run.err.log"
# matlab.exe is only a launcher: started directly it returns immediately (exit 0,
# empty log) while the real MATLAB runs detached. Going through a PowerShell
# child makes the call operator hold the output pipe until MATLAB itself exits,
# and gives us one PID whose process tree can be killed if the parity gate fails.
$Launcher = Join-Path $BundleRoot "a00_run_launcher.ps1"
Set-Content -Path $Launcher -Encoding ascii -Value @(
    "& '$MatlabExe' -batch 'cd(''$ScourDir''); A00_Run'",
    'exit $LASTEXITCODE'
)
$Proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Launcher) `
    -WorkingDirectory $ScourDir `
    -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr `
    -PassThru -NoNewWindow
# Touching .Handle once makes Windows PowerShell keep the process handle, without
# which $Proc.ExitCode stays $null after the process ends.
$null = $Proc.Handle
Write-Host "A00_Run started (PID $($Proc.Id)); log: $LogOut"
Write-Host "Waiting for the first state (0001.mat)..."

function Find-FirstState {
    return Get-ChildItem -Path $ResultsDir -Filter "0001.mat" -Recurse `
        -ErrorAction SilentlyContinue |
        Where-Object { $PreExistingStates -notcontains $_.FullName } |
        Select-Object -First 1
}

$FirstState = $null
$Started = Get-Date
$Polls = 0
while ($true) {
    $FirstState = Find-FirstState
    if ($FirstState) { break }
    if ($Proc.HasExited) {
        # The state may have landed between the last poll and the exit, and a
        # short stage can finish entirely inside one poll interval.
        $FirstState = Find-FirstState
        break
    }
    $Polls++
    # The run folder and case_info appear within minutes but the first state is
    # a full 50-passage solve, so report progress instead of going silent.
    if ($Polls % 30 -eq 0) {
        $RunFolder = Get-ChildItem -Path $ResultsDir -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime | Select-Object -Last 1
        $Where = if ($RunFolder) { $RunFolder.Name } else { "run folder not created yet" }
        Write-Host ("  still generating state 1 after {0:N0} min ({1})" -f `
            ((Get-Date) - $Started).TotalMinutes, $Where)
    }
    Start-Sleep -Seconds 10
}
if (-not $FirstState) {
    # WaitForExit() first: ExitCode is not materialised on a -PassThru object
    # until the handle has been waited on.
    $Proc.WaitForExit()
    Write-Host "---- last 60 lines of $LogOut / $LogErr ----"
    Get-Content $LogOut, $LogErr -ErrorAction SilentlyContinue -Tail 60
    throw "A00_Run exited (code $($Proc.ExitCode)) before producing 0001.mat."
}

$CaseFolder = $FirstState.Directory
$RelativeCase = "Results/$($CaseFolder.Name)"
Write-Host "First state ready: $($CaseFolder.FullName)" -ForegroundColor Cyan
Write-Host "== Raw-parity gate on state 1 ==" -ForegroundColor Cyan

try {
    Invoke-MatlabBatch "cd('$ScourDir'); smoke_raw_parity('$RelativeCase')"
    Push-Location $BundleRoot
    try {
        & $PythonExe (Join-Path $BundleRoot "check_raw_parity.py") $CaseFolder.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "check_raw_parity.py failed (exit $LASTEXITCODE)."
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Host "PARITY CHECK FAILED -- stopping generation now." -ForegroundColor Red
    if (-not $Proc.HasExited) {
        & taskkill /PID $Proc.Id /T /F | Out-Null
    }
    throw
}
Write-Host "Raw-parity gate PASSED." -ForegroundColor Green

Write-Host "== Waiting for A00_Run to finish (this can take PC-days) ==" -ForegroundColor Cyan
$Proc.WaitForExit()
if ($Proc.ExitCode -ne 0) {
    Write-Host "---- last 60 lines of $LogOut / $LogErr ----"
    Get-Content $LogOut, $LogErr -ErrorAction SilentlyContinue -Tail 60
    throw "A00_Run exited with code $($Proc.ExitCode)."
}

$Marker = Join-Path $CaseFolder.FullName "_GENERATION_COMPLETE"
if (-not (Test-Path $Marker)) {
    throw "A00_Run exited 0 but _GENERATION_COMPLETE is missing in $($CaseFolder.FullName) -- do not treat this as finished."
}
Write-Host "DONE. $($CaseFolder.FullName) is complete (_GENERATION_COMPLETE present)." -ForegroundColor Green
Write-Host "Next: copy it byte-for-byte into TTBI_DATA_ROOT with Copy-TtbiDataset (see README_CAMPAIGN.md)."
