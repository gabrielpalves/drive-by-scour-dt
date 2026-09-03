<#
.SYNOPSIS
  One command per machine: runs every extracted bundle found beside this
  script, generation first and then training, unattended.

  For each generation bundle it runs the smokes, A00_Run, the state-1 parity
  gate, waits for _GENERATION_COMPLETE, and copies the finished dataset into
  the shared data root. Then, if a training bundle is present, it runs the
  capacity preflight and every job assigned to this host.

  Nothing here checks which machine you are on: a different MATLAB build or
  GPU is recorded as provenance, never blocked.

.PARAMETER Root
  Durable base directory for data/results/cache/studies/receipts. Defaults to
  <parent of the extracted bundles>\ttbi-campaign.

.PARAMETER Bundles
  Extracted bundle roots to run. Defaults to every folder beside this script
  (and this script's own folder) that carries a bundle manifest.

.PARAMETER MatlabExe / PythonExe
  Auto-detected when omitted: the newest C:\Program Files\MATLAB\R*, and
  python from PATH.

.PARAMETER RunTag
  Shared campaign run tag. Defaults to ttbi-<yyyyMMdd>.

.PARAMETER SkipSmokes
  Skip the six MATLAB physics/capability smokes. They are the check that the
  generated physics is right, so the default is to run them.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\run_campaign.ps1
#>
param(
    [string]$Root = "",
    [string[]]$Bundles = @(),
    [string]$MatlabExe = "",
    [string]$PythonExe = "",
    [string]$RunTag = "",
    [switch]$SkipSmokes
)

$ErrorActionPreference = "Stop"

function Resolve-MatlabExe {
    if ($MatlabExe) { return $MatlabExe }
    $found = Get-ChildItem "C:\Program Files\MATLAB\R*\bin\matlab.exe" -ErrorAction SilentlyContinue |
        Sort-Object { $_.Directory.Parent.Name } -Descending | Select-Object -First 1
    if ($found) { return $found.FullName }
    $onPath = Get-Command matlab -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "No MATLAB found. Pass -MatlabExe 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'."
}

function Resolve-PythonExe {
    if ($PythonExe) { return $PythonExe }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "No python found. Pass -PythonExe 'C:\path\to\python.exe'."
}

function Get-BundleKind {
    param([string]$Dir)
    if (Test-Path (Join-Path $Dir "generation_bundle_manifest.json")) { return "generation" }
    if (Test-Path (Join-Path $Dir "training_job_manifest.json")) { return "training" }
    return $null
}

function Find-Bundles {
    $candidates = @($PSScriptRoot)
    $candidates += @(Get-ChildItem -Path $PSScriptRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { $_.FullName })
    $parent = Split-Path $PSScriptRoot -Parent
    if ($parent) {
        $candidates += @(Get-ChildItem -Path $parent -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
    }
    $seen = @{}
    $out = @()
    foreach ($dir in $candidates) {
        $full = (Resolve-Path -LiteralPath $dir -ErrorAction SilentlyContinue).Path
        if (-not $full -or $seen.ContainsKey($full)) { continue }
        $seen[$full] = $true
        if (Get-BundleKind $full) { $out += $full }
    }
    return $out
}

function Copy-Dataset {
    param([string]$CaseFolder, [string]$DataRoot)
    $name = Split-Path -Leaf $CaseFolder
    $destination = Join-Path $DataRoot $name
    if (Test-Path -LiteralPath $destination) {
        Write-Host "  dataset already in the data root: $destination"
        return $destination
    }
    Write-Host "  copying $name into $DataRoot ..."
    Copy-Item -LiteralPath $CaseFolder -Destination $destination -Recurse -Force
    # A truncated copy would only surface days into training, so compare the
    # per-file digests of source and destination once, here.
    $sourceHashes = Get-ChildItem -LiteralPath $CaseFolder -Recurse -File |
        ForEach-Object { "{0}:{1}" -f $_.FullName.Substring($CaseFolder.Length), (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
    $destHashes = Get-ChildItem -LiteralPath $destination -Recurse -File |
        ForEach-Object { "{0}:{1}" -f $_.FullName.Substring($destination.Length), (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
    if (@(Compare-Object $sourceHashes $destHashes).Count -ne 0) {
        throw "Copy of $name into the data root does not match the source byte for byte."
    }
    Set-Content -Path "$destination.copied.sha256" -Value $destHashes -Encoding ascii
    Write-Host "  copy verified: $($destHashes.Count) files" -ForegroundColor Green
    return $destination
}

$MatlabExe = Resolve-MatlabExe
$PythonExe = Resolve-PythonExe
if (-not $Bundles -or $Bundles.Count -eq 0) { $Bundles = Find-Bundles }
if (-not $Bundles -or $Bundles.Count -eq 0) {
    throw "No extracted bundles found beside $PSScriptRoot. Extract the ZIPs side by side and rerun."
}
if (-not $Root) {
    $Root = Join-Path (Split-Path $PSScriptRoot -Parent) "ttbi-campaign"
}
if (-not $RunTag) { $RunTag = "ttbi-" + (Get-Date -Format "yyyyMMdd") }

$DataRoot = Join-Path $Root "data"
foreach ($d in @($DataRoot, (Join-Path $Root "results"), (Join-Path $Root "cache"),
                 (Join-Path $Root "studies"), (Join-Path $Root "receipts"))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$Generation = @($Bundles | Where-Object { (Get-BundleKind $_) -eq "generation" } | Sort-Object)
$Training = @($Bundles | Where-Object { (Get-BundleKind $_) -eq "training" } | Sort-Object)

Write-Host "MATLAB     : $MatlabExe"
Write-Host "Python     : $PythonExe"
Write-Host "Durable    : $Root  (run tag $RunTag)"
Write-Host "Generation : $($Generation.Count) bundle(s)"
$Generation | ForEach-Object { Write-Host "             $_" }
Write-Host "Training   : $($Training.Count) bundle(s)"
$Training | ForEach-Object { Write-Host "             $_" }
Write-Host ""

$Step = 0
foreach ($bundle in $Generation) {
    $Step++
    Write-Host "########## [$Step] GENERATION: $bundle ##########" -ForegroundColor Yellow
    # Hashtable splatting: an array would be bound positionally, so -BundleRoot
    # would arrive as a value rather than as a parameter name.
    $genArgs = @{
        BundleRoot = $bundle
        MatlabExe  = $MatlabExe
        PythonExe  = $PythonExe
    }
    if ($SkipSmokes) { $genArgs['SkipSmokes'] = $true }
    & (Join-Path $bundle "run_generation_host.ps1") @genArgs
    $case = Get-ChildItem -Path (Join-Path $bundle "scour_MATLAB\Results") -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "_GENERATION_COMPLETE") } |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $case) {
        throw "No completed dataset under $bundle after generation."
    }
    Copy-Dataset -CaseFolder $case.FullName -DataRoot $DataRoot | Out-Null
}

foreach ($bundle in $Training) {
    $Step++
    Write-Host "########## [$Step] TRAINING: $bundle ##########" -ForegroundColor Yellow
    & (Join-Path $bundle "run_training_host.ps1") `
        -BundleRoot $bundle -PythonExe $PythonExe `
        -DataRoot $DataRoot `
        -ResultsRoot (Join-Path $Root "results") `
        -CacheRoot (Join-Path $Root "cache") `
        -StudyRoot (Join-Path $Root "studies") `
        -ReceiptDir (Join-Path $Root "receipts") `
        -RunTag $RunTag
}

Write-Host ""
Write-Host "ALL DONE. Datasets and results are under $Root" -ForegroundColor Green
