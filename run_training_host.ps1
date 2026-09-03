<#
.SYNOPSIS
  Sequences one training host end to end: capacity preflight, then every job
  assigned to this host's manifest partition. Every provenance/hash check
  already inside capacity_preflight_compute.py and
  comprehensive_ablation_multidamage.py runs unchanged -- this script only
  removes the need to type the sequence by hand on each host.

.PARAMETER BundleRoot
  Absolute path to the extracted training bundle root. Defaults to this
  script's own folder, since the script ships inside the bundle.

.PARAMETER DataRoot / ResultsRoot / CacheRoot / StudyRoot / ReceiptDir / RunTag
  Durable, absolute paths and the shared run tag documented under
  "Durable paths and run identity" in each bundle's README_BUNDLE.md /
  README_CAMPAIGN.md. All required -- there are no defaults on purpose,
  because these must be genuinely durable storage outside the extracted
  workspace and shared consistently across LabA/LabB.

.PARAMETER TrainingJobManifest
  Path to training_job_manifest.json. Defaults to the copy shipped inside
  this bundle.

.EXAMPLE
  .\run_training_host.ps1 -DataRoot D:\ttbi\data -ResultsRoot D:\ttbi\results `
      -CacheRoot D:\ttbi\cache -StudyRoot D:\ttbi\study `
      -ReceiptDir D:\ttbi\receipts -RunTag paper1-2026-09
#>
param(
    [string]$BundleRoot = $PSScriptRoot,
    [string]$PythonExe = "python",
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$ResultsRoot,
    [Parameter(Mandatory = $true)][string]$CacheRoot,
    [Parameter(Mandatory = $true)][string]$StudyRoot,
    [Parameter(Mandatory = $true)][string]$ReceiptDir,
    [Parameter(Mandatory = $true)][string]$RunTag,
    [string]$TrainingJobManifest = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($TrainingJobManifest)) {
    $TrainingJobManifest = Join-Path $BundleRoot "training_job_manifest.json"
}
if (-not (Test-Path $TrainingJobManifest)) {
    throw "training_job_manifest.json not found at '$TrainingJobManifest'."
}

$env:TTBI_TRAINING_JOB_MANIFEST = (Resolve-Path $TrainingJobManifest).Path
$env:TTBI_DATA_ROOT = $DataRoot
$env:TTBI_RESULTS_ROOT = $ResultsRoot
$env:TTBI_CACHE_ROOT = $CacheRoot
$env:TTBI_STUDY_ROOT = $StudyRoot
$env:TTBI_EXECUTION_RECEIPT_DIR = $ReceiptDir
$env:TTBI_CAMPAIGN_RUN_TAG = $RunTag

Push-Location $BundleRoot
try {
    Write-Host "== Capacity preflight (all four execution blocks) ==" -ForegroundColor Cyan
    & $PythonExe -B capacity_preflight_compute.py --all-stages --receipt-dir $ReceiptDir
    if ($LASTEXITCODE -ne 0) {
        throw "capacity_preflight_compute.py failed (exit $LASTEXITCODE)."
    }

    Write-Host "== Manifest authentication ==" -ForegroundColor Cyan
    & $PythonExe -B comprehensive_ablation_multidamage.py --validate-manifest
    if ($LASTEXITCODE -ne 0) {
        throw "--validate-manifest failed (exit $LASTEXITCODE)."
    }

    Write-Host "== Assigned jobs on this host ==" -ForegroundColor Cyan
    $Listed = & $PythonExe -B comprehensive_ablation_multidamage.py --list-jobs
    if ($LASTEXITCODE -ne 0) {
        throw "--list-jobs failed (exit $LASTEXITCODE)."
    }
    # Only canonical job ids may be executed: anything else on stdout (a warning,
    # a banner from an imported package) must never be dispatched as a job.
    $Jobs = @($Listed | ForEach-Object { "$_".Trim() } |
        Where-Object { $_ -match '^[0-9a-f]{24}$' })
    if ($Jobs.Count -eq 0) {
        throw "No canonical job ids listed for this host's manifest partition."
    }
    $Expected = (Get-Content $env:TTBI_TRAINING_JOB_MANIFEST -Raw |
        ConvertFrom-Json).assigned_job_count
    if ($Jobs.Count -ne $Expected) {
        throw "Listed $($Jobs.Count) canonical job ids but the manifest assigns $Expected -- refusing to run a partial partition."
    }
    Write-Host "$($Jobs.Count) jobs assigned to this host (matches the manifest)."

    $i = 0
    foreach ($job in $Jobs) {
        $i++
        Write-Host "-- [$i/$($Jobs.Count)] $job --" -ForegroundColor Cyan
        & $PythonExe -B comprehensive_ablation_multidamage.py --execute-job $job
        if ($LASTEXITCODE -ne 0) {
            throw "Job '$job' failed (exit $LASTEXITCODE) -- stopping before the remaining $($Jobs.Count - $i) jobs."
        }
    }
    Write-Host "DONE. All $($Jobs.Count) assigned jobs on this host finished." -ForegroundColor Green
    Write-Host "Next: publish this host's results per README_CAMPAIGN.md so LabA/LabB can be consolidated."
}
finally {
    Pop-Location
}
