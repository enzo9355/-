[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$TargetDate = '',
    [switch]$PublishObservation
)
$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
. (Join-Path $PSScriptRoot 'us_pipeline_native.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)

# Resolve default TargetDate in America/New_York
if (-not $TargetDate) {
    $TargetDate = (& $PythonExe -c "
import datetime, zoneinfo
from stock_papi.integrations.market_data.us_calendar import get_us_exchange_holidays
ny_now = datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York'))
closed, _ = get_us_exchange_holidays(ny_now.year)
cur = ny_now.date()
while cur.weekday() >= 5 or cur in closed:
    cur += datetime.timedelta(days=1)
print(cur.isoformat())
").Trim()
}

Write-Output "Running US PreMarket observation pipeline for $TargetDate..."
Invoke-AbsorbUsPipelineNativeCommand `
    -PythonExe $PythonExe `
    -Arguments @(
        '-m', 'stock_papi.batch.us_pre_market_cli',
        '--root', $DataRoot,
        '--target-market-date', $TargetDate
    ) `
    -FailureLabel 'US PreMarket pipeline'

Write-Output "US PreMarket observation pipeline completed successfully for $TargetDate."

# Upload if requested
if ($PublishObservation) {
    Write-Output "Uploading US pre-market observation products to GCS..."
    & (Join-Path $PSScriptRoot 'upload_local_quant.ps1') `
        -DataRoot $DataRoot `
        -RequireReportV2 `
        -ObservationOnly `
        -Market 'US'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
