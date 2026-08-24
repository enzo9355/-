[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$TargetDate = (Get-Date).ToString('yyyy-MM-dd')
)
$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
$Invariant = [Globalization.CultureInfo]::InvariantCulture
try {
    $ParsedTargetDate = [DateTime]::ParseExact(
        $TargetDate,
        'yyyy-MM-dd',
        $Invariant
    ).Date
}
catch { throw 'TargetDate must be YYYY-MM-DD' }
if ($ParsedTargetDate.ToString('yyyy-MM-dd', $Invariant) -ne $TargetDate) {
    throw 'TargetDate must be canonical YYYY-MM-DD'
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
. (Join-Path $PSScriptRoot 'pre_market_pipeline_guard.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi', 'pypdf')
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)

$Year = $ParsedTargetDate.Year
$PrimaryCalendarPath = if ($env:TWSE_CALENDAR_ARTIFACT) {
    $env:TWSE_CALENDAR_ARTIFACT
}
else {
    Join-Path $DataRoot "publish\calendars\v1\TW-$Year.json"
}
$CalendarPaths = New-Object System.Collections.Generic.List[string]
$CalendarPaths.Add($PrimaryCalendarPath)
foreach ($CandidateYear in @(($Year - 1), ($Year + 1))) {
    $CandidatePath = Join-Path $DataRoot "publish\calendars\v1\TW-$CandidateYear.json"
    if (
        (Test-Path -LiteralPath $CandidatePath -PathType Leaf) -and
        -not $CalendarPaths.Contains($CandidatePath)
    ) {
        $CalendarPaths.Add($CandidatePath)
    }
}
$CalendarArguments = @('-m', 'stock_papi.batch.cli', 'calendar-check')
foreach ($Path in $CalendarPaths) {
    $CalendarArguments += @('--calendar-artifact', $Path)
}
$CalendarArguments += @('--date', $TargetDate)
& $PythonExe @CalendarArguments
$CalendarExitCode = $LASTEXITCODE
if ($CalendarExitCode -eq 3) { Write-Output "$TargetDate is not a TW trading session; skipped"; exit 0 }
if ($CalendarExitCode -ne 0) { exit $CalendarExitCode }

if (Test-PreMarketCompletion -DataRoot $DataRoot -TargetDate $TargetDate) {
    Write-Output "TW pre-market observation report for $TargetDate is already verified and published end-to-end; skipping duplicate execution."
    exit 0
}

$LatestPath = Join-Path $DataRoot 'publish\reports\v2\latest-TW-post_close.json'
if (-not (Test-Path -LiteralPath $LatestPath -PathType Leaf)) { throw 'Verified post-close base is unavailable' }
$Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($Latest.product_mode -ne 'observation') { throw 'Verified post-close base is not observation mode' }
if (
    [string]$Latest.applicable_trading_date -ne $TargetDate -or
    [string]$Latest.source_market_date -notmatch '^\d{4}-\d{2}-\d{2}$'
) {
    throw "stale_post_close_base: post-close base ($([string]$Latest.source_market_date) -> $([string]$Latest.applicable_trading_date)) is not the verified completed session preceding $TargetDate"
}
$Arguments = @(
    '-m', 'stock_papi.batch.cli', 'pre-market',
    '--root', $DataRoot,
    '--applicable-trading-date', $TargetDate
)
foreach ($Path in $CalendarPaths) {
    $Arguments += @('--calendar-artifact', $Path)
}
if ($env:TW_PREMARKET_SOURCE_FILES) {
    foreach ($Source in $env:TW_PREMARKET_SOURCE_FILES.Split(';', [StringSplitOptions]::RemoveEmptyEntries)) {
        $ResolvedSource = (Resolve-Path -LiteralPath $Source).Path
        if (-not $ResolvedSource.StartsWith($DataRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Overnight source escaped data root' }
        $Arguments += @('--source-file', $ResolvedSource)
    }
}
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot 'upload_local_quant.ps1') -DataRoot $DataRoot -RequireReportV2 -ObservationOnly
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m stock_papi.batch.cli notify --root $DataRoot --report-type pre_market --audience admin --audience broadcast
exit $LASTEXITCODE
