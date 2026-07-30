[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$TargetDate = (Get-Date).ToString('yyyy-MM-dd'),
    [switch]$PublishObservation,
    [switch]$ReconcileLegacyOverlaps
)
$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
try {
    $ParsedTargetDate = [DateTime]::ParseExact(
        $TargetDate,
        'yyyy-MM-dd',
        [Globalization.CultureInfo]::InvariantCulture
    )
}
catch { throw 'TargetDate must be YYYY-MM-DD' }
$Year = $ParsedTargetDate.Year
$PrimaryCalendarPath = if ($env:TWSE_CALENDAR_ARTIFACT) {
    $env:TWSE_CALENDAR_ARTIFACT
}
else {
    Join-Path $DataRoot "publish\calendars\v1\TW-$Year.json"
}
$CalendarPaths = New-Object System.Collections.Generic.List[string]
$CalendarPaths.Add($PrimaryCalendarPath)
foreach ($CandidateYear in @($Year - 1, $Year + 1)) {
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
$QuantArguments = @(
    '-m', 'stock_papi.batch.tw_official_post_close_cli',
    '--root', $DataRoot,
    '--target-market-date', $TargetDate,
    '--limit', '5000',
    '--delay', '0.5'
)
foreach ($Path in $CalendarPaths) {
    $QuantArguments += @('--calendar-artifact', $Path)
}
if ($ReconcileLegacyOverlaps) {
    $QuantArguments += '--reconcile-legacy-overlaps'
}
& $PythonExe @QuantArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Latest = Get-Content -LiteralPath (Join-Path $DataRoot 'publish\quant\v1\latest-TW.json') -Raw -Encoding utf8 | ConvertFrom-Json
$ManifestRelative = [string]$Latest.manifest
$ManifestPath = Join-Path $DataRoot "publish\quant\v1\$ManifestRelative"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$CandidateArguments = @(
    '-m', 'stock_papi.batch.observation_products_cli', 'build',
    '--root', $DataRoot,
    '--source-market-date', $Manifest.market_as_of,
    '--source-manifest', "quant/v1/$ManifestRelative",
    '--source-manifest-sha256', $Latest.manifest_sha256
)
foreach ($Path in $CalendarPaths) {
    $CandidateArguments += @('--calendar-artifact', $Path)
}
$CandidateJson = (& $PythonExe @CandidateArguments | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Candidate = $CandidateJson | ConvertFrom-Json
Write-Output $CandidateJson
if (-not $PublishObservation) { exit 0 }
& $PythonExe -m stock_papi.batch.observation_products_cli promote --root $DataRoot --candidate $Candidate.candidate_path
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot 'upload_local_quant.ps1') -DataRoot $DataRoot -RequireReportV2 -RequireDashboard -ObservationOnly
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m stock_papi.batch.cli notify --root $DataRoot --report-type post_close --audience admin --audience broadcast
exit $LASTEXITCODE
