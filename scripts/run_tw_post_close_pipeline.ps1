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
$QuantRoot = [IO.Path]::GetFullPath(
    (Join-Path $DataRoot 'publish\quant\v1')
)
if (-not [IO.Directory]::Exists($QuantRoot)) {
    throw 'Quant publish root is missing'
}
if (
    ((Get-Item -LiteralPath $QuantRoot -Force).Attributes -band
        [IO.FileAttributes]::ReparsePoint) -ne 0
) {
    throw 'Quant publish root must not be a reparse point'
}
$LatestPath = Join-Path $QuantRoot 'latest-TW.json'
$LatestItem = Get-Item -LiteralPath $LatestPath -Force
if (
    ($LatestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
) {
    throw 'Latest pointer must not be a reparse point'
}
if ($LatestItem.PSIsContainer -or $LatestItem.Length -le 0 -or $LatestItem.Length -gt 100KB) {
    throw 'Latest pointer size is invalid'
}
$Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
$LatestSchema = [int]$Latest.schema_version
if ($LatestSchema -notin @(2, 3) -or [string]$Latest.market -ne 'TW') {
    throw 'Invalid TW latest pointer'
}
$ManifestRelative = [string]$Latest.manifest
$LatestManifestHash = ([string]$Latest.manifest_sha256).ToLowerInvariant()
if (
    $ManifestRelative -notmatch
        '^manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
    $LatestManifestHash -notmatch '^[0-9a-f]{64}$' -or
    -not $ManifestRelative.EndsWith(
        "-$($LatestManifestHash.Substring(0, 12)).json"
    )
) {
    throw 'Manifest path is not allowlisted'
}
$ManifestPath = [IO.Path]::GetFullPath(
    (Join-Path $QuantRoot $ManifestRelative.Replace('/', '\'))
)
$QuantRootPrefix = [IO.Path]::GetFullPath($QuantRoot).TrimEnd('\') + '\'
if (-not $ManifestPath.StartsWith(
    $QuantRootPrefix,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw 'Manifest path is not allowlisted'
}
$ManifestDirectory = Join-Path $QuantRoot 'manifests'
$ManifestItem = Get-Item -LiteralPath $ManifestPath -Force
if (
    ((Get-Item -LiteralPath $ManifestDirectory -Force).Attributes -band
        [IO.FileAttributes]::ReparsePoint) -ne 0 -or
    ($ManifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
) {
    throw 'Manifest path must not be a reparse point'
}
if ($ManifestItem.PSIsContainer -or $ManifestItem.Length -le 0 -or $ManifestItem.Length -gt 5MB) {
    throw 'Manifest size is invalid'
}
$ManifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ManifestHash -ne $LatestManifestHash) {
    throw 'Manifest hash mismatch'
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if (
    [int]$Manifest.schema_version -ne $LatestSchema -or
    [string]$Manifest.market -ne 'TW' -or
    [string]::IsNullOrWhiteSpace([string]$Manifest.generated_at) -or
    [string]$Manifest.generated_at -ne [string]$Latest.generated_at
) {
    throw 'Manifest identity mismatch'
}
$SourceMarketDate = $null
foreach ($CandidateMarketDate in @(
    if ($Manifest.PSObject.Properties['market_as_of']) {
        [string]$Manifest.market_as_of
    }
    if ($Manifest.PSObject.Properties['observation_as_of']) {
        [string]$Manifest.observation_as_of
    }
    if ($Manifest.PSObject.Properties['target_market_date']) {
        [string]$Manifest.target_market_date
    }
)) {
    if ($CandidateMarketDate -notmatch '^\d{4}-\d{2}-\d{2}$') {
        continue
    }
    try {
        $CanonicalMarketDate = Get-Date -Date $CandidateMarketDate `
            -Format 'yyyy-MM-dd' -ErrorAction Stop
    } catch {
        continue
    }
    if ($CanonicalMarketDate -eq $CandidateMarketDate) {
        $SourceMarketDate = $CanonicalMarketDate
        break
    }
}
if (-not $SourceMarketDate) {
    throw 'Manifest source market date is missing or invalid'
}
if (
    ($LatestSchema -eq 2 -and
        (
            $Manifest.PSObject.Properties['market_as_of'] -eq $null -or
            [string]$Manifest.market_as_of -ne $SourceMarketDate
        )) -or
    ($LatestSchema -eq 3 -and
        (
            $Manifest.PSObject.Properties['market_as_of'] -ne $null -or
            [string]$Manifest.observation_as_of -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Manifest.target_market_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Manifest.observation_as_of -ne
                [string]$Manifest.target_market_date
        ))
) {
    throw 'Manifest date contract is invalid'
}
if ($SourceMarketDate -ne $TargetDate) {
    throw 'Source market date does not match TargetDate'
}
$CandidateArguments = @(
    '-m', 'stock_papi.batch.observation_products_cli', 'build',
    '--root', $DataRoot,
    '--source-market-date', $SourceMarketDate,
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
