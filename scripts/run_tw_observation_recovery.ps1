[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$Project = 'line-stock-bot-498908',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($Project -ne 'line-stock-bot-498908') {
    throw 'Project is not allowlisted'
}
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') {
    throw 'Bucket is not allowlisted'
}

$Invariant = [Globalization.CultureInfo]::InvariantCulture
try {
    $LocalToday = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
        [DateTimeOffset]::UtcNow,
        'Taipei Standard Time'
    ).Date
}
catch {
    throw 'Unable to determine Taipei local today'
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)

$Year = $LocalToday.Year
$CalendarPaths = New-Object System.Collections.Generic.List[string]
foreach ($CandidateYear in @(($Year - 1), $Year, ($Year + 1))) {
    $CandidatePath = Join-Path $DataRoot "publish\calendars\v1\TW-$CandidateYear.json"
    if (Test-Path -LiteralPath $CandidatePath -PathType Leaf) {
        $CalendarPaths.Add($CandidatePath)
    }
}
if ($CalendarPaths.Count -lt 1) {
    throw 'Verified TW calendar artifact is unavailable'
}
$LatestSessionArguments = @(
    '-m', 'stock_papi.batch.cli', 'calendar-latest-session'
)
foreach ($Path in $CalendarPaths) {
    $LatestSessionArguments += @('--calendar-artifact', $Path)
}
$LatestSessionArguments += @(
    '--before',
    $LocalToday.AddDays(-1).ToString('yyyy-MM-dd', $Invariant)
)
$LatestSessionJson = (& $PythonExe @LatestSessionArguments | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Calendar latest-session derivation failed with exit code $LASTEXITCODE"
}
$Derived = $LatestSessionJson | ConvertFrom-Json
$DerivedSession = [string]$Derived.latest_session
if ($DerivedSession -notmatch '^\d{4}-\d{2}-\d{2}$') {
    throw 'Calendar latest-session derivation returned an invalid date'
}
Write-Output "TW observation recovery derived latest completed session: $DerivedSession"

function Get-LocalPostCloseSourceDate {
    $LatestPath = Join-Path $DataRoot 'publish\reports\v2\latest-TW-post_close.json'
    if (-not (Test-Path -LiteralPath $LatestPath -PathType Leaf)) { return $null }
    try {
        $Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch { return $null }
    $SourceDate = [string]$Latest.source_market_date
    if ($SourceDate -notmatch '^\d{4}-\d{2}-\d{2}$') { return $null }
    return $SourceDate
}

function Get-RemotePostCloseSourceDate {
    param([string]$Gcloud)
    if ($null -eq $Gcloud) { return $null }
    try {
        $RemoteJson = (& $Gcloud storage cat "gs://$Bucket/reports/v2/latest-TW-post_close.json" 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { return $null }
        $Remote = $RemoteJson | ConvertFrom-Json
        $SourceDate = [string]$Remote.source_market_date
        if ($SourceDate -notmatch '^\d{4}-\d{2}-\d{2}$') { return $null }
        return $SourceDate
    }
    catch {
        return $null
    }
}

$Gcloud = $null
try {
    $Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
}
catch {
    Write-Output 'gcloud is unavailable; remote pointer read is skipped and catch-up will re-verify remotely'
}

$LocalSourceDate = Get-LocalPostCloseSourceDate
$RemoteSourceDate = Get-RemotePostCloseSourceDate -Gcloud $Gcloud
$LocalCurrent = $LocalSourceDate -eq $DerivedSession
$RemoteCurrent = $RemoteSourceDate -eq $DerivedSession
if ($LocalCurrent -and $RemoteCurrent) {
    Write-Output "TW observation pointers are current for $DerivedSession (local and verified remote); no-op."
    exit 0
}
if ($LocalCurrent -and $null -eq $RemoteSourceDate) {
    Write-Output "Local TW observation pointer is current for $DerivedSession but remote could not be verified; invoking supported catch-up which re-verifies remotely."
}
else {
    Write-Output "TW observation pointers are stale for $DerivedSession (local=$LocalSourceDate remote=$RemoteSourceDate); invoking supported catch-up for exactly that session."
}
& (Join-Path $PSScriptRoot 'catch_up_latest_completed_session.ps1') `
    -TargetDate $DerivedSession `
    -DataRoot $DataRoot `
    -Project $Project `
    -Bucket $Bucket
exit $LASTEXITCODE
