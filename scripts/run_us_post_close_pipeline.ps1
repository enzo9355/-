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

# Resolve default TargetDate if not explicitly provided
if (-not $TargetDate) {
    $TargetDate = (& $PythonExe -c "
import datetime, zoneinfo
from stock_papi.integrations.market_data.us_calendar import get_us_exchange_holidays
ny_now = datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York'))
closed, early = get_us_exchange_holidays(ny_now.year)
cur = ny_now.date()
close_hour = 13 if cur in early else 16
if ny_now.hour < close_hour:
    cur -= datetime.timedelta(days=1)
while cur.weekday() >= 5 or cur in closed:
    cur -= datetime.timedelta(days=1)
print(cur.isoformat())
").Trim()
}

try {
    $ParsedTargetDate = [DateTime]::ParseExact(
        $TargetDate,
        'yyyy-MM-dd',
        [Globalization.CultureInfo]::InvariantCulture
    )
}
catch { throw 'TargetDate must be YYYY-MM-DD' }

$Year = $ParsedTargetDate.Year
$CalendarDir = Join-Path $DataRoot "publish\calendars\v1"
New-Item -ItemType Directory -Path $CalendarDir -Force | Out-Null
$PrimaryCalendarPath = Join-Path $CalendarDir "US-$Year.json"

# Ensure calendar artifact exists
if (-not (Test-Path -LiteralPath $PrimaryCalendarPath -PathType Leaf)) {
    & $PythonExe -c "
import json, pathlib
from stock_papi.integrations.market_data.us_calendar import generate_us_calendar_document
doc = generate_us_calendar_document($Year)
p = pathlib.Path(r'$PrimaryCalendarPath')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
"
}

# Verify trading session
& $PythonExe -c "
import json, datetime, pathlib, sys
from stock_papi.batch.calendar import TradingCalendar
doc = json.loads(pathlib.Path(r'$PrimaryCalendarPath').read_text(encoding='utf-8'))
cal = TradingCalendar.from_document(doc)
target = datetime.date.fromisoformat('$TargetDate')
if cal.is_session(target):
    sys.exit(0)
else:
    sys.exit(3)
"
$CalendarCheckCode = $LASTEXITCODE

if ($CalendarCheckCode -eq 3) {
    Write-Output "$TargetDate is not a US trading session; skipped"
    exit 0
}
if ($CalendarCheckCode -ne 0) { exit $CalendarCheckCode }

# Run US observation data collection & publish
Write-Output "Running US PostClose observation pipeline for $TargetDate..."
Invoke-AbsorbUsPipelineNativeCommand `
    -PythonExe $PythonExe `
    -Arguments @(
        '-m', 'stock_papi.batch.us_official_post_close_cli',
        '--root', $DataRoot,
        '--target-market-date', $TargetDate
    ) `
    -FailureLabel 'US PostClose pipeline'

Write-Output "US PostClose observation pipeline completed successfully for $TargetDate."

# Upload if requested
if ($PublishObservation) {
    Write-Output "Uploading US observation products to GCS..."
    & (Join-Path $PSScriptRoot 'upload_local_quant.ps1') `
        -DataRoot $DataRoot `
        -RequireReportV2 `
        -RequireDashboard `
        -ObservationOnly `
        -Market 'US'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
