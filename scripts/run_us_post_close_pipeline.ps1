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
& $PythonExe -c "
import datetime, json, os, pathlib, sys
from stock_papi.integrations.market_data.us_universe import get_us_symbols
from stock_papi.integrations.market_data.us_market_data import fetch_us_stock_history
from local_quant import write_stock_artifact, publish_market_snapshot
from stock_papi.batch.observation_products import build_observation_dashboard, promote_observation_candidate, write_observation_candidate
from stock_papi.config.capabilities import PredictionCapabilityState
from reporting.source_loader import load_report_source
from reporting.schemas import ReportMetadataV2

root = pathlib.Path(r'$DataRoot')
target_date = datetime.date.fromisoformat('$TargetDate')
now = datetime.datetime.now(datetime.timezone.utc)

# 1. Universe
symbols = get_us_symbols(root, now=now)
print(f'US active universe count: {len(symbols)}')

# 2. Build stock artifacts for target session
completed = []
failed = []
for sym in symbols:
    try:
        df = fetch_us_stock_history(sym, target_market_date=target_date)
        daily = json.loads(df.reset_index().to_json(orient='records', date_format='iso', date_unit='ms'))
        latest = daily[-1]
        as_of = str(latest.get('Date', '')).split('T', 1)[0]
        if as_of != target_date.isoformat():
            failed.append(sym)
            continue
        payload = {
            'schema_version': 2,
            'market': 'US',
            'symbol': sym,
            'as_of': as_of,
            'target_market_date': as_of,
            'observation_as_of': as_of,
            'latest_regular_price_date': as_of,
            'observation_kind': 'regular_price',
            'lineage': {
                'source_schema_version': 'us-market-data-v1',
                'observation_as_of': as_of,
                'latest_regular_price_date': as_of,
                'observation_kind': 'regular_price',
            },
            'rows': len(daily),
            'latest': latest,
            'backtest': {},
            'daily': daily,
        }
        write_stock_artifact(root, 'US', sym, payload)
        completed.append(sym)
    except Exception as exc:
        failed.append(sym)

print(f'US stock artifacts: completed={len(completed)}, failed={len(failed)}')

# 3. Publish Manifest v4 with >95% coverage contract
manifest_path = publish_market_snapshot(
    root, 'US', symbols, generated_at=now,
    failed_symbols=failed, target_market_date=target_date,
    unavailable_symbols=failed,
)
print(f'Published US Manifest v4: {manifest_path}')

# 4. Build Observation Dashboard & Report Candidate
source = load_report_source(root, market='US')

industry_map = {'ETF專區': ['SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'IVV', 'SOXX', 'SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'VNQ', 'GLD', 'TLT', 'VTI', 'VEA', 'VWO', 'BND']}
pred_cap = PredictionCapabilityState.from_environment()
dashboard = build_observation_dashboard(source, industry_map, pred_cap, generated_at=now, today=target_date)

report_metadata = {
    'schema_version': 2,
    'kind': 'absorb-report',
    'product_mode': 'observation',
    'market': 'US',
    'report_type': 'post_close',
    'source_market_date': target_date.isoformat(),
    'applicable_trading_date': target_date.isoformat(),
    'published_at': now.isoformat().replace('+00:00', 'Z'),
    'data_as_of': target_date.isoformat(),
    'observation_start_date': target_date.isoformat(),
    'observation_end_date': target_date.isoformat(),
    'source_manifest': f'quant/v1/{source.manifest.manifest_path}',
    'source_manifest_sha256': source.manifest.manifest_sha256,
    'model_versions': {},
    'title': f'ABSORB 美股盤後市場觀察報告 ({target_date})',
    'summary': f'美股 {target_date} 交易日收盤觀察與市場結構概況。',
    'content_sha256': source.manifest.manifest_sha256,
    'prediction_capability': pred_cap.to_document(),
}

cand_dir = write_observation_candidate(root, report_metadata, dashboard)
promoted = promote_observation_candidate(root, cand_dir)
print(f'Successfully promoted US observation candidate: {promoted}')
"

if ($LASTEXITCODE -ne 0) {
    Write-Error "US PostClose pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output "US PostClose observation pipeline completed successfully for $TargetDate."

# Upload if requested
if ($PublishObservation) {
    Write-Output "Uploading US observation products to GCS..."
    & (Join-Path $PSScriptRoot 'upload_local_quant.ps1') `
        -DataRoot $DataRoot `
        -RequireReportV2 `
        -RequireDashboard `
        -ObservationOnly
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
