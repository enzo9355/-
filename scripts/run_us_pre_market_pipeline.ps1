[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$TargetDate = ''
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
& $PythonExe -c "
import datetime, json, pathlib
from reporting.schemas import ReportMetadataV2
from reporting.publisher import publish_report_v2

root = pathlib.Path(r'$DataRoot')
target_date = datetime.date.fromisoformat('$TargetDate')
now = datetime.datetime.now(datetime.timezone.utc)

# Load latest US post-close pointer to bind pre-market base
post_close_path = root / 'publish' / 'reports' / 'v2' / 'latest-US-post_close.json'
if not post_close_path.exists():
    raise RuntimeError('US pre-market requires published US post-close base')

post_close_ptr = json.loads(post_close_path.read_text(encoding='utf-8'))
meta_rel = post_close_ptr['metadata']
base_meta = json.loads((root / 'publish' / 'reports' / 'v2' / meta_rel).read_text(encoding='utf-8'))

doc = {
    'schema_version': 2,
    'kind': 'absorb-report',
    'product_mode': 'observation',
    'market': 'US',
    'report_type': 'pre_market',
    'source_market_date': base_meta['source_market_date'],
    'applicable_trading_date': target_date.isoformat(),
    'published_at': now.isoformat().replace('+00:00', 'Z'),
    'data_as_of': base_meta['data_as_of'],
    'observation_start_date': base_meta.get('observation_start_date', base_meta['data_as_of']),
    'observation_end_date': base_meta.get('observation_end_date', base_meta['data_as_of']),
    'source_manifest': base_meta['source_manifest'],
    'source_manifest_sha256': base_meta['source_manifest_sha256'],
    'model_versions': {},
    'title': f'ABSORB 美股盤前市場觀察報告 ({target_date})',
    'summary': f'美股 {target_date} 開盤前市場觀察與前一交易日結構摘要。',
    'content_sha256': base_meta['content_sha256'],
    'prediction_capability': base_meta['prediction_capability'],
}

res = publish_report_v2(root, doc)
print(f'Published US pre-market report: {res}')
"

if ($LASTEXITCODE -ne 0) {
    Write-Error "US PreMarket pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output "US PreMarket observation pipeline completed successfully for $TargetDate."
