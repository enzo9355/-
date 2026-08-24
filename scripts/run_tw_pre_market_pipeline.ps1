[CmdletBinding()]
param([string]$DataRoot = 'D:\AbsorbData')
$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi', 'pypdf')
$LatestPath = Join-Path $DataRoot 'publish\reports\v2\latest-TW-post_close.json'
if (-not (Test-Path -LiteralPath $LatestPath -PathType Leaf)) { throw 'Verified post-close base is unavailable' }
$Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($Latest.product_mode -ne 'observation') { throw 'Verified post-close base is not observation mode' }
$ApplicableDate = [string]$Latest.applicable_trading_date
$Arguments = @('-m', 'stock_papi.batch.cli', 'pre-market', '--root', $DataRoot, '--applicable-trading-date', $ApplicableDate)
if ($env:TW_PREMARKET_SOURCE_FILES) {
    foreach ($Source in $env:TW_PREMARKET_SOURCE_FILES.Split(';', [StringSplitOptions]::RemoveEmptyEntries)) {
        $ResolvedSource = (Resolve-Path -LiteralPath $Source).Path
        if (-not $ResolvedSource.StartsWith($DataRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Overnight source escaped data root' }
        $Arguments += @('--source-file', $ResolvedSource)
    }
}
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot 'upload_local_quant.ps1') -DataRoot $DataRoot -RequireReportV2 -ObservationOnly
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m stock_papi.batch.cli notify --root $DataRoot --report-type pre_market --audience admin --audience broadcast
exit $LASTEXITCODE
