[CmdletBinding()]
param([string]$DataRoot = 'D:\AbsorbData', [int]$MaxItems = 25)

function Test-AbsorbFullBacktestCompletedCheckpoint {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$DataRoot)

    $CheckpointPath = Join-Path $DataRoot 'checkpoints\jobs\full_backtest\current.json'
    if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
        return $false
    }
    try {
        $Checkpoint = Get-Content -LiteralPath $CheckpointPath -Raw -Encoding utf8 | ConvertFrom-Json
        return $Checkpoint.status -eq 'completed'
    } catch {
        Write-Warning 'Full backtest checkpoint could not be read; it will not be skipped'
        return $false
    }
}

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
if ($MaxItems -lt 1 -or $MaxItems -gt 500) { throw 'MaxItems is outside the safe range' }

# This guard intentionally uses only PowerShell/JSON so a completed run exits
# before selecting Python or importing yfinance through the stock pipeline.
if (Test-AbsorbFullBacktestCompletedCheckpoint -DataRoot $DataRoot) {
    Write-Output 'Full backtest checkpoint is already completed; skipping execution'
    exit 0
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi', 'yfinance')
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
$CommandExe = $env:ComSpec
if (-not $CommandExe -or -not (Test-Path -LiteralPath $CommandExe -PathType Leaf)) { throw 'Command processor was not found' }
$PythonCommand = '""{0}" -m stock_papi.batch.full_backtest_cli --root "{1}" --max-items {2} 2>&1"' -f $PythonExe, $DataRoot, $MaxItems
& $CommandExe /d /c $PythonCommand
$ExitCode = $LASTEXITCODE
exit $ExitCode
