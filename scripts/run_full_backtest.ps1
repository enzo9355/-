[CmdletBinding()]
param([string]$DataRoot = 'D:\AbsorbData', [int]$MaxItems = 25)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
if ($MaxItems -lt 1 -or $MaxItems -gt 500) { throw 'MaxItems is outside the safe range' }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi')
& $PythonExe -m stock_papi.batch.full_backtest_cli --root $DataRoot --verify-completion
$VerifyExitCode = $LASTEXITCODE
if ($VerifyExitCode -eq 0) { exit 0 }
if ($VerifyExitCode -ne 3) { exit $VerifyExitCode }
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi', 'yfinance')
$CommandExe = $env:ComSpec
if (-not $CommandExe -or -not (Test-Path -LiteralPath $CommandExe -PathType Leaf)) { throw 'Command processor was not found' }
$PythonCommand = '""{0}" -m stock_papi.batch.full_backtest_cli --root "{1}" --max-items {2} 2>&1"' -f $PythonExe, $DataRoot, $MaxItems
& $CommandExe /d /c $PythonCommand
$ExitCode = $LASTEXITCODE
exit $ExitCode
