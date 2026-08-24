[CmdletBinding()]
param([string]$DataRoot = 'D:\AbsorbData')
$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi', 'yfinance')
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
& $PythonExe (Join-Path $RepoRoot 'local_quant.py') --root $DataRoot --run --market US --limit 5000 --delay 0.5
exit $LASTEXITCODE
