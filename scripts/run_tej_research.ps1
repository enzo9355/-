[CmdletBinding()]
param(
    [ValidateSet('status', 'fetch', 'normalize', 'factor', 'challenger', 'shadow', 'quota', 'backfill', 'coverage', 'schema-evidence', 'prove-pit')]
    [string]$Command = 'status',
    [string]$DataRoot = 'D:\AbsorbData',
    [string[]]$Arguments = @()
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'TEJ DataRoot is not allowlisted'
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot

if (-not $env:TEJ_API_KEY) {
    $UserKey = [Environment]::GetEnvironmentVariable('TEJ_API_KEY', 'User')
    if ($UserKey) { $env:TEJ_API_KEY = $UserKey }
}
if (-not $env:TEJ_ENABLED) {
    $UserEnabled = [Environment]::GetEnvironmentVariable('TEJ_ENABLED', 'User')
    if ($UserEnabled) { $env:TEJ_ENABLED = $UserEnabled }
}

# The CLI itself returns a safe disabled status when TEJ_ENABLED is absent or
# false. No TW pipeline is imported or invoked from this script.
$PreviousLocation = Get-Location
try {
    Set-Location -LiteralPath $RepoRoot
    & $PythonExe -m stock_papi.research.tej_cli $Command '--root' $DataRoot @Arguments
    $ExitCode = $LASTEXITCODE
} finally {
    Set-Location -LiteralPath $PreviousLocation
}
exit $ExitCode
