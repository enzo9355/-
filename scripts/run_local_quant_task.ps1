[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = Join-Path $RepoRoot 'local_quant.py'
$BundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$PythonExe = if (Test-Path $BundledPython) { $BundledPython } elseif ($PythonCommand) { $PythonCommand.Source } else { $null }
if (-not $PythonExe) { throw 'Python executable was not found' }

$CacheRoot = Join-Path $DataRoot 'cache'
$TempRoot = Join-Path $CacheRoot 'tmp'
foreach ($Directory in @($CacheRoot, $TempRoot, (Join-Path $CacheRoot 'huggingface'), (Join-Path $CacheRoot 'pycache'))) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:XDG_CACHE_HOME = $CacheRoot
$env:HF_HOME = Join-Path $CacheRoot 'huggingface'
$env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot 'pycache'
$env:PYTHONPATH = Join-Path $RepoRoot '.deps'

& $PythonExe $Runner --root $DataRoot --insights

$TwLatestPath = Join-Path $DataRoot 'publish\quant\v1\latest-TW.json'
$TwLatestBefore = if (Test-Path -LiteralPath $TwLatestPath -PathType Leaf) {
    (Get-FileHash -LiteralPath $TwLatestPath -Algorithm SHA256).Hash
} else { $null }

& $PythonExe $Runner --root $DataRoot --run --market TW --limit 5000 --delay 0.5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$TwLatestAfter = if (Test-Path -LiteralPath $TwLatestPath -PathType Leaf) {
    (Get-FileHash -LiteralPath $TwLatestPath -Algorithm SHA256).Hash
} else { $null }
if ($TwLatestAfter -and $TwLatestAfter -ne $TwLatestBefore) {
    & $PythonExe -m reporting.cli --root $DataRoot --market TW
    $ReportExitCode = $LASTEXITCODE
    if ($ReportExitCode -ne 0) {
        Write-Warning "日報生成失敗（code $ReportExitCode）；保留上一份 latest，繼續美股批次。"
    }
} else {
    Write-Warning '台股沒有新 manifest；略過正式日報生成。'
}

$Now = Get-Date
$UsStart = $Now.Date.AddHours(5).AddMinutes(30)
if ($Now -lt $UsStart) {
    $WaitSeconds = [Math]::Ceiling(($UsStart - $Now).TotalSeconds)
    Start-Sleep -Seconds $WaitSeconds
}

& $PythonExe $Runner --root $DataRoot --run --market US --limit 5000 --delay 0.5
exit $LASTEXITCODE
