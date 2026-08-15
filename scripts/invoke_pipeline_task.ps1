[CmdletBinding()]
param(
    [ValidateSet('TW-PostClose', 'TW-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
    [string]$Job,
    [string]$DataRoot = 'D:\AbsorbData'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -ne 'D:\AbsorbData') { throw 'Data root is not allowlisted' }
. (Join-Path $PSScriptRoot 'native_process.ps1')

$Definitions = @{
    'TW-PostClose' = @{ Script = 'run_tw_post_close_pipeline.ps1'; Arguments = @('-PublishObservation') }
    'TW-PreMarket' = @{ Script = 'run_tw_pre_market_pipeline.ps1'; Arguments = @() }
    'FullBacktest' = @{ Script = 'run_full_backtest.ps1'; Arguments = @('-MaxItems', '500') }
    'US-Daily' = @{ Script = 'run_us_daily.ps1'; Arguments = @() }
    'WeeklyModel' = @{ Script = 'run_weekly_model.ps1'; Arguments = @() }
    'ReportUploadRecovery' = @{ Script = 'upload_local_quant.ps1'; Arguments = @('-RequireReportV2', '-RequireDashboard', '-ObservationOnly') }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Definition = $Definitions[$Job]
$ScriptPath = Join-Path $PSScriptRoot $Definition.Script
if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) { throw "Task wrapper not found: $ScriptPath" }

$LogDirectory = Join-Path $DataRoot 'logs\tasks'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$StartedAt = [DateTimeOffset]::Now
$LogPath = Join-Path $LogDirectory ("{0}-{1:yyyyMMdd}.log" -f $Job, $StartedAt)
$StatusPath = Join-Path $LogDirectory ("current-{0}.json" -f $Job)
$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) { throw 'PowerShell executable was not found' }
$Arguments = @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $ScriptPath, '-DataRoot', $DataRoot) + $Definition.Arguments
$ObservationWriterJobs = @(
    'TW-PostClose',
    'TW-PreMarket',
    'WeeklyModel',
    'ReportUploadRecovery'
)
$ObservationWriterMutexName = 'Global\ABSORB-TW-Observation-Writer'
$ObservationWriterMutex = $null
$ObservationWriterMutexHeld = $false

try {
    if ($ObservationWriterJobs -contains $Job) {
        $ObservationWriterMutex = [Threading.Mutex]::new(
            $false,
            $ObservationWriterMutexName
        )
        try {
            if (-not $ObservationWriterMutex.WaitOne(0)) {
                throw 'Another TW observation writer is active'
            }
        }
        catch [Threading.AbandonedMutexException] {
            throw 'TW observation writer mutex was abandoned; refusing to run'
        }
        $ObservationWriterMutexHeld = $true
    }
    $Result = Invoke-NativeProcessStreaming `
        -FilePath $PowerShellExe `
        -Arguments $Arguments `
        -LogPath $LogPath `
        -AllowFailure
    $ExitCode = $Result.exit_code
    if ($ExitCode -ne 0) { throw "Pipeline exited with code $ExitCode" }
    @{ job = $Job; started_at = $StartedAt.ToString('o'); finished_at = [DateTimeOffset]::Now.ToString('o'); success = $true; exit_code = 0; log = $LogPath } |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $StatusPath -Encoding utf8
    if ($Job -eq 'FullBacktest') {
        $CheckpointPath = Join-Path $DataRoot 'checkpoints\jobs\full_backtest\current.json'
        $Checkpoint = if (Test-Path -LiteralPath $CheckpointPath) { Get-Content -LiteralPath $CheckpointPath -Raw -Encoding utf8 | ConvertFrom-Json } else { $null }
        if ($Checkpoint.status -eq 'completed') {
            try { Disable-ScheduledTask -TaskName 'ABSORB-FullBacktest' -ErrorAction Stop | Out-Null } catch { Write-Warning 'Unable to disable completed full-backtest task' }
        }
    }
} catch {
    @{ job = $Job; started_at = $StartedAt.ToString('o'); finished_at = [DateTimeOffset]::Now.ToString('o'); success = $false; exit_code = if ($ExitCode -is [int]) { $ExitCode } else { 1 }; error = $_.Exception.Message; log = $LogPath } |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $StatusPath -Encoding utf8
    throw
} finally {
    if ($null -ne $ObservationWriterMutex) {
        if ($ObservationWriterMutexHeld) {
            try { $ObservationWriterMutex.ReleaseMutex() } catch { }
        }
        $ObservationWriterMutex.Dispose()
    }
}
