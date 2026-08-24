[CmdletBinding()]
param(
    [ValidateSet('TW-PostClose', 'TW-PreMarket', 'US-PostClose', 'US-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
    [string]$Job,
    [string]$DataRoot = 'D:\AbsorbData'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -ne 'D:\AbsorbData') { throw 'Data root is not allowlisted' }
. (Join-Path $PSScriptRoot 'native_process.ps1')

function Get-AbsorbPipelineMutexPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('TW-PostClose', 'TW-PreMarket', 'US-PostClose', 'US-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
        [string]$Job
    )

    # Lock ordering is always market -> publication. Pre-market work takes
    # only its own market lock, so US work cannot block TW pre-market.
    $Plan = New-Object 'System.Collections.Generic.List[object]'
    $Market = switch ($Job) {
        { $_ -in @('TW-PostClose', 'TW-PreMarket') } { 'TW'; break }
        { $_ -in @('US-PostClose', 'US-PreMarket') } { 'US'; break }
        default { $null }
    }
    if ($Market) {
        $Plan.Add([pscustomobject]@{
            scope = 'market'
            market = $Market
            mutex_name = "Global\ABSORB-$Market-Observation-Writer"
            wait_milliseconds = 120000
        })
    }
    if ($Job -in @('TW-PostClose', 'US-PostClose', 'WeeklyModel', 'ReportUploadRecovery')) {
        $Plan.Add([pscustomobject]@{
            scope = 'publication'
            market = $null
            mutex_name = 'Global\ABSORB-Observation-Publication-Writer'
            wait_milliseconds = 300000
        })
    }
    return @($Plan.ToArray())
}

function Enter-AbsorbPipelineMutexPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory)][object[]]$Plan)

    $Leases = New-Object 'System.Collections.Generic.List[object]'
    try {
        foreach ($Entry in $Plan) {
            $WaitMilliseconds = [int]$Entry.wait_milliseconds
            if ($WaitMilliseconds -lt 1 -or $WaitMilliseconds -gt 600000) {
                throw 'Pipeline mutex wait is outside the safe range'
            }
            $Mutex = [Threading.Mutex]::new($false, [string]$Entry.mutex_name)
            $Watch = [Diagnostics.Stopwatch]::StartNew()
            try {
                $Acquired = $Mutex.WaitOne($WaitMilliseconds)
            } catch [Threading.AbandonedMutexException] {
                $Mutex.Dispose()
                throw "Pipeline $($Entry.scope) mutex was abandoned; refusing to run"
            } finally {
                $Watch.Stop()
            }
            if (-not $Acquired) {
                $Mutex.Dispose()
                throw [TimeoutException]::new("Timed out waiting for pipeline $($Entry.scope) mutex")
            }
            $Leases.Add([pscustomobject]@{
                mutex = $Mutex
                receipt = [ordered]@{
                    scope = [string]$Entry.scope
                    market = $Entry.market
                    mutex_name = [string]$Entry.mutex_name
                    wait_milliseconds = $WaitMilliseconds
                    waited_milliseconds = [int]$Watch.ElapsedMilliseconds
                    acquired = $true
                }
            })
        }
    } catch {
        Exit-AbsorbPipelineMutexPlan -Leases $Leases.ToArray()
        throw
    }
    return @($Leases.ToArray())
}

function Exit-AbsorbPipelineMutexPlan {
    [CmdletBinding()]
    param([object[]]$Leases = @())

    for ($Index = $Leases.Count - 1; $Index -ge 0; $Index--) {
        $Lease = $Leases[$Index]
        try { $Lease.mutex.ReleaseMutex() } catch { }
        try { $Lease.mutex.Dispose() } catch { }
    }
}

$Definitions = @{
    'TW-PostClose' = @{ Script = 'run_tw_post_close_pipeline.ps1'; Arguments = @('-PublishObservation') }
    'TW-PreMarket' = @{ Script = 'run_tw_pre_market_pipeline.ps1'; Arguments = @() }
    'US-PostClose' = @{ Script = 'run_us_post_close_pipeline.ps1'; Arguments = @('-PublishObservation') }
    'US-PreMarket' = @{ Script = 'run_us_pre_market_pipeline.ps1'; Arguments = @() }
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
$MutexPlan = @(Get-AbsorbPipelineMutexPlan -Job $Job)
$MutexReceipts = @(
    $MutexPlan | ForEach-Object {
        [ordered]@{
            scope = [string]$_.scope
            market = $_.market
            mutex_name = [string]$_.mutex_name
            wait_milliseconds = [int]$_.wait_milliseconds
            waited_milliseconds = 0
            acquired = $false
        }
    }
)
$MutexLeases = @()
$ExitCode = 1

try {
    $MutexLeases = @(Enter-AbsorbPipelineMutexPlan -Plan $MutexPlan)
    $MutexReceipts = @($MutexLeases | ForEach-Object { $_.receipt })
    $Result = Invoke-NativeProcessStreaming `
        -FilePath $PowerShellExe `
        -Arguments $Arguments `
        -LogPath $LogPath `
        -AllowFailure
    $ExitCode = $Result.exit_code
    if ($ExitCode -ne 0) { throw "Pipeline exited with code $ExitCode" }
    @{ job = $Job; started_at = $StartedAt.ToString('o'); finished_at = [DateTimeOffset]::Now.ToString('o'); success = $true; exit_code = 0; log = $LogPath; mutexes = $MutexReceipts } |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $StatusPath -Encoding utf8
    if ($Job -eq 'FullBacktest') {
        $CheckpointPath = Join-Path $DataRoot 'checkpoints\jobs\full_backtest\current.json'
        $Checkpoint = if (Test-Path -LiteralPath $CheckpointPath) { Get-Content -LiteralPath $CheckpointPath -Raw -Encoding utf8 | ConvertFrom-Json } else { $null }
        if ($Checkpoint.status -eq 'completed') {
            try { Disable-ScheduledTask -TaskName 'ABSORB-FullBacktest' -ErrorAction Stop | Out-Null } catch { Write-Warning 'Unable to disable completed full-backtest task' }
        }
    }
} catch {
    @{ job = $Job; started_at = $StartedAt.ToString('o'); finished_at = [DateTimeOffset]::Now.ToString('o'); success = $false; exit_code = if ($ExitCode -is [int]) { $ExitCode } else { 1 }; error = $_.Exception.Message; log = $LogPath; mutexes = $MutexReceipts } |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $StatusPath -Encoding utf8
    throw
} finally {
    Exit-AbsorbPipelineMutexPlan -Leases $MutexLeases
}
