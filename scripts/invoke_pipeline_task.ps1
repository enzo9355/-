[CmdletBinding()]
param(
    [ValidateSet('TW-PostClose', 'TW-PreMarket', 'TW-ObservationRecovery', 'US-PostClose', 'US-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
    [string]$Job,
    [string]$DataRoot = 'D:\AbsorbData'
)

$ErrorActionPreference = 'Stop'

function Test-AbsorbVerifiedFullBacktestCompletion {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$DataRoot,
        [Parameter(Mandatory)][string]$RepoRoot
    )
    . (Join-Path $RepoRoot 'scripts\python_runtime.ps1')
    $PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
    Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot -RequiredImports @('stock_papi')
    & $PythonExe -m stock_papi.batch.full_backtest_cli --root $DataRoot --verify-completion | Out-Null
    return $LASTEXITCODE -eq 0
}

function Get-AbsorbPipelineMutexPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('TW-PostClose', 'TW-PreMarket', 'TW-ObservationRecovery', 'US-PostClose', 'US-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
        [string]$Job
    )

    # Market writer locks cover computation. Publication serialization is
    # acquired by upload_local_quant.ps1 only around the shared transaction.
    #
    # TW-ObservationRecovery deliberately acquires NO mutex here: its child
    # (run_tw_observation_recovery.ps1 -> catch_up_latest_completed_session.ps1)
    # acquires Global\ABSORB-TW-Observation-Writer itself. Wrapping the same
    # named mutex around the child would self-lock, because a child process
    # can never acquire a mutex the wrapper already holds.
    $Plan = New-Object 'System.Collections.Generic.List[object]'
    if ($Job -eq 'TW-ObservationRecovery') {
        return @()
    }
    $Market = switch ($Job) {
        { $_ -in @('TW-PostClose', 'TW-PreMarket') } { 'TW'; break }
        { $_ -in @('US-PostClose', 'US-PreMarket', 'US-Daily') } { 'US'; break }
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
    return @($Plan.ToArray())
}

function Enter-AbsorbPipelineMutexPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Plan,
        [System.Collections.Generic.List[object]]$Receipts = (New-Object 'System.Collections.Generic.List[object]')
    )

    $Leases = New-Object 'System.Collections.Generic.List[object]'
    try {
        foreach ($Entry in $Plan) {
            $WaitMilliseconds = [int]$Entry.wait_milliseconds
            if ($WaitMilliseconds -lt 1 -or $WaitMilliseconds -gt 600000) {
                throw 'Pipeline mutex wait is outside the safe range'
            }
            $Receipt = [ordered]@{
                scope = [string]$Entry.scope
                market = $Entry.market
                mutex_name = [string]$Entry.mutex_name
                wait_milliseconds = $WaitMilliseconds
                waited_milliseconds = 0
                acquired = $false
                ever_acquired = $false
                released = $false
                acquisition_reason = $null
                failure_reason = $null
            }
            [void]$Receipts.Add($Receipt)
            $Watch = [Diagnostics.Stopwatch]::StartNew()
            $Mutex = $null
            try {
                try {
                    $Mutex = [Threading.Mutex]::new($false, [string]$Entry.mutex_name)
                } catch {
                    $Receipt.failure_reason = 'mutex_create_failed'
                    throw 'Pipeline mutex could not be created'
                }
                try {
                    $Acquired = $Mutex.WaitOne($WaitMilliseconds)
                    if ($Acquired) {
                        $Receipt.acquisition_reason = 'acquired'
                    }
                } catch [Threading.AbandonedMutexException] {
                    # WaitOne transfers ownership to the current thread before
                    # reporting abandonment. Treat it as an acquired lease.
                    $Acquired = $true
                    $Receipt.acquisition_reason = 'abandoned_mutex_acquired'
                } catch {
                    $Receipt.failure_reason = 'wait_failed'
                    try { $Mutex.Dispose() } catch { }
                    throw 'Pipeline mutex wait failed'
                }
            } finally {
                $Watch.Stop()
                $Receipt.waited_milliseconds = [int]$Watch.ElapsedMilliseconds
            }
            if (-not $Acquired) {
                $Receipt.failure_reason = 'timeout'
                try { $Mutex.Dispose() } catch { }
                throw [TimeoutException]::new("Timed out waiting for pipeline $($Entry.scope) mutex")
            }
            $Receipt.acquired = $true
            $Receipt.ever_acquired = $true
            [void]$Leases.Add([pscustomobject]@{
                mutex = $Mutex
                receipt = $Receipt
                released = $false
                disposed = $false
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
        if (-not $Lease.released) {
            try {
                $Lease.mutex.ReleaseMutex()
                $Lease.released = $true
                $Lease.receipt.acquired = $false
                $Lease.receipt.released = $true
            } catch {
                $Lease.receipt.failure_reason = 'release_failed'
            }
        }
        if (-not $Lease.disposed) {
            try { $Lease.mutex.Dispose() } catch { }
            $Lease.disposed = $true
        }
    }
}

function Disable-AbsorbCompletedFullBacktestTask {
    [CmdletBinding()]
    param()

    try {
        $ScheduledTask = Get-ScheduledTask -TaskName 'ABSORB-FullBacktest' -ErrorAction Stop
    } catch {
        throw 'Unable to query the completed FullBacktest scheduled task'
    }
    if ([string]$ScheduledTask.State -eq 'Disabled') {
        return [pscustomobject]@{ disabled = $true; already_disabled = $true }
    }
    try {
        Disable-ScheduledTask -TaskName 'ABSORB-FullBacktest' -ErrorAction Stop | Out-Null
    } catch {
        throw 'Unable to disable the completed FullBacktest scheduled task'
    }
    try {
        [xml]$RegistrationXml = Export-ScheduledTask -TaskName 'ABSORB-FullBacktest' -ErrorAction Stop
        $EnabledNode = $RegistrationXml.SelectSingleNode(
            "/*[local-name()='Task']/*[local-name()='Settings']/*[local-name()='Enabled']"
        )
    } catch {
        throw 'Unable to verify the completed FullBacktest scheduled task registration'
    }
    if ($null -eq $EnabledNode) {
        throw 'Unable to verify the completed FullBacktest scheduled task registration'
    }
    $RegistrationEnabled = ([string]$EnabledNode.InnerText).Trim().ToLowerInvariant()
    if ($RegistrationEnabled -eq 'true') {
        throw 'Completed FullBacktest scheduled task registration remains enabled'
    }
    if ($RegistrationEnabled -ne 'false') {
        throw 'Unable to verify the completed FullBacktest scheduled task registration'
    }
    return [pscustomobject]@{
        disabled = $true
        already_disabled = $false
        registration_enabled = $false
        operational_state = [string]$ScheduledTask.State
    }
}

function Invoke-AbsorbPipelineTask {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('TW-PostClose', 'TW-PreMarket', 'TW-ObservationRecovery', 'US-PostClose', 'US-PreMarket', 'FullBacktest', 'US-Daily', 'WeeklyModel', 'ReportUploadRecovery')]
        [string]$Job,
        [Parameter(Mandatory)][string]$DataRoot,
        [Parameter(Mandatory)][string]$ScriptPath,
        [string[]]$ScriptArguments = @(),
        [Parameter(Mandatory)][string]$LogDirectory,
        [Parameter(Mandatory)][string]$PowerShellExe
    )

    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $StartedAt = [DateTimeOffset]::Now
    $LogPath = Join-Path $LogDirectory ("{0}-{1:yyyyMMdd}.log" -f $Job, $StartedAt)
    $StatusPath = Join-Path $LogDirectory ("current-{0}.json" -f $Job)
    $MutexPlan = @(Get-AbsorbPipelineMutexPlan -Job $Job)
    $MutexReceipts = New-Object 'System.Collections.Generic.List[object]'
    $MutexLeases = @()
    $TaskExitCode = 1
    $TaskSucceeded = $false
    $SafeError = $null
    $SafeFailureStage = $null
    $FullBacktestTask = $null
    $FullBacktestCompletionVerified = $false

    try {
        $MutexLeases = @(Enter-AbsorbPipelineMutexPlan -Plan $MutexPlan -Receipts $MutexReceipts)
        $Arguments = @(
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            $ScriptPath,
            '-DataRoot',
            $DataRoot
        ) + $ScriptArguments
        $Result = Invoke-NativeProcessStreaming `
            -FilePath $PowerShellExe `
            -Arguments $Arguments `
            -LogPath $LogPath `
            -AllowFailure
        $TaskExitCode = [int]$Result.exit_code
        if ($TaskExitCode -ne 0) {
            $SafeError = "Pipeline exited with code $TaskExitCode"
        } else {
            if ($Job -eq 'FullBacktest') {
                $FullBacktestCompletionVerified = Test-AbsorbVerifiedFullBacktestCompletion `
                    -DataRoot $DataRoot `
                    -RepoRoot $RepoRoot
                if ($FullBacktestCompletionVerified) {
                    # Disable and verify before a success receipt is committed.
                    $SafeFailureStage = 'full_backtest_disable'
                    $FullBacktestTask = Disable-AbsorbCompletedFullBacktestTask
                    $SafeFailureStage = $null
                }
            }
            $TaskSucceeded = $true
        }
    } catch [TimeoutException] {
        $TaskExitCode = 1
        $SafeError = 'Pipeline mutex acquisition timed out'
    } catch {
        $TaskExitCode = 1
        $SafeError = if ($SafeFailureStage -eq 'full_backtest_disable') {
            # The helper converts every cmdlet/XML failure into a fixed safe message.
            $_.Exception.Message
        } else {
            'Pipeline task failed before successful completion'
        }
    } finally {
        Exit-AbsorbPipelineMutexPlan -Leases $MutexLeases
    }
    if (@($MutexReceipts | Where-Object { $_.failure_reason -eq 'release_failed' }).Count -gt 0) {
        $TaskExitCode = 1
        $TaskSucceeded = $false
        $SafeError = 'Pipeline mutex release failed'
    }

    $Status = [ordered]@{
        job = $Job
        started_at = $StartedAt.ToString('o')
        finished_at = [DateTimeOffset]::Now.ToString('o')
        success = $TaskSucceeded
        exit_code = $TaskExitCode
        full_backtest_completion_verified = $FullBacktestCompletionVerified
        log = $LogPath
        mutexes = @($MutexReceipts.ToArray())
    }
    if ($null -ne $SafeError) {
        $Status.error = $SafeError
    }
    if ($null -ne $FullBacktestTask) {
        $Status.full_backtest_task = $FullBacktestTask
    }
    $Status | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $StatusPath -Encoding utf8
    return [int]$TaskExitCode
}

$Definitions = @{
    'TW-PostClose' = @{ Script = 'run_tw_post_close_pipeline.ps1'; Arguments = @('-PublishObservation') }
    'TW-PreMarket' = @{ Script = 'run_tw_pre_market_pipeline.ps1'; Arguments = @() }
    'TW-ObservationRecovery' = @{ Script = 'run_tw_observation_recovery.ps1'; Arguments = @() }
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
$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) { throw 'PowerShell executable was not found' }
if ($DataRoot -ne 'D:\AbsorbData') { throw 'Data root is not allowlisted' }
. (Join-Path $PSScriptRoot 'native_process.ps1')
$TaskExitCode = Invoke-AbsorbPipelineTask `
    -Job $Job `
    -DataRoot $DataRoot `
    -ScriptPath $ScriptPath `
    -ScriptArguments $Definition.Arguments `
    -LogDirectory $LogDirectory `
    -PowerShellExe $PowerShellExe
exit $TaskExitCode
