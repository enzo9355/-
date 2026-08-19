[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Quant')]
    [ValidateSet('TW', 'US')]
    [string]$Market,

    [Parameter(Mandatory, ParameterSetName = 'Quant')]
    [ValidatePattern('^manifests/(TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$')]
    [string]$LkgManifest,

    [Parameter(Mandatory, ParameterSetName = 'Observation')]
    [string]$ObservationDeploymentReceipt,

    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots',
    [string]$Project = 'line-stock-bot-498908',
    [string]$Service = 'line-stock-bot',
    [string]$Region = 'asia-east1',
    [string]$DataRoot = 'D:\AbsorbData',
    [int]$MaximumSeconds = 10
)

$ErrorActionPreference = 'Stop'
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') { throw 'Bucket is not allowlisted' }
if ($Project -ne 'line-stock-bot-498908') { throw 'Project is not allowlisted' }
if ($Service -ne 'line-stock-bot') { throw 'Service is not allowlisted' }
if ($Region -ne 'asia-east1') { throw 'Region is not allowlisted' }
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($MaximumSeconds -lt 1) { throw 'MaximumSeconds must be positive' }

if ($PSCmdlet.ParameterSetName -eq 'Observation') {
    . (Join-Path $PSScriptRoot 'observation_release_common.ps1')

    function Invoke-ObservationGcloud {
        param([string[]]$Arguments)

        $GcloudPath = (Get-Command gcloud -ErrorAction Stop).Source
        $PreviousPythonPath = $env:PYTHONPATH
        $PreviousWhatIfPreference = $WhatIfPreference
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            $WhatIfPreference = $false
            $ErrorActionPreference = 'SilentlyContinue'
            $env:PYTHONPATH = $null
            $Output = & $GcloudPath @Arguments 2>&1
            $ExitCode = $LASTEXITCODE
        } finally {
            $env:PYTHONPATH = $PreviousPythonPath
            $WhatIfPreference = $PreviousWhatIfPreference
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($ExitCode -ne 0) {
            throw "gcloud command failed with exit code ${ExitCode}: $($Output | Out-String)"
        }
        return ($Output | Out-String)
    }

    $ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
    $ResolvedDeploymentReceipt = Assert-PathWithinRoot `
        -Path $ObservationDeploymentReceipt `
        -Root $ReceiptRoot
    $Deployment = Get-Content `
        -LiteralPath $ResolvedDeploymentReceipt `
        -Raw `
        -Encoding utf8 | ConvertFrom-Json
    if (
        $Deployment.schema_version -ne 1 -or
        $Deployment.kind -ne 'absorb-observation-deployment' -or
        $Deployment.project -ne $Project -or
        $Deployment.service -ne $Service -or
        $Deployment.region -ne $Region -or
        $Deployment.traffic_applied -ne $true -or
        [string]$Deployment.candidate_revision -notmatch
            '^line-stock-bot-[0-9]{5}-[a-z0-9]+$' -or
        -not ($Deployment.previous_traffic -is [array])
    ) {
        throw 'Observation deployment rollback receipt is invalid'
    }

    $CaptureRoot = Split-Path -Parent $ResolvedDeploymentReceipt
    $PreviousServicePath = Assert-PathWithinRoot `
        -Path (Join-Path $CaptureRoot ([string]$Deployment.previous_service.file)) `
        -Root $CaptureRoot
    $PreviousServiceHash = (
        Get-FileHash -LiteralPath $PreviousServicePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($PreviousServiceHash -ne [string]$Deployment.previous_service.sha256) {
        throw 'Observation previous service snapshot hash mismatch'
    }

    $LkgReceipt = Assert-PathWithinRoot `
        -Path ([string]$Deployment.observation_lkg_receipt) `
        -Root $ReceiptRoot
    $LkgHash = (
        Get-FileHash -LiteralPath $LkgReceipt -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($LkgHash -ne [string]$Deployment.observation_lkg_sha256) {
        throw 'Observation LKG receipt hash mismatch'
    }

    $PreviousTraffic = @(
        $Deployment.previous_traffic |
            ForEach-Object {
                [ordered]@{
                    revision = [string]$_.revision
                    percent = [int]$_.percent
                }
            }
    )
    $PreviousTrafficPercent = (
        $PreviousTraffic |
            ForEach-Object { [int]$_['percent'] } |
            Measure-Object -Sum
    ).Sum
    if (
        $PreviousTraffic.Count -lt 1 -or
        $PreviousTrafficPercent -ne 100
    ) {
        throw 'Observation previous_traffic is incomplete'
    }
    $PreviousTrafficSpec = (
        $PreviousTraffic |
            ForEach-Object { "$($_['revision'])=$($_['percent'])" }
    ) -join ','
    if ($PreviousTrafficSpec -ne [string]$Deployment.previous_traffic_spec) {
        throw 'Observation previous_traffic specification mismatch'
    }

    function Restore-ObservationCandidateTraffic {
        param([string]$CandidateRevision)

        $CandidateTrafficSpec = "$CandidateRevision=100"
        Invoke-ObservationGcloud @(
            'run', 'services', 'update-traffic', $Service,
            '--project', $Project,
            '--region', $Region,
            "--to-revisions=$CandidateTrafficSpec",
            '--quiet'
        ) | Out-Null
        $After = Invoke-ObservationGcloud @(
            'run', 'services', 'describe', $Service,
            '--project', $Project,
            '--region', $Region,
            '--format=json'
        ) | ConvertFrom-Json
        $CandidateActive = @(
            $After.status.traffic |
                Where-Object {
                    $_.revisionName -eq $CandidateRevision -and
                    [int]$_.percent -eq 100
                }
        )
        if ($CandidateActive.Count -ne 1) {
            throw 'Observation candidate traffic compensation failed'
        }
    }

    function Write-ObservationRecoveryReceipt {
        param(
            [string]$PointerErrorType,
            [bool]$TrafficCompensated,
            [string]$TrafficCompensationErrorType
        )

        $RecoveryPath = Join-Path `
            $ReceiptRoot `
            ('manual-rollback-recovery-' + [Guid]::NewGuid().ToString('N') + '.json')
        $TemporaryRecoveryPath = "$RecoveryPath.tmp"
        $Recovery = [ordered]@{
            schema_version = 1
            kind = 'absorb-observation-manual-rollback-recovery'
            created_at = [DateTimeOffset]::UtcNow.ToString('o')
            project = $Project
            service = $Service
            region = $Region
            bucket = $Bucket
            deployment_receipt = $ResolvedDeploymentReceipt
            observation_lkg_receipt = $LkgReceipt
            candidate_revision = [string]$Deployment.candidate_revision
            previous_traffic = $PreviousTraffic
            pointer_rollback_attempted = $true
            pointer_error_type = $PointerErrorType
            traffic_compensation = if ($TrafficCompensated) { 'restored_candidate' } else { 'blocked' }
            traffic_compensation_error_type = $TrafficCompensationErrorType
            next_action = 'Reconcile Cloud Run traffic and Observation pointers from this receipt before retrying rollback'
        }
        try {
            [IO.File]::WriteAllText(
                $TemporaryRecoveryPath,
                ($Recovery | ConvertTo-Json -Depth 8),
                [Text.UTF8Encoding]::new($false)
            )
            Move-Item -LiteralPath $TemporaryRecoveryPath -Destination $RecoveryPath -Force
        } catch {
            if (Test-Path -LiteralPath $TemporaryRecoveryPath) {
                Remove-Item -LiteralPath $TemporaryRecoveryPath -Force
            }
            throw
        }
        return $RecoveryPath
    }

    $Current = Invoke-ObservationGcloud @(
        'run', 'services', 'describe', $Service,
        '--project', $Project,
        '--region', $Region,
        '--format=json'
    ) | ConvertFrom-Json
    $CandidateActive = @(
        $Current.status.traffic |
            Where-Object {
                $_.revisionName -eq [string]$Deployment.candidate_revision -and
                [int]$_.percent -eq 100
            }
    )
    if ($CandidateActive.Count -ne 1) {
        throw 'Observation candidate is not the sole active Production revision'
    }

    if (-not $PSCmdlet.ShouldProcess(
        "$Project/$Region/$Service",
        'restore previous traffic and Observation pointers'
    )) {
        return
    }

    $StartedAt = [DateTimeOffset]::UtcNow
    $TrafficChanged = $false
    $PointerRestored = $false
    try {
        $TrafficChanged = $true
        Invoke-ObservationGcloud @(
            'run', 'services', 'update-traffic', $Service,
            '--project', $Project,
            '--region', $Region,
            "--to-revisions=$PreviousTrafficSpec",
            '--quiet'
        ) | Out-Null
        $AfterTraffic = Invoke-ObservationGcloud @(
            'run', 'services', 'describe', $Service,
            '--project', $Project,
            '--region', $Region,
            '--format=json'
        ) | ConvertFrom-Json
        foreach ($Expected in $PreviousTraffic) {
            $Match = @(
                $AfterTraffic.status.traffic |
                    Where-Object {
                        $_.revisionName -eq [string]$Expected['revision'] -and
                        [int]$_.percent -eq [int]$Expected['percent']
                    }
            )
            if ($Match.Count -ne 1) {
                throw "Observation Cloud Run traffic rollback verification failed: $($Expected['revision'])"
            }
        }

        $PointerResult = & (Join-Path $PSScriptRoot 'rollback_observation.ps1') `
            -ReceiptPath $LkgReceipt `
            -DataRoot $DataRoot `
            -Bucket $Bucket `
            -Confirm:$false
        $PointerRestored = $true
    } catch {
        if ($TrafficChanged -and -not $PointerRestored) {
            $PointerErrorType = $_.Exception.GetType().Name
            $TrafficCompensated = $false
            $TrafficCompensationErrorType = $null
            try {
                Restore-ObservationCandidateTraffic `
                    -CandidateRevision ([string]$Deployment.candidate_revision)
                $TrafficCompensated = $true
            } catch {
                $TrafficCompensationErrorType = $_.Exception.GetType().Name
            }
            $RecoveryPath = Write-ObservationRecoveryReceipt `
                -PointerErrorType $PointerErrorType `
                -TrafficCompensated $TrafficCompensated `
                -TrafficCompensationErrorType $TrafficCompensationErrorType
            if ($TrafficCompensated) {
                throw "Observation pointer rollback failed; candidate traffic was restored. Recovery receipt: $RecoveryPath"
            }
            throw "Observation rollback is blocked; traffic compensation failed. Recovery receipt: $RecoveryPath"
        }
        throw
    }
    $ElapsedSeconds = (
        [DateTimeOffset]::UtcNow - $StartedAt
    ).TotalSeconds
    [ordered]@{
        event = 'OBSERVATION_MANUAL_ROLLBACK'
        deployment_receipt = $ResolvedDeploymentReceipt
        restored_traffic = $PreviousTraffic
        observation_lkg_receipt = $LkgReceipt
        pointer_result = $PointerResult
        elapsed_seconds = [Math]::Round($ElapsedSeconds, 3)
    } | ConvertTo-Json -Depth 8 -Compress
    return
}

if ($LkgManifest -notmatch "^manifests/$Market-") { throw 'LKG manifest market does not match Market' }

$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$LatestUri = "gs://$Bucket/quant/v1/latest-$Market.json"
$LkgUri = "gs://$Bucket/quant/v1/$LkgManifest"
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ("stock-papi-rollback-" + [Guid]::NewGuid().ToString('N'))
$StartedAt = [DateTimeOffset]::UtcNow

function Invoke-Gcloud {
    param([string[]]$Arguments)

    $PreviousPythonPath = $env:PYTHONPATH
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        $env:PYTHONPATH = $null
        $Output = & $Gcloud @Arguments 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $env:PYTHONPATH = $PreviousPythonPath
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "gcloud command failed with exit code ${ExitCode}: $($Output | Out-String)"
    }
    return ($Output | Out-String)
}

function Get-JsonFile {
    param([string]$Path)

    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    } catch {
        throw "Invalid JSON object: $Path"
    }
}

function Download-Object {
    param([string]$Source, [string]$Destination)

    Invoke-Gcloud @('storage', 'cp', '--quiet', $Source, $Destination) | Out-Null
}

function Test-JsonInteger {
    param([object]$Value)

    return $null -ne $Value -and ($Value -is [int] -or $Value -is [long])
}

function Test-JsonNumber {
    param([object]$Value)

    if (
        $null -eq $Value -or
        $Value -is [bool] -or
        (
            $Value -isnot [int] -and
            $Value -isnot [long] -and
            $Value -isnot [float] -and
            $Value -isnot [double] -and
            $Value -isnot [decimal]
        )
    ) {
        return $false
    }
    $Number = [double]$Value
    return -not [double]::IsNaN($Number) -and -not [double]::IsInfinity($Number)
}

function Test-IsoDate {
    param([object]$Value)

    return $Value -is [string] -and [string]$Value -match '^\d{4}-\d{2}-\d{2}$'
}

function Test-IsoTimestamp {
    param([object]$Value)

    if ($Value -is [DateTime]) { return $Value.Kind -eq [DateTimeKind]::Utc }
    if ($Value -is [DateTimeOffset]) { return $true }
    return $Value -is [string] -and [string]$Value -match
        '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$'
}

function Test-MarketSymbol {
    param(
        [object]$Value,
        [string]$Market
    )

    if ($Value -isnot [string]) { return $false }
    $Symbol = [string]$Value
    if ($Market -eq 'TW') {
        return $Symbol -match '^\d{4,6}$'
    }
    return $Symbol.Length -le 10 -and
        $Symbol -cmatch '^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?$'
}

function Assert-QuantManifest {
    param(
        [object]$Manifest,
        [string]$Market
    )

    if (
        $null -eq $Manifest -or
        -not (Test-JsonInteger $Manifest.schema_version) -or
        $Manifest.schema_version -notin @(2, 3, 4) -or
        $Manifest.market -ne $Market
    ) {
        throw 'LKG manifest schema or market is invalid'
    }
    if ($Manifest.schema_version -in @(3, 4) -and $Market -ne 'TW') {
        throw 'Manifest v3/v4 is TW-only'
    }
    if (
        -not (Test-IsoTimestamp $Manifest.generated_at) -or
        $Manifest.PSObject.Properties['sample_data'] -and
        $Manifest.sample_data -ne $false
    ) {
        throw 'LKG manifest lacks point-in-time metadata'
    }
    if (
        $null -eq $Manifest.PSObject.Properties['symbols'] -or
        $Manifest.schema_version -eq 2 -and
        -not (Test-IsoDate $Manifest.market_as_of)
    ) {
        throw 'LKG manifest point-in-time or symbols metadata is invalid'
    }

    $SymbolProperties = if ($null -eq $Manifest.symbols) {
        @()
    } else {
        @($Manifest.symbols.PSObject.Properties)
    }
    if ($Manifest.schema_version -eq 2) {
        $FailureThreshold = if ($Market -eq 'TW') { 0.05 } else { 0.25 }
        if (
            -not (Test-JsonInteger $Manifest.universe_count) -or
            -not (Test-JsonInteger $Manifest.symbol_count) -or
            -not (Test-JsonInteger $Manifest.failure_count) -or
            [long]$Manifest.universe_count -lt 1 -or
            [long]$Manifest.symbol_count -lt 1 -or
            [long]$Manifest.symbol_count -ne $SymbolProperties.Count -or
            [long]$Manifest.failure_count -ne
                ([long]$Manifest.universe_count - [long]$Manifest.symbol_count) -or
            [long]$Manifest.failure_count -lt 0 -or
            -not (Test-JsonNumber $Manifest.coverage) -or
            [double]$Manifest.coverage -lt 0.95 -or
            [double]$Manifest.coverage -gt 1 -or
            [math]::Abs(
                [double]$Manifest.coverage -
                ([long]$Manifest.symbol_count / [double]$Manifest.universe_count)
            ) -gt 1e-12 -or
            -not (Test-JsonNumber $Manifest.failure_rate) -or
            [double]$Manifest.failure_rate -lt 0 -or
            [double]$Manifest.failure_rate -ge $FailureThreshold -or
            [math]::Abs(
                [double]$Manifest.failure_rate -
                ([long]$Manifest.failure_count / [double]$Manifest.universe_count)
            ) -gt 1e-12
        ) {
            throw 'LKG manifest v2 arithmetic or coverage is invalid'
        }
        if ($null -eq $Manifest.PSObject.Properties['failed_symbols']) {
            throw 'LKG manifest failed symbols are missing'
        }
        $FailedSymbols = if ($null -eq $Manifest.failed_symbols) {
            @()
        } elseif ($Manifest.failed_symbols -is [string]) {
            throw 'LKG manifest failed symbols are invalid'
        } else {
            @($Manifest.failed_symbols)
        }
        if ($FailedSymbols.Count -ne [long]$Manifest.failure_count) {
            throw 'LKG manifest failed symbol count is invalid'
        }
        $SeenFailed = @{}
        foreach ($RawSymbol in $FailedSymbols) {
            $Symbol = [string]$RawSymbol
            if (
                -not (Test-MarketSymbol $Symbol $Market) -or
                $SeenFailed.ContainsKey($Symbol) -or
                @($SymbolProperties | Where-Object { $_.Name -eq $Symbol }).Count -ne 0
            ) {
                throw 'LKG manifest failed symbols are invalid'
            }
            $SeenFailed[$Symbol] = $true
        }
        $SeenSymbols = @{}
        foreach ($Property in $SymbolProperties) {
            $Symbol = [string]$Property.Name
            $Entry = $Property.Value
            $Sha = [string]$Entry.sha256
            if (
                -not (Test-MarketSymbol $Symbol $Market) -or
                $null -eq $Entry -or
                $Sha -notmatch '^[0-9a-f]{64}$' -or
                [string]$Entry.path -ne "objects/$Sha.json.gz" -or
                -not (Test-JsonInteger $Entry.size) -or
                -not (Test-JsonInteger $Entry.uncompressed_size) -or
                [long]$Entry.size -le 0 -or
                [long]$Entry.uncompressed_size -le 0 -or
                [long]$Entry.size -gt 5MB -or
                [long]$Entry.uncompressed_size -gt 20MB -or
                -not (Test-IsoDate $Entry.as_of) -or
                [string]$Entry.as_of -ne [string]$Manifest.market_as_of -or
                -not [string]$Entry.model_version
            ) {
                throw "LKG manifest v2 symbol entry is invalid: $Symbol"
            }
            if ($SeenSymbols.ContainsKey($Symbol)) {
                throw "LKG manifest v2 symbol is duplicated: $Symbol"
            }
            $SeenSymbols[$Symbol] = $true
        }
        return
    }

    if (
        $null -eq $Manifest.PSObject.Properties['expected_non_price_symbols'] -or
        $null -eq $Manifest.PSObject.Properties['operational_failed_symbols'] -or
        $Manifest.PSObject.Properties['market_as_of'] -or
        -not (Test-IsoDate $Manifest.target_market_date) -or
        [string]$Manifest.observation_as_of -ne [string]$Manifest.target_market_date
    ) {
        throw 'LKG manifest v3/v4 date metadata is invalid'
    }
    $Schema = [int]$Manifest.schema_version
    $UniverseName = if ($Schema -eq 4) { 'active_universe_count' } else { 'universe_count' }
    $StatusCountName = if ($Schema -eq 4) {
        'verified_non_price_symbol_count'
    } else {
        'expected_non_price_symbol_count'
    }
    $CountNames = @(
        $UniverseName,
        'observation_count',
        'regular_price_symbol_count',
        $StatusCountName,
        'operational_failure_count',
        'regular_price_denominator'
    )
    foreach ($Name in $CountNames) {
        if (-not (Test-JsonInteger $Manifest.$Name) -or [long]$Manifest.$Name -lt 0) {
            throw "LKG manifest v3/v4 count is invalid: $Name"
        }
    }
    if ($Schema -eq 4) {
        $UnavailableSymbols = if ($null -eq $Manifest.unavailable_symbols) {
            @()
        } elseif ($Manifest.unavailable_symbols -is [string]) {
            throw 'LKG manifest v4 unavailable symbols are invalid'
        } else {
            @($Manifest.unavailable_symbols)
        }
        if (
            -not (Test-JsonInteger $Manifest.unavailable_count) -or
            [long]$Manifest.unavailable_count -ne $UnavailableSymbols.Count -or
            [long]$Manifest.operational_failure_count -ne 0
        ) {
            throw 'LKG manifest v4 unavailable partition is invalid'
        }
    }
    $ExpectedProperties = if ($null -eq $Manifest.expected_non_price_symbols) {
        @()
    } else {
        @($Manifest.expected_non_price_symbols.PSObject.Properties)
    }
    $OperationalFailures = if ($null -eq $Manifest.operational_failed_symbols) {
        @()
    } elseif ($Manifest.operational_failed_symbols -is [string]) {
        throw 'LKG manifest v3/v4 operational failures are invalid'
    } else {
        @($Manifest.operational_failed_symbols)
    }
    $UniverseCount = [long]$Manifest.$UniverseName
    $StatusCount = [long]$Manifest.$StatusCountName
    $ExpectedDenominator = if ($Schema -eq 4) {
        [long]$Manifest.observation_count
    } else {
        $UniverseCount
    }
    if (
        $UniverseCount -lt 1 -or
        [long]$Manifest.regular_price_denominator -lt 1 -or
        [long]$Manifest.observation_count -ne $SymbolProperties.Count -or
        $StatusCount -ne $ExpectedProperties.Count -or
        [long]$Manifest.operational_failure_count -ne $OperationalFailures.Count -or
        [long]$Manifest.regular_price_symbol_count +
            $StatusCount -ne
            [long]$Manifest.observation_count -or
        [long]$Manifest.observation_count +
            [long]$Manifest.operational_failure_count +
            $(if ($Schema -eq 4) { [long]$Manifest.unavailable_count } else { 0 }) -ne
            $UniverseCount -or
        [long]$Manifest.regular_price_denominator -ne
            ($ExpectedDenominator - $StatusCount) -or
        -not (Test-JsonNumber $Manifest.regular_price_coverage) -or
        [double]$Manifest.regular_price_coverage -lt 0 -or
        [double]$Manifest.regular_price_coverage -gt 1 -or
        [math]::Abs(
            [double]$Manifest.regular_price_coverage -
            ([long]$Manifest.regular_price_symbol_count /
            [double]$Manifest.regular_price_denominator)
        ) -gt 1e-12 -or
        -not (Test-JsonNumber $Manifest.observation_coverage) -or
        [double]$Manifest.observation_coverage -lt 0.95 -or
        [double]$Manifest.observation_coverage -gt 1 -or
        [math]::Abs(
            [double]$Manifest.observation_coverage -
            ([long]$Manifest.observation_count / [double]$UniverseCount)
        ) -gt 1e-12 -or
        -not (Test-JsonNumber $Manifest.operational_failure_rate) -or
        [double]$Manifest.operational_failure_rate -lt 0 -or
        [double]$Manifest.operational_failure_rate -ge 0.05 -or
        [math]::Abs(
            [double]$Manifest.operational_failure_rate -
            ([long]$Manifest.operational_failure_count / [double]$UniverseCount)
        ) -gt 1e-12
    ) {
        throw 'LKG manifest v3/v4 arithmetic or coverage is invalid'
    }

    $ExpectedBySymbol = @{}
    foreach ($Property in $ExpectedProperties) {
        $Symbol = [string]$Property.Name
        $Status = $Property.Value
        if (
            -not (Test-MarketSymbol $Symbol $Market) -or
            $null -eq $Status -or
            $ExpectedBySymbol.ContainsKey($Symbol) -or
            [string]$Status.status -notin @(
                'official_no_regular_trade',
                'officially_suspended'
            ) -or
            [string]$Status.evidence_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$Status.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or
            -not (Test-IsoDate $Status.latest_regular_price_date)
        ) {
            throw "LKG manifest v3 status entry is invalid: $Symbol"
        }
        $ExpectedBySymbol[$Symbol] = $Status
    }
    $SeenOperational = @{}
    foreach ($RawSymbol in $OperationalFailures) {
        $Symbol = [string]$RawSymbol
        if (
            -not (Test-MarketSymbol $Symbol $Market) -or
            $SeenOperational.ContainsKey($Symbol) -or
            $ExpectedBySymbol.ContainsKey($Symbol) -or
            @($SymbolProperties | Where-Object { $_.Name -eq $Symbol }).Count -ne 0
        ) {
            throw 'LKG manifest v3/v4 operational failures are invalid'
        }
        $SeenOperational[$Symbol] = $true
    }
    if ($Schema -eq 4) {
        $SeenUnavailable = @{}
        foreach ($RawSymbol in $UnavailableSymbols) {
            $Symbol = [string]$RawSymbol
            if (
                -not (Test-MarketSymbol $Symbol $Market) -or
                $SeenUnavailable.ContainsKey($Symbol) -or
                $ExpectedBySymbol.ContainsKey($Symbol) -or
                @($SymbolProperties | Where-Object { $_.Name -eq $Symbol }).Count -ne 0
            ) {
                throw 'LKG manifest v4 unavailable symbols are invalid'
            }
            $SeenUnavailable[$Symbol] = $true
        }
    }
    $RegularSeen = 0
    $StatusSeen = 0
    foreach ($Property in $SymbolProperties) {
        $Symbol = [string]$Property.Name
        $Entry = $Property.Value
        $Sha = [string]$Entry.sha256
        if (
            -not (Test-MarketSymbol $Symbol $Market) -or
            $null -eq $Entry -or
            $Sha -notmatch '^[0-9a-f]{64}$' -or
            [string]$Entry.path -ne "objects/$Sha.json.gz" -or
            [string]$Entry.model_version -ne 'observation-source-v1' -or
            -not (Test-JsonInteger $Entry.size) -or
            -not (Test-JsonInteger $Entry.uncompressed_size) -or
            [long]$Entry.size -le 0 -or
            [long]$Entry.uncompressed_size -le 0 -or
            [long]$Entry.size -gt 5MB -or
            [long]$Entry.uncompressed_size -gt 20MB -or
            [string]$Entry.observation_as_of -ne [string]$Manifest.target_market_date -or
            -not (Test-IsoDate $Entry.latest_regular_price_date)
        ) {
            throw "LKG manifest v3 symbol entry is invalid: $Symbol"
        }
        $Kind = [string]$Entry.observation_kind
        if ($Kind -eq 'regular_price') {
            if (
                $ExpectedBySymbol.ContainsKey($Symbol) -or
                [string]$Entry.as_of -ne [string]$Manifest.target_market_date -or
                [string]$Entry.latest_regular_price_date -ne
                    [string]$Manifest.target_market_date
            ) {
                throw "LKG manifest v3 regular symbol is invalid: $Symbol"
            }
            $RegularSeen += 1
            continue
        }
        if (
            $Kind -notin @('official_no_regular_trade', 'officially_suspended') -or
            -not $ExpectedBySymbol.ContainsKey($Symbol)
        ) {
            throw "LKG manifest v3 status symbol is invalid: $Symbol"
        }
        $Expected = $ExpectedBySymbol[$Symbol]
        if (
            [string]$Entry.as_of -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Entry.as_of -ge [string]$Manifest.target_market_date -or
            [string]$Entry.latest_regular_price_date -ne [string]$Entry.as_of -or
            [string]$Entry.evidence_sha256 -ne [string]$Expected.evidence_sha256 -or
            [string]$Entry.sha256 -ne [string]$Expected.artifact_sha256 -or
            [string]$Entry.latest_regular_price_date -ne
                [string]$Expected.latest_regular_price_date -or
            [string]$Expected.status -ne $Kind
        ) {
            throw "LKG manifest v3 status binding is invalid: $Symbol"
        }
        $StatusSeen += 1
    }
    if (
        $RegularSeen -ne [long]$Manifest.regular_price_symbol_count -or
        $StatusSeen -ne $StatusCount
    ) {
        throw 'LKG manifest v3/v4 symbol partition is invalid'
    }
}

try {
    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
    $CurrentMetadata = Invoke-Gcloud @('storage', 'objects', 'describe', $LatestUri, '--format=json') |
        ConvertFrom-Json
    if ([string]$CurrentMetadata.generation -notmatch '^\d+$') {
        throw 'Active latest pointer has no valid GCS generation'
    }

    $CurrentPath = Join-Path $TempRoot 'current-latest.json'
    $ManifestPath = Join-Path $TempRoot 'lkg-manifest.json'
    Download-Object $LatestUri $CurrentPath
    Download-Object $LkgUri $ManifestPath

    $Current = Get-JsonFile $CurrentPath
    $Manifest = Get-JsonFile $ManifestPath
    if (
        -not (Test-JsonInteger $Current.schema_version) -or
        $Current.schema_version -notin @(2, 3, 4) -or
        $Current.market -ne $Market -or
        -not (Test-IsoTimestamp $Current.generated_at)
    ) {
        throw 'Active latest pointer schema or market is invalid'
    }
    if ($Current.schema_version -in @(3, 4) -and $Market -ne 'TW') {
        throw 'Manifest v3/v4 is TW-only'
    }
    if (
        [string]$Current.manifest -notmatch
            "^manifests/$Market-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$" -or
        [string]$Current.manifest_sha256 -notmatch '^[0-9a-f]{64}$' -or
        -not $Current.manifest.EndsWith(
            "-$($Current.manifest_sha256.Substring(0, 12)).json"
        )
    ) {
        throw 'Active latest pointer manifest identity is invalid'
    }
    if ($Current.manifest -eq $LkgManifest) {
        throw 'Requested LKG manifest is already active'
    }
    Assert-QuantManifest -Manifest $Manifest -Market $Market

    $ManifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $LkgManifest.EndsWith("-$($ManifestHash.Substring(0, 12)).json")) {
        throw 'LKG manifest path is not bound to its SHA-256'
    }
    $RollbackPointer = [ordered]@{
        schema_version = [int]$Manifest.schema_version
        market = $Market
        generated_at = [string]$Manifest.generated_at
        manifest = $LkgManifest
        manifest_sha256 = $ManifestHash
    }
    $PointerPath = Join-Path $TempRoot 'rollback-latest.json'
    [IO.File]::WriteAllText(
        $PointerPath,
        ($RollbackPointer | ConvertTo-Json -Compress),
        [Text.UTF8Encoding]::new($false)
    )

    if (-not $PSCmdlet.ShouldProcess($LatestUri, "replace active pointer with $LkgManifest")) {
        return
    }

    Invoke-Gcloud @(
        'storage', 'cp', '--quiet',
        "--if-generation-match=$($CurrentMetadata.generation)",
        $PointerPath,
        $LatestUri
    ) | Out-Null

    $VerifiedPath = Join-Path $TempRoot 'verified-latest.json'
    Download-Object $LatestUri $VerifiedPath
    $Verified = Get-JsonFile $VerifiedPath
    if (
        $Verified.schema_version -ne [int]$Manifest.schema_version -or
        $Verified.manifest -ne $LkgManifest -or
        $Verified.manifest_sha256 -ne $ManifestHash -or
        $Verified.market -ne $Market
    ) {
        throw 'Rollback pointer verification failed'
    }

    $ElapsedSeconds = ([DateTimeOffset]::UtcNow - $StartedAt).TotalSeconds
    if ($ElapsedSeconds -gt $MaximumSeconds) {
        throw "Rollback completed but exceeded ${MaximumSeconds}s target: $ElapsedSeconds"
    }
    [ordered]@{
        event = 'MANUAL_ROLLBACK'
        market = $Market
        lkg_manifest = $LkgManifest
        manifest_sha256 = $ManifestHash
        elapsed_seconds = [Math]::Round($ElapsedSeconds, 3)
    } | ConvertTo-Json -Compress
} finally {
    if (Test-Path -LiteralPath $TempRoot) {
        $ResolvedTemp = (Resolve-Path -LiteralPath $TempRoot).Path
        $SystemTemp = [IO.Path]::GetTempPath().TrimEnd([IO.Path]::DirectorySeparatorChar)
        if ($ResolvedTemp.StartsWith($SystemTemp + [IO.Path]::DirectorySeparatorChar)) {
            Remove-Item -LiteralPath $ResolvedTemp -Recurse -Force
        }
    }
}
