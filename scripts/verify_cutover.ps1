[CmdletBinding()]
param(
    [string]$Project = 'line-stock-bot-498908',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots',
    [string]$Service = 'line-stock-bot',
    [string]$Region = 'asia-east1',
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$ReleaseEvidencePath = 'release-evidence.json',
    [double]$MinimumCoverage = 0.95,
    [double]$MinimumFreeGB = 100,
    [int]$MaximumMarketAgeDays = 7,
    [switch]$ObservationOnly,
    [string]$BaseUrl
)

$ErrorActionPreference = 'Stop'
if ($Project -ne 'line-stock-bot-498908') { throw 'Project is not allowlisted' }
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') { throw 'Bucket is not allowlisted' }
if ($Service -ne 'line-stock-bot') { throw 'Service is not allowlisted' }
if ($Region -ne 'asia-east1') { throw 'Region is not allowlisted' }
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($MaximumMarketAgeDays -lt 0) { throw 'MaximumMarketAgeDays must not be negative' }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Checks = New-Object System.Collections.Generic.List[object]
$CloudRunTrafficEvidence = $null
$CloudRunActiveRevision = $null
$Gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
$RequiredSecrets = @(
    'stock-papi-line-channel-access-token',
    'stock-papi-line-channel-secret',
    'stock-papi-gemini-api-key',
    'stock-papi-finmind-user',
    'stock-papi-finmind-password',
    'stock-papi-alert-task-token'
)
$PredictionPointerUris = @(
    "gs://$Bucket/predictions/v1/latest-TW.json",
    "gs://$Bucket/predictions/v1/latest-US.json"
)

function Add-Check {
    param([string]$Name, [bool]$Ready, [string]$Detail)

    $Checks.Add([ordered]@{
        name = $Name
        status = if ($Ready) { 'READY' } else { 'BLOCKED' }
        detail = $Detail
    }) | Out-Null
}

function Invoke-Gcloud {
    param([string[]]$Arguments)

    if (-not $Gcloud) { throw 'gcloud was not found' }
    $PreviousPythonPath = $env:PYTHONPATH
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        $env:PYTHONPATH = $null
        $Output = & $Gcloud.Source @Arguments 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $env:PYTHONPATH = $PreviousPythonPath
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) { throw 'gcloud command failed' }
    return ($Output | Out-String)
}

function Invoke-Checked {
    param([string]$Name, [scriptblock]$Action)

    try {
        $Detail = & $Action
        Add-Check $Name $true ([string]$Detail)
    } catch {
        Add-Check $Name $false $_.Exception.GetType().Name
    }
}

function Get-JsonFile {
    param([string]$Path)

    $Document = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    if ($null -eq $Document) { throw 'JSON object is empty' }
    return $Document
}

function Test-ReleaseEvidence {
    $EvidencePath = Join-Path $RepoRoot $ReleaseEvidencePath
    $Evidence = Get-JsonFile $EvidencePath
    if ($Evidence.schema_version -ne 1 -or $Evidence.quality_gate -ne 'PASS') {
        throw 'Quality Gate evidence is not PASS'
    }
    if ($null -eq $Evidence.source_hashes -or $Evidence.source_hashes.PSObject.Properties.Count -lt 1) {
        throw 'Quality Gate evidence has no source hashes'
    }
    foreach ($Property in $Evidence.source_hashes.PSObject.Properties) {
        $RelativePath = [string]$Property.Name
        $ExpectedHash = [string]$Property.Value
        if ($RelativePath -notmatch '^[A-Za-z0-9._/-]+$' -or $ExpectedHash -notmatch '^[0-9a-f]{64}$') {
            throw 'Release evidence contains an invalid hash target'
        }
        $Candidate = (Join-Path $RepoRoot $RelativePath).Replace('/', [IO.Path]::DirectorySeparatorChar)
        $Resolved = (Resolve-Path -LiteralPath $Candidate).Path
        if (-not $Resolved.StartsWith($RepoRoot + [IO.Path]::DirectorySeparatorChar)) {
            throw 'Release evidence path escaped repository root'
        }
        if ((Get-FileHash -LiteralPath $Resolved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) {
            throw "Release hash mismatch: $RelativePath"
        }
    }
    return 'Quality Gate PASS evidence and source hashes match'
}

function Get-CloudRunService {
    return Invoke-Gcloud @(
        'run', 'services', 'describe', $Service,
        '--region', $Region,
        '--project', $Project,
        '--format=json'
    ) | ConvertFrom-Json
}

function Get-CloudRunTrafficEvidence {
    param([object]$ServiceInfo)

    $ActiveTraffic = @(
        $ServiceInfo.status.traffic |
            Where-Object {
                [int]$_.percent -eq 100 -and [string]$_.revisionName
            }
    )
    if ($ActiveTraffic.Count -ne 1 -or -not [string]$ServiceInfo.status.url) {
        throw 'Cloud Run traffic is not bound to one 100 percent revision and URL'
    }
    return [ordered]@{
        revision = $ActiveTraffic[0].revisionName
        percent = [int]$ActiveTraffic[0].percent
        url = [string]$ServiceInfo.status.url
    }
}

function Get-CloudRunRevision {
    param([string]$Revision)

    return Invoke-Gcloud @(
        'run', 'revisions', 'describe', $Revision,
        '--region', $Region,
        '--project', $Project,
        '--format=json'
    ) | ConvertFrom-Json
}

function Test-BucketSecurity {
    $BucketInfo = Invoke-Gcloud @('storage', 'buckets', 'describe', "gs://$Bucket", '--format=json') |
        ConvertFrom-Json
    $UniformBucketLevelAccess = if (
        $null -ne $BucketInfo.PSObject.Properties['uniform_bucket_level_access']
    ) {
        $BucketInfo.uniform_bucket_level_access
    } else {
        $BucketInfo.iamConfiguration.uniformBucketLevelAccess.enabled
    }
    $PublicAccessPrevention = if (
        $null -ne $BucketInfo.PSObject.Properties['public_access_prevention']
    ) {
        [string]$BucketInfo.public_access_prevention
    } else {
        [string]$BucketInfo.iamConfiguration.publicAccessPrevention
    }
    $LifecycleRules = @(
        if ($null -ne $BucketInfo.PSObject.Properties['lifecycle_config']) {
            $BucketInfo.lifecycle_config.rule
        } else {
            $BucketInfo.lifecycle.rule
        }
    )
    if ($UniformBucketLevelAccess -ne $true) {
        throw 'Uniform bucket-level access is disabled'
    }
    if ($PublicAccessPrevention -ne 'enforced') {
        throw 'Public access prevention is not enforced'
    }
    if ($LifecycleRules.Count -lt 1) {
        throw 'Lifecycle rule is missing'
    }
    return 'Bucket is private with uniform access, public access prevention and lifecycle'
}

function Test-CloudRunIdentity {
    $ServiceInfo = Get-CloudRunService
    $TrafficEvidence = Get-CloudRunTrafficEvidence $ServiceInfo
    $RevisionInfo = Get-CloudRunRevision ([string]$TrafficEvidence.revision)
    $ReadyCondition = @(
        $RevisionInfo.status.conditions |
            Where-Object { $_.type -eq 'Ready' -and $_.status -eq 'True' }
    )
    if ($ReadyCondition.Count -ne 1) {
        throw 'Cloud Run active traffic revision is not Ready'
    }
    $ServiceAccount = [string]$RevisionInfo.spec.serviceAccountName
    if (-not $ServiceAccount) { throw 'Cloud Run active revision service account is missing' }
    $script:CloudRunActiveRevision = $RevisionInfo
    $script:CloudRunTrafficEvidence = $TrafficEvidence
    return $ServiceAccount
}

function Test-ServiceAccountAccess {
    param([string]$ServiceAccount)

    $Member = "serviceAccount:$ServiceAccount"
    $BucketPolicy = Invoke-Gcloud @('storage', 'buckets', 'get-iam-policy', "gs://$Bucket", '--format=json') |
        ConvertFrom-Json
    $ViewerBinding = @($BucketPolicy.bindings | Where-Object {
        $_.role -eq 'roles/storage.objectViewer' -and $_.members -contains $Member
    })
    $WriterBinding = @($BucketPolicy.bindings | Where-Object {
        $_.role -in @('roles/storage.objectAdmin', 'roles/storage.objectUser', 'roles/storage.admin') -and
        $_.members -contains $Member
    })
    if ($ViewerBinding.Count -ne 1 -or $WriterBinding.Count -ne 0) {
        throw 'Cloud Run storage IAM is not least privilege'
    }

    $ProjectPolicy = Invoke-Gcloud @('projects', 'get-iam-policy', $Project, '--format=json') | ConvertFrom-Json
    $SecretBinding = @($ProjectPolicy.bindings | Where-Object {
        $_.role -eq 'roles/secretmanager.secretAccessor' -and $_.members -contains $Member
    })
    if ($SecretBinding.Count -lt 1) {
        foreach ($Secret in $RequiredSecrets) {
            $SecretPolicy = Invoke-Gcloud @(
                'secrets', 'get-iam-policy', $Secret, '--project', $Project, '--format=json'
            ) | ConvertFrom-Json
            $SecretAccess = @($SecretPolicy.bindings | Where-Object {
                $_.role -eq 'roles/secretmanager.secretAccessor' -and $_.members -contains $Member
            })
            if ($SecretAccess.Count -lt 1) {
                throw 'Cloud Run Secret Manager access is missing'
            }
        }
    }
    return 'Cloud Run has viewer-only GCS and Secret Manager accessor roles'
}

function Test-RequiredSecrets {
    foreach ($Secret in $RequiredSecrets) {
        Invoke-Gcloud @('secrets', 'describe', $Secret, '--project', $Project, '--format=json') | Out-Null
    }
    return 'Required Secret Manager names exist without reading values'
}

function Test-MarketPointer {
    param([string]$Market, [string]$TemporaryRoot)

    $LatestUri = "gs://$Bucket/quant/v1/latest-$Market.json"
    $LatestEvidence = Get-GcsJsonEvidence `
        -Uri $LatestUri `
        -TemporaryRoot $TemporaryRoot `
        -Name "latest-$Market.json"
    $Latest = $LatestEvidence.document
    if (
        -not (Test-ObservationJsonInteger $Latest.schema_version) -or
        $Latest.schema_version -notin @(2, 3, 4) -or
        $Latest.market -ne $Market
    ) {
        throw 'Latest pointer schema or market is invalid'
    }
    if ($Latest.schema_version -eq 3 -and $Market -ne 'TW') {
        throw 'Manifest v3 is TW-only'
    }
    if ([string]$Latest.manifest -notmatch "^manifests/$Market-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$") {
        throw 'Latest manifest path is invalid'
    }
    if ([string]$Latest.manifest_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'Latest manifest hash is invalid'
    }
    if (-not $Latest.manifest.EndsWith(
        "-$($Latest.manifest_sha256.Substring(0, 12)).json"
    )) {
        throw 'Latest manifest path is not hash-bound'
    }

    $ManifestEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/quant/v1/$($Latest.manifest)" `
        -TemporaryRoot $TemporaryRoot `
        -Name "manifest-$Market.json" `
        -ExpectedSha256 ([string]$Latest.manifest_sha256)
    $Manifest = $ManifestEvidence.document
    if (
        $Manifest.schema_version -ne $Latest.schema_version -or
        $Manifest.market -ne $Market -or
        [string]$Manifest.generated_at -ne [string]$Latest.generated_at
    ) {
        throw 'Manifest schema or market is invalid'
    }
    $FailureThreshold = if ($Market -eq 'TW') { 0.05 } else { 0.25 }
    $ExpectedModelVersion = if ($Latest.schema_version -in @(3, 4)) {
        'observation-source-v1'
    } else {
        ''
    }
    $Coverage = Get-ObservationManifestCoverage `
        -Manifest $Manifest `
        -ExpectedMarket $Market `
        -FailureThreshold $FailureThreshold `
        -MaximumMarketAgeDays $MaximumMarketAgeDays `
        -ExpectedModelVersion $ExpectedModelVersion
    return "$Market latest pointer and manifest are verified with coverage $Coverage"
}

function Test-LocalOperations {
    if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
    $Drive = [IO.DriveInfo]::new('D')
    if (-not $Drive.IsReady -or $Drive.AvailableFreeSpace -lt $MinimumFreeGB * 1GB) {
        throw 'D drive free space is below threshold'
    }
    $Acl = Get-Acl -LiteralPath $DataRoot
    if (-not $Acl.AreAccessRulesProtected) { throw 'Data root ACL is not protected' }
    $TaskNames = if ($DataRoot -eq 'D:\AbsorbData') {
        @('ABSORB-LocalQuant', 'ABSORB-QuantUpload')
    } else {
        @('StockPapi-LocalQuant', 'StockPapi-QuantUpload')
    }
    foreach ($TaskName in $TaskNames) {
        $Task = Get-ScheduledTask -TaskName $TaskName
        if ($Task.State -eq 'Disabled') { throw "Scheduled task is disabled: $TaskName" }
    }
    return 'D drive, private ACL and scheduled tasks are ready'
}

$ObservationForbiddenKeys = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
foreach ($Key in @(
    'forecast_probability',
    'probability',
    'ranking_score',
    'model_version',
    'backtest_version',
    'recommendation'
)) {
    $ObservationForbiddenKeys.Add($Key) | Out-Null
}

function Assert-ObservationNoPredictionKeys {
    param(
        [object]$Value,
        [string]$Path = '$'
    )

    if ($null -eq $Value) { return }
    if ($Value -is [Collections.IDictionary]) {
        foreach ($Key in $Value.Keys) {
            if ($ObservationForbiddenKeys.Contains([string]$Key)) {
                throw "Prediction key is forbidden at ${Path}: $Key"
            }
            Assert-ObservationNoPredictionKeys `
                -Value $Value[$Key] `
                -Path "$Path.$Key"
        }
        return
    }
    if ($Value -is [Management.Automation.PSCustomObject]) {
        foreach ($Property in $Value.PSObject.Properties) {
            if ($ObservationForbiddenKeys.Contains([string]$Property.Name)) {
                throw "Prediction key is forbidden at ${Path}: $($Property.Name)"
            }
            Assert-ObservationNoPredictionKeys `
                -Value $Property.Value `
                -Path "$Path.$($Property.Name)"
        }
        return
    }
    if ($Value -is [Collections.IEnumerable] -and $Value -isnot [string]) {
        $Index = 0
        foreach ($Item in $Value) {
            Assert-ObservationNoPredictionKeys `
                -Value $Item `
                -Path "$Path[$Index]"
            $Index += 1
        }
    }
}

function Get-ObservationEnvironment {
    param([object]$ServiceInfo)

    $Environment = @{}
    foreach ($Entry in @($ServiceInfo.spec.containers[0].env)) {
        if ($Entry.name -and $null -ne $Entry.value) {
            $Environment[[string]$Entry.name] = [string]$Entry.value
        }
    }
    return $Environment
}

function Test-ObservationCloudRunEnvironment {
    if ($null -eq $CloudRunActiveRevision) {
        throw 'Cloud Run active traffic revision was not captured'
    }
    $Environment = Get-ObservationEnvironment $CloudRunActiveRevision
    $Expected = [ordered]@{
        ABSORB_PREDICTION_MODE = 'research'
        ABSORB_OBSERVATION_ENABLED = 'true'
        ABSORB_PREDICTION_PROBABILITY_ENABLED = 'false'
        ABSORB_PREDICTION_RANKING_ENABLED = 'false'
        ABSORB_PREDICTION_STRONG_ACTIONS_ENABLED = 'false'
        ABSORB_PREDICTION_PERFORMANCE_ENDORSEMENT_ENABLED = 'false'
    }
    foreach ($Property in $Expected.GetEnumerator()) {
        if (
            -not $Environment.ContainsKey($Property.Key) -or
            $Environment[$Property.Key] -ne $Property.Value
        ) {
            throw "Observation Cloud Run environment mismatch: $($Property.Key)"
        }
    }
    foreach ($Name in @(
        'ABSORB_PREVIEW_CANDIDATE_PREFIX',
        'PREVIEW_CANDIDATE_PREFIX'
    )) {
        if ($Environment.ContainsKey($Name)) {
            throw "Preview prefix remains in Observation Production: $Name"
        }
    }
    return 'Observation mode is research, all Prediction flags are false, and preview prefix is absent'
}

function Get-GcsJsonEvidence {
    param(
        [string]$Uri,
        [string]$TemporaryRoot,
        [string]$Name,
        [string]$ExpectedSha256
    )

    $Metadata = Invoke-Gcloud @(
        'storage', 'objects', 'describe', $Uri, '--format=json'
    ) | ConvertFrom-Json
    if ([string]$Metadata.generation -notmatch '^\d+$') {
        throw "GCS object has no generation: $Uri"
    }
    if (
        [string]$Metadata.size -notmatch '^\d+$' -or
        [long]$Metadata.size -le 0 -or
        [long]$Metadata.size -gt 5MB
    ) {
        throw "GCS JSON object size is outside the allowlist: $Uri"
    }
    $Path = Join-Path $TemporaryRoot $Name
    $GenerationUri = "$Uri#$($Metadata.generation)"
    Invoke-Gcloud @('storage', 'cp', '--quiet', $GenerationUri, $Path) | Out-Null
    $File = Get-Item -LiteralPath $Path
    if ($File.Length -ne [long]$Metadata.size) {
        throw "GCS object size changed during read: $Uri"
    }
    $AfterMetadata = Invoke-Gcloud @(
        'storage', 'objects', 'describe', $Uri, '--format=json'
    ) | ConvertFrom-Json
    if (
        [string]$AfterMetadata.generation -ne [string]$Metadata.generation -or
        [long]$AfterMetadata.size -ne [long]$Metadata.size
    ) {
        throw "GCS object generation changed during read: $Uri"
    }
    $Digest = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ExpectedSha256 -and $Digest -ne $ExpectedSha256) {
        throw "GCS object SHA-256 mismatch: $Uri"
    }
    $Document = Get-JsonFile $Path
    return [pscustomobject]@{
        metadata = $Metadata
        path = $Path
        file = $File
        sha256 = $Digest
        generation = [string]$Metadata.generation
        document = $Document
    }
}

function Assert-ObservationCapability {
    param([object]$Capability)

    if (
        $Capability.mode -ne 'research' -or
        $Capability.observation_enabled -ne $true -or
        $Capability.probability_allowed -ne $false -or
        $Capability.ranking_allowed -ne $false -or
        $Capability.strong_action_allowed -ne $false -or
        $Capability.performance_endorsement_allowed -ne $false
    ) {
        throw 'Observation prediction capability is not fail-closed'
    }
}

function Test-ObservationJsonInteger {
    param([object]$Value)

    return $null -ne $Value -and ($Value -is [int] -or $Value -is [long])
}

function Test-ObservationJsonNumber {
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

function Test-ObservationIsoDate {
    param([object]$Value)

    return $Value -is [string] -and [string]$Value -match '^\d{4}-\d{2}-\d{2}$'
}

function Test-ObservationIsoTimestamp {
    param([object]$Value)

    if ($Value -is [DateTime]) { return $Value.Kind -eq [DateTimeKind]::Utc }
    if ($Value -is [DateTimeOffset]) { return $true }
    return $Value -is [string] -and [string]$Value -match
        '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$'
}

function Test-ObservationMarketSymbol {
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

function Get-ObservationManifestCoverage {
    param(
        [object]$Manifest,
        [string]$ExpectedObservationAsOf,
        [string]$ExpectedMarket = 'TW',
        [double]$FailureThreshold = 0.05,
        [int]$MaximumMarketAgeDays = 0,
        [string]$ExpectedModelVersion = 'observation-source-v1'
    )

    if ($null -eq $Manifest) { throw 'Observation source manifest is empty' }
    $RawSchemaVersion = $Manifest.schema_version
    if (-not (Test-ObservationJsonInteger $RawSchemaVersion)) {
        throw 'Observation source manifest schema is invalid'
    }
    $SchemaVersion = [long]$RawSchemaVersion
    if ($SchemaVersion -notin @(2, 3, 4) -or [string]$Manifest.market -ne $ExpectedMarket) {
        throw 'Observation source manifest schema or market is invalid'
    }
    if ($SchemaVersion -eq 3 -and $ExpectedMarket -ne 'TW') {
        throw 'Manifest v3 is TW-only'
    }
    if (
        $Manifest.PSObject.Properties['sample_data'] -and
        $Manifest.sample_data -ne $false
    ) {
        throw 'Observation source manifest contains sample data'
    }

    $SourceDate = if ($SchemaVersion -in @(3, 4)) {
        if ($Manifest.PSObject.Properties['market_as_of']) {
            throw 'Observation source manifest v3/v4 contains legacy date field'
        }
        [string]$Manifest.target_market_date
    } else {
        [string]$Manifest.market_as_of
    }
    if (
        -not (Test-ObservationIsoDate $SourceDate) -or
        ($ExpectedObservationAsOf -and $SourceDate -ne $ExpectedObservationAsOf)
    ) {
        throw 'Observation source manifest date does not match the dashboard'
    }
    if (-not (Test-ObservationIsoTimestamp $Manifest.generated_at)) {
        throw 'Observation source manifest generated_at is invalid'
    }
    if ($MaximumMarketAgeDays -gt 0) {
        $SourceDateValue = [DateTime]::ParseExact(
            $SourceDate,
            'yyyy-MM-dd',
            [Globalization.CultureInfo]::InvariantCulture
        ).Date
        $Today = [DateTime]::Today
        if (
            $SourceDateValue -gt $Today -or
            ($Today - $SourceDateValue).TotalDays -gt $MaximumMarketAgeDays
        ) {
            throw 'Observation source manifest is stale or future-dated'
        }
    }
    if ($SchemaVersion -in @(3, 4) -and [string]$Manifest.observation_as_of -ne $SourceDate) {
        throw 'Observation source manifest v3/v4 observation date is invalid'
    }

    $RawCoverage = if ($SchemaVersion -in @(3, 4)) {
        $Manifest.observation_coverage
    } else {
        $Manifest.coverage
    }
    if (-not (Test-ObservationJsonNumber $RawCoverage)) {
        throw 'Observation source manifest coverage is invalid'
    }
    $Coverage = [double]$RawCoverage
    if ($Coverage -lt $MinimumCoverage -or $Coverage -gt 1) {
        throw 'Observation source manifest coverage is below the cutover threshold'
    }

    $RawUniverse = if ($SchemaVersion -eq 4) {
        $Manifest.active_universe_count
    } else {
        $Manifest.universe_count
    }
    $RawObserved = if ($SchemaVersion -in @(3, 4)) {
        $Manifest.observation_count
    } else {
        $Manifest.symbol_count
    }
    $RawFailures = if ($SchemaVersion -eq 4) {
        [long]$Manifest.unavailable_count + [long]$Manifest.operational_failure_count
    } elseif ($SchemaVersion -eq 3) {
        $Manifest.operational_failure_count
    } else {
        $Manifest.failure_count
    }
    foreach ($Name in @('universe_count', 'observed_count', 'failure_count')) {
        $Value = switch ($Name) {
            'universe_count' { $RawUniverse }
            'observed_count' { $RawObserved }
            'failure_count' { $RawFailures }
        }
        if (-not (Test-ObservationJsonInteger $Value) -or [long]$Value -lt 0) {
            throw "Observation source manifest count is invalid: $Name"
        }
    }
    $UniverseCount = [long]$RawUniverse
    $ObservedCount = [long]$RawObserved
    $FailureCount = [long]$RawFailures
    if (
        $UniverseCount -le 0 -or
        $ObservedCount + $FailureCount -ne $UniverseCount -or
        [math]::Abs($Coverage - ($ObservedCount / [double]$UniverseCount)) -gt 1e-12
    ) {
        throw 'Observation source manifest counts do not match coverage'
    }

    $SymbolProperties = if ($null -eq $Manifest.symbols) {
        @()
    } else {
        @($Manifest.symbols.PSObject.Properties)
    }
    if ($SymbolProperties.Count -ne $ObservedCount) {
        throw 'Observation source manifest symbol count is invalid'
    }
    $SymbolEntries = @{}
    foreach ($Property in $SymbolProperties) {
        $Symbol = [string]$Property.Name
        $Entry = $Property.Value
        if (
            -not (Test-ObservationMarketSymbol $Symbol $ExpectedMarket) -or
            $null -eq $Entry -or
            $SymbolEntries.ContainsKey($Symbol)
        ) {
            throw 'Observation source manifest symbol entry is invalid'
        }
        $Sha = [string]$Entry.sha256
        $ObjectPath = [string]$Entry.path
        if (
            $Sha -notmatch '^[0-9a-f]{64}$' -or
            $ObjectPath -ne "objects/$Sha.json.gz" -or
            $ObjectPath -notmatch '^objects/[0-9a-f]{64}\.json\.gz$' -or
            -not [string]$Entry.model_version -or
            ($ExpectedModelVersion -and
                [string]$Entry.model_version -ne $ExpectedModelVersion) -or
            -not (Test-ObservationIsoDate $Entry.as_of) -or
            ($SchemaVersion -eq 2 -and [string]$Entry.as_of -ne $SourceDate) -or
            -not (Test-ObservationJsonInteger $Entry.size) -or
            -not (Test-ObservationJsonInteger $Entry.uncompressed_size) -or
            [long]$Entry.size -le 0 -or
            [long]$Entry.size -gt 5MB -or
            [long]$Entry.uncompressed_size -le 0 -or
            [long]$Entry.uncompressed_size -gt 20MB
        ) {
            throw "Observation source manifest symbol entry is invalid: $Symbol"
        }
        $SymbolEntries[$Symbol] = $Entry
    }

    if ($SchemaVersion -eq 2) {
        if (
            $null -eq $Manifest.PSObject.Properties['failed_symbols'] -or
            -not (Test-ObservationJsonNumber $Manifest.failure_rate) -or
            [double]$Manifest.failure_rate -lt 0 -or
            [double]$Manifest.failure_rate -ge $FailureThreshold -or
            [math]::Abs(
                [double]$Manifest.failure_rate -
                ($FailureCount / [double]$UniverseCount)
            ) -gt 1e-12
        ) {
            throw 'Observation source manifest v2 failure rate is invalid'
        }
        $FailedSymbols = if ($null -eq $Manifest.failed_symbols) {
            @()
        } elseif ($Manifest.failed_symbols -is [string]) {
            throw 'Observation source manifest v2 failed symbols are invalid'
        } else {
            @($Manifest.failed_symbols)
        }
        if ($FailedSymbols.Count -ne $FailureCount) {
            throw 'Observation source manifest v2 failed symbol count is invalid'
        }
        $SeenFailed = @{}
        foreach ($Symbol in $FailedSymbols) {
            $Symbol = [string]$Symbol
            if (
                -not (Test-ObservationMarketSymbol $Symbol $ExpectedMarket) -or
                $SeenFailed.ContainsKey($Symbol) -or
                $SymbolEntries.ContainsKey($Symbol)
            ) {
                throw 'Observation source manifest v2 failed symbols are invalid'
            }
            $SeenFailed[$Symbol] = $true
        }
        return $Coverage
    }

    $ExpectedProperties = if ($null -eq $Manifest.expected_non_price_symbols) {
        @()
    } else {
        @($Manifest.expected_non_price_symbols.PSObject.Properties)
    }
    $OperationalFailures = if ($null -eq $Manifest.operational_failed_symbols) {
        @()
    } elseif ($Manifest.operational_failed_symbols -is [string]) {
        throw 'Observation source manifest v3/v4 operational failures are invalid'
    } else {
        @($Manifest.operational_failed_symbols)
    }
    if (
        $null -eq $Manifest.PSObject.Properties['expected_non_price_symbols'] -or
        $null -eq $Manifest.PSObject.Properties['operational_failed_symbols']
    ) {
        throw 'Observation source manifest v3/v4 partition fields are missing'
    }
    $StatusCountName = if ($SchemaVersion -eq 4) {
        'verified_non_price_symbol_count'
    } else {
        'expected_non_price_symbol_count'
    }
    foreach ($Name in @(
        'regular_price_symbol_count',
        $StatusCountName,
        'operational_failure_count',
        'regular_price_denominator'
    )) {
        if (-not (Test-ObservationJsonInteger $Manifest.$Name) -or [long]$Manifest.$Name -lt 0) {
            throw "Observation source manifest v3/v4 count is invalid: $Name"
        }
    }
    if ($SchemaVersion -eq 4) {
        $UnavailableSymbols = if ($null -eq $Manifest.unavailable_symbols) {
            @()
        } elseif ($Manifest.unavailable_symbols -is [string]) {
            throw 'Observation source manifest v4 unavailable symbols are invalid'
        } else {
            @($Manifest.unavailable_symbols)
        }
        if (
            -not (Test-ObservationJsonInteger $Manifest.unavailable_count) -or
            [long]$Manifest.unavailable_count -lt 0 -or
            [long]$Manifest.unavailable_count -ne $UnavailableSymbols.Count -or
            [long]$Manifest.operational_failure_count -ne 0 -or
            $OperationalFailures.Count -ne 0
        ) {
            throw 'Observation source manifest v4 unavailable partition is invalid'
        }
        $SeenUnavailable = @{}
        foreach ($RawSymbol in $UnavailableSymbols) {
            $Symbol = [string]$RawSymbol
            if (
                -not (Test-ObservationMarketSymbol $Symbol $ExpectedMarket) -or
                $SeenUnavailable.ContainsKey($Symbol) -or
                $SymbolEntries.ContainsKey($Symbol)
            ) {
                throw 'Observation source manifest v4 unavailable symbols are invalid'
            }
            $SeenUnavailable[$Symbol] = $true
        }
    }
    $RegularCount = [long]$Manifest.regular_price_symbol_count
    $StatusCount = [long]$Manifest.$StatusCountName
    $RegularDenominator = [long]$Manifest.regular_price_denominator
    if (
        $StatusCount -ne $ExpectedProperties.Count -or
        $FailureCount -ne $OperationalFailures.Count + $(if ($SchemaVersion -eq 4) { [long]$Manifest.unavailable_count } else { 0 }) -or
        $RegularCount + $StatusCount -ne $ObservedCount -or
        $RegularDenominator -le 0 -or
        $RegularDenominator -ne $(if ($SchemaVersion -eq 4) { $ObservedCount } else { $UniverseCount }) - $StatusCount
    ) {
        throw 'Observation source manifest v3/v4 partition counts are invalid'
    }
    if (
        -not (Test-ObservationJsonNumber $Manifest.regular_price_coverage) -or
        [double]$Manifest.regular_price_coverage -lt 0 -or
        [double]$Manifest.regular_price_coverage -gt 1 -or
        [math]::Abs(
            [double]$Manifest.regular_price_coverage -
            ($RegularCount / [double]$RegularDenominator)
        ) -gt 1e-12
    ) {
        throw 'Observation source manifest v3 regular price coverage is invalid'
    }
    if (
        -not (Test-ObservationJsonNumber $Manifest.operational_failure_rate) -or
        [double]$Manifest.operational_failure_rate -lt 0 -or
        [double]$Manifest.operational_failure_rate -ge 0.05 -or
        [math]::Abs(
            [double]$Manifest.operational_failure_rate -
            ($FailureCount / [double]$UniverseCount)
        ) -gt 1e-12
    ) {
        throw 'Observation source manifest v3 failure rate is invalid'
    }

    $ExpectedBySymbol = @{}
    foreach ($Property in $ExpectedProperties) {
        $Symbol = [string]$Property.Name
        $Status = $Property.Value
        if (
            -not (Test-ObservationMarketSymbol $Symbol $ExpectedMarket) -or
            $null -eq $Status -or
            $ExpectedBySymbol.ContainsKey($Symbol) -or
            [string]$Status.status -notin @(
                'official_no_regular_trade',
                'officially_suspended'
            ) -or
            [string]$Status.evidence_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$Status.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or
            -not (Test-ObservationIsoDate $Status.latest_regular_price_date)
        ) {
            throw "Observation source manifest v3 status entry is invalid: $Symbol"
        }
        $ExpectedBySymbol[$Symbol] = $Status
    }
    $SeenOperational = @{}
    foreach ($RawSymbol in $OperationalFailures) {
        $Symbol = [string]$RawSymbol
        if (
            -not (Test-ObservationMarketSymbol $Symbol $ExpectedMarket) -or
            $SeenOperational.ContainsKey($Symbol) -or
            $ExpectedBySymbol.ContainsKey($Symbol) -or
            $SymbolEntries.ContainsKey($Symbol)
        ) {
            throw 'Observation source manifest v3 operational failures are invalid'
        }
        $SeenOperational[$Symbol] = $true
    }

    $RegularSeen = 0
    $StatusSeen = 0
    foreach ($Symbol in $SymbolEntries.Keys) {
        $Entry = $SymbolEntries[$Symbol]
        if (
            [string]$Entry.observation_as_of -ne $SourceDate -or
            -not (Test-ObservationIsoDate $Entry.latest_regular_price_date)
        ) {
            throw "Observation source manifest v3 symbol date is invalid: $Symbol"
        }
        $Kind = [string]$Entry.observation_kind
        if ($Kind -eq 'regular_price') {
            if (
                $ExpectedBySymbol.ContainsKey($Symbol) -or
                [string]$Entry.as_of -ne $SourceDate -or
                [string]$Entry.latest_regular_price_date -ne $SourceDate -or
                $Entry.PSObject.Properties['evidence_sha256']
            ) {
                throw "Observation source manifest v3 regular symbol is invalid: $Symbol"
            }
            $RegularSeen += 1
            continue
        }
        if (
            $Kind -notin @('official_no_regular_trade', 'officially_suspended') -or
            -not $ExpectedBySymbol.ContainsKey($Symbol)
        ) {
            throw "Observation source manifest v3 status symbol is invalid: $Symbol"
        }
        $Expected = $ExpectedBySymbol[$Symbol]
        if (
            [string]$Entry.as_of -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Entry.as_of -ge $SourceDate -or
            [string]$Entry.latest_regular_price_date -ne [string]$Entry.as_of -or
            [string]$Entry.evidence_sha256 -ne [string]$Expected.evidence_sha256 -or
            [string]$Entry.sha256 -ne [string]$Expected.artifact_sha256 -or
            [string]$Entry.latest_regular_price_date -ne
                [string]$Expected.latest_regular_price_date -or
            [string]$Expected.status -ne $Kind
        ) {
            throw "Observation source manifest v3 status binding is invalid: $Symbol"
        }
        $StatusSeen += 1
    }
    if ($RegularSeen -ne $RegularCount -or $StatusSeen -ne $StatusCount) {
        throw 'Observation source manifest v3 symbol partition is invalid'
    }
    return $Coverage
}

function Test-ObservationDashboardPointer {
    param([string]$TemporaryRoot)

    $LatestUri = "gs://$Bucket/dashboard/v1/latest-TW.json"
    $LatestEvidence = Get-GcsJsonEvidence `
        -Uri $LatestUri `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-dashboard-latest.json'
    $Latest = $LatestEvidence.document
    if (
        $Latest.schema_version -ne 2 -or
        $Latest.kind -ne 'absorb-observation-dashboard' -or
        $Latest.product_mode -ne 'observation' -or
        $Latest.market -ne 'TW' -or
        [string]$Latest.path -notmatch '^objects/[0-9a-f]{64}\.json$' -or
        [string]$Latest.sha256 -notmatch '^[0-9a-f]{64}$' -or
        [long]$Latest.size -le 0
    ) {
        throw 'Observation dashboard latest pointer is invalid'
    }

    $ObjectEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/dashboard/v1/$($Latest.path)" `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-dashboard-object.json' `
        -ExpectedSha256 ([string]$Latest.sha256)
    $Digest = (
        Get-FileHash -LiteralPath $ObjectEvidence.path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $ObjectEvidence.file.Length -ne [long]$Latest.size -or
        $Digest -ne [string]$Latest.sha256 -or
        [string]$Latest.path -ne "objects/$Digest.json"
    ) {
        throw 'Observation dashboard immutable object hash or size mismatch'
    }
    $Dashboard = $ObjectEvidence.document
    if (
        $Dashboard.schema_version -ne 2 -or
        $Dashboard.kind -ne 'absorb-observation-dashboard' -or
        $Dashboard.product_mode -ne 'observation' -or
        $Dashboard.market -ne 'TW' -or
        $null -eq $Dashboard.market_index -or
        [string]$Dashboard.observation_as_of -ne [string]$Latest.observation_as_of
    ) {
        throw 'Observation dashboard immutable object schema mismatch'
    }
    Assert-ObservationCapability $Dashboard.prediction_capability
    Assert-ObservationNoPredictionKeys $Dashboard
    foreach ($GateName in @(
        'source_identity',
        'source_schema',
        'finite_json',
        'sample_data',
        'coverage',
        'prediction_separation'
    )) {
        if ([string]$Dashboard.gates.$GateName -ne 'PASS') {
            throw "Observation dashboard gate is not PASS: $GateName"
        }
    }

    $SourceManifest = [string]$Dashboard.source_manifest
    $SourceManifestHash = [string]$Dashboard.source_manifest_sha256
    if (
        $SourceManifest -notmatch
        '^quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
        $SourceManifestHash -notmatch '^[0-9a-f]{64}$'
    ) {
        throw 'Observation dashboard source manifest identity is invalid'
    }
    if (-not $SourceManifest.EndsWith("-$($SourceManifestHash.Substring(0, 12)).json")) {
        throw 'Observation dashboard source manifest path is not hash-bound'
    }
    $SourceEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/$SourceManifest" `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-source-manifest.json' `
        -ExpectedSha256 $SourceManifestHash
    $SourceCoverage = Get-ObservationManifestCoverage `
        -Manifest $SourceEvidence.document `
        -ExpectedObservationAsOf ([string]$Dashboard.observation_as_of) `
        -MaximumMarketAgeDays $MaximumMarketAgeDays
    if (
        $SourceEvidence.sha256 -ne $SourceManifestHash
    ) {
        throw 'Observation dashboard source manifest hash Gate failed'
    }
    return "dashboard/v1/latest-TW.json generation $($LatestEvidence.metadata.generation), coverage $SourceCoverage and immutable object are verified"
}

function Test-ObservationReportPointers {
    param([string]$TemporaryRoot)

    $IndexEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/reports/v2/index-TW.json" `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-reports-index.json'
    $LatestEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/reports/v2/latest-TW-post_close.json" `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-report-latest.json'
    $Index = $IndexEvidence.document
    $Latest = $LatestEvidence.document
    if (
        $Index.schema_version -ne 2 -or
        $Index.kind -ne 'absorb-report-index' -or
        $Index.market -ne 'TW' -or
        $Latest.schema_version -ne 2 -or
        $Latest.kind -ne 'absorb-report' -or
        $Latest.product_mode -ne 'observation' -or
        $Latest.report_type -ne 'post_close' -or
        [string]$Latest.metadata -notmatch '^metadata/[0-9a-f]{64}\.json$' -or
        [string]$Latest.metadata_sha256 -notmatch '^[0-9a-f]{64}$'
    ) {
        throw 'Observation report index or latest pointer is invalid'
    }
    $Matches = @(
        $Index.reports |
            Where-Object {
                $_.report_type -eq 'post_close' -and
                $_.product_mode -eq 'observation' -and
                $_.metadata -eq $Latest.metadata -and
                $_.metadata_sha256 -eq $Latest.metadata_sha256
            }
    )
    if ($Matches.Count -ne 1) {
        throw 'Observation report latest pointer is not bound to exactly one index entry'
    }
    if (@($Matches[0].model_versions.PSObject.Properties).Count -ne 0) {
        throw 'Observation report index contains model versions'
    }

    $MetadataEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/reports/v2/$($Latest.metadata)" `
        -TemporaryRoot $TemporaryRoot `
        -Name 'observation-report-metadata.json'
    $MetadataHash = (
        Get-FileHash -LiteralPath $MetadataEvidence.path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $Metadata = $MetadataEvidence.document
    if (
        $MetadataHash -ne [string]$Latest.metadata_sha256 -or
        $Metadata.schema_version -ne 2 -or
        $Metadata.product_mode -ne 'observation' -or
        $Metadata.report_type -ne 'post_close' -or
        @($Metadata.model_versions.PSObject.Properties).Count -ne 0
    ) {
        throw 'Observation report immutable metadata failed hash or schema Gate'
    }
    Assert-ObservationCapability $Metadata.prediction_capability
    Assert-ObservationNoPredictionKeys $Metadata
    return "reports/v2/index-TW.json generation $($IndexEvidence.metadata.generation) and post-close latest generation $($LatestEvidence.metadata.generation) are verified"
}

function Test-PredictionPointer {
    param([string]$Market, [string]$TemporaryRoot)

    $LatestEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/predictions/v1/latest-$Market.json" `
        -TemporaryRoot $TemporaryRoot `
        -Name "prediction-latest-$Market.json"
    $Latest = $LatestEvidence.document
    $PromotedPointer = (
        [int]$Latest.schema_version -eq 1 -and
        [string]$Latest.backtest_sha256 -match '^[0-9a-f]{64}$'
    )
    $ResearchPointer = (
        [int]$Latest.schema_version -eq 2 -and
        [string]$Latest.validation_mode -eq 'research' -and
        $null -eq $Latest.backtest_sha256
    )
    if (
        -not ($PromotedPointer -or $ResearchPointer) -or
        [string]$Latest.kind -ne 'absorb-five-session-predictions-pointer' -or
        [string]$Latest.market -ne $Market -or
        [string]$Latest.path -notmatch '^objects/[0-9a-f]{64}\.json$' -or
        [string]$Latest.sha256 -notmatch '^[0-9a-f]{64}$' -or
        [long]$Latest.size -le 0 -or [long]$Latest.size -gt 5MB
    ) { throw "Prediction pointer is invalid for $Market" }
    $ObjectEvidence = Get-GcsJsonEvidence `
        -Uri "gs://$Bucket/predictions/v1/$($Latest.path)" `
        -TemporaryRoot $TemporaryRoot `
        -Name "prediction-object-$Market.json" `
        -ExpectedSha256 ([string]$Latest.sha256)
    $Document = $ObjectEvidence.document
    $PromotedDocument = (
        [int]$Document.schema_version -eq 1 -and
        [string]$Document.backtest_sha256 -eq [string]$Latest.backtest_sha256
    )
    $ResearchDocument = (
        [int]$Document.schema_version -eq 2 -and
        [string]$Document.validation_mode -eq 'research' -and
        [string]$Latest.validation_mode -eq 'research' -and
        $null -eq $Document.backtest_sha256 -and
        [int]$Document.prediction_count -eq @($Document.entities.PSObject.Properties).Count -and
        [int]$Document.unavailable_count -eq @($Document.unavailable_symbols).Count -and
        [int]$Document.source_symbol_count -eq ([int]$Document.prediction_count + [int]$Document.unavailable_count)
    )
    if (
        $ObjectEvidence.file.Length -ne [long]$Latest.size -or
        -not ($PromotedDocument -or $ResearchDocument) -or
        [string]$Document.kind -ne 'absorb-five-session-predictions' -or
        [string]$Document.market -ne $Market -or
        [int]$Document.horizon_sessions -ne 5 -or
        [string]$Document.as_of -ne [string]$Latest.as_of -or
        [string]$Document.source_manifest -ne [string]$Latest.source_manifest -or
        [string]$Document.source_manifest_sha256 -ne [string]$Latest.source_manifest_sha256 -or
        @($Document.entities.PSObject.Properties).Count -lt 1
    ) { throw "Prediction object is invalid for $Market" }
    return "predictions/v1/latest-$Market.json generation $($LatestEvidence.metadata.generation) is verified"
}

function Test-ObservationHttp {
    $ServiceInfo = Get-CloudRunService
    $TrafficEvidence = Get-CloudRunTrafficEvidence $ServiceInfo
    $ServiceUrl = [string]$TrafficEvidence.url
    if (
        $CloudRunTrafficEvidence -and
        (
            [string]$CloudRunTrafficEvidence.revision -ne [string]$TrafficEvidence.revision -or
            [int]$CloudRunTrafficEvidence.percent -ne [int]$TrafficEvidence.percent -or
            [string]$CloudRunTrafficEvidence.url -ne $ServiceUrl
        )
    ) {
        throw 'Cloud Run traffic revision or URL changed during verification'
    }
    $script:CloudRunTrafficEvidence = [ordered]@{
        revision = $TrafficEvidence.revision
        percent = $TrafficEvidence.percent
        url = $TrafficEvidence.url
    }
    if ($BaseUrl -and -not $ServiceUrl.TrimEnd('/').Equals($BaseUrl.TrimEnd('/'), [StringComparison]::OrdinalIgnoreCase)) {
        if (-not ($BaseUrl -like "*.run.app*")) {
            throw 'BaseUrl does not match Cloud Run service URL'
        }
    }
    $Target = if ($BaseUrl) { $BaseUrl } else { $ServiceUrl }
    if (-not $Target) { throw 'Observation HTTP base URL is unavailable' }
    foreach ($Path in @(
        '/health',
        '/',
        '/api/dashboard',
        '/reports',
        '/market-map',
        '/stock/2330'
    )) {
        $Response = Invoke-WebRequest `
            -Uri ($Target.TrimEnd('/') + $Path) `
            -UseBasicParsing `
            -MaximumRedirection 5 `
            -TimeoutSec 45
        if ([int]$Response.StatusCode -ne 200) {
            throw "Observation HTTP smoke failed for ${Path}: $($Response.StatusCode)"
        }
        if ($Path -eq '/api/dashboard') {
            try {
                $Document = $Response.Content | ConvertFrom-Json
            } catch {
                throw 'Observation dashboard API did not return valid JSON'
            }
            if (
                $Document.product_mode -ne 'observation' -or
                $null -eq $Document.market_observation -or
                $null -eq $Document.industry_observations -or
                $null -eq $Document.data_quality
            ) {
                throw 'Observation dashboard API schema is invalid'
            }
            Assert-ObservationNoPredictionKeys $Document
        }
    }
    $AfterServiceInfo = Get-CloudRunService
    $AfterTrafficEvidence = Get-CloudRunTrafficEvidence $AfterServiceInfo
    if (
        [string]$AfterTrafficEvidence.revision -ne [string]$TrafficEvidence.revision -or
        [int]$AfterTrafficEvidence.percent -ne [int]$TrafficEvidence.percent -or
        [string]$AfterTrafficEvidence.url -ne [string]$TrafficEvidence.url
    ) {
        throw 'Cloud Run traffic or URL changed during HTTP smoke'
    }
    return "Observation HTTP smoke passed at $Target"
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("absorb-cutover-" + [Guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $TemporaryRoot -Force | Out-Null
    if ($ObservationOnly) {
        Invoke-Checked 'gcloud_available' {
            if (-not $Gcloud) { throw 'gcloud was not found' }
            return 'gcloud is available'
        }
        Invoke-Checked 'gcs_bucket_security' { Test-BucketSecurity }

        $ServiceAccount = $null
        try {
            $ServiceAccount = Test-CloudRunIdentity
            Add-Check 'cloud_run_revision' $true 'Cloud Run has a ready revision and service account'
        } catch {
            Add-Check 'cloud_run_revision' $false $_.Exception.GetType().Name
        }
        if ($ServiceAccount) {
            Invoke-Checked 'cloud_run_iam' { Test-ServiceAccountAccess $ServiceAccount }
        } else {
            Add-Check 'cloud_run_iam' $false 'Service account unavailable'
        }
        Invoke-Checked 'secret_manager_names' { Test-RequiredSecrets }
        Invoke-Checked 'observation_environment' {
            Test-ObservationCloudRunEnvironment
        }
        Invoke-Checked 'observation_dashboard' {
            Test-ObservationDashboardPointer $TemporaryRoot
        }
        Invoke-Checked 'observation_reports' {
            Test-ObservationReportPointers $TemporaryRoot
        }
        Invoke-Checked 'prediction_tw' {
            Test-PredictionPointer 'TW' $TemporaryRoot
        }
        Invoke-Checked 'prediction_us' {
            Test-PredictionPointer 'US' $TemporaryRoot
        }
        Invoke-Checked 'observation_http' { Test-ObservationHttp }
    } else {
        Invoke-Checked 'release_evidence' { Test-ReleaseEvidence }
        Invoke-Checked 'gcloud_available' {
            if (-not $Gcloud) { throw 'gcloud was not found' }
            return 'gcloud is available'
        }
        Invoke-Checked 'gcs_bucket_security' { Test-BucketSecurity }

        $ServiceAccount = $null
        try {
            $ServiceAccount = Test-CloudRunIdentity
            Add-Check 'cloud_run_revision' $true 'Cloud Run has a ready revision and service account'
        } catch {
            Add-Check 'cloud_run_revision' $false $_.Exception.GetType().Name
        }
        if ($ServiceAccount) {
            Invoke-Checked 'cloud_run_iam' { Test-ServiceAccountAccess $ServiceAccount }
        } else {
            Add-Check 'cloud_run_iam' $false 'Service account unavailable'
        }

        Invoke-Checked 'secret_manager_names' { Test-RequiredSecrets }
        Invoke-Checked 'latest_tw' { Test-MarketPointer 'TW' $TemporaryRoot }
        Invoke-Checked 'latest_us' { Test-MarketPointer 'US' $TemporaryRoot }
        Invoke-Checked 'local_operations' { Test-LocalOperations }
    }
} finally {
    if (Test-Path -LiteralPath $TemporaryRoot) {
        $ResolvedTemp = (Resolve-Path -LiteralPath $TemporaryRoot).Path
        $SystemTemp = [IO.Path]::GetTempPath().TrimEnd([IO.Path]::DirectorySeparatorChar)
        if ($ResolvedTemp.StartsWith($SystemTemp + [IO.Path]::DirectorySeparatorChar)) {
            Remove-Item -LiteralPath $ResolvedTemp -Recurse -Force
        }
    }
}

$Ready = @($Checks | Where-Object { $_.status -eq 'BLOCKED' }).Count -eq 0
[ordered]@{
    overall = if ($Ready) { 'READY' } else { 'BLOCKED' }
    mode = if ($ObservationOnly) { 'observation' } else { 'prediction' }
    checked_at = [DateTimeOffset]::UtcNow.ToString('o')
    target = [ordered]@{
        project = $Project
        bucket = $Bucket
        service = $Service
        region = $Region
        data_root = $DataRoot
        cloud_run = $CloudRunTrafficEvidence
    }
    checks = $Checks
} | ConvertTo-Json -Depth 4

if (-not $Ready) { exit 2 }
