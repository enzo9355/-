[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory)][string]$ObservationLkgReceipt,
    [string]$Project = 'line-stock-bot-498908',
    [string]$Service = 'line-stock-bot',
    [string]$Region = 'asia-east1',
    [string]$DataRoot = 'D:\AbsorbData',
    [switch]$ApplyTraffic,
    [bool]$RollbackOnFailure = $true
)

$ErrorActionPreference = 'Stop'
if ($Project -ne 'line-stock-bot-498908') { throw 'Project is not allowlisted' }
if ($Service -ne 'line-stock-bot') { throw 'Service is not allowlisted' }
if ($Region -ne 'asia-east1') { throw 'Region is not allowlisted' }
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}

. (Join-Path $PSScriptRoot 'observation_release_common.ps1')

$RepoRoot = [IO.Path]::GetFullPath([IO.Path]::Combine($PSScriptRoot, '..'))
$LkgRoot = Join-Path $DataRoot 'release\observation-lkg'
$ResolvedLkgReceipt = Assert-PathWithinRoot `
    -Path $ObservationLkgReceipt `
    -Root $LkgRoot
$CaptureRoot = Split-Path -Parent $ResolvedLkgReceipt
$ObservationLkgBytes = [IO.File]::ReadAllBytes($ResolvedLkgReceipt)
$ObservationLkg = [Text.Encoding]::UTF8.GetString(
    $ObservationLkgBytes
) | ConvertFrom-Json
if (
    $ObservationLkg.schema_version -ne 1 -or
    $ObservationLkg.kind -ne 'absorb-observation-lkg' -or
    -not ($ObservationLkg.pointers -is [array])
) {
    throw 'Observation LKG receipt is invalid'
}
$ObservationLkgHasher = [Security.Cryptography.SHA256]::Create()
try {
    $ObservationLkgHash = (
        [BitConverter]::ToString(
            $ObservationLkgHasher.ComputeHash($ObservationLkgBytes)
        ) -replace '-', ''
    ).ToLowerInvariant()
} finally {
    $ObservationLkgHasher.Dispose()
}
$DeploymentReceiptPath = Join-Path $CaptureRoot 'deployment-receipt.json'
if ([IO.File]::Exists($DeploymentReceiptPath)) {
    throw 'Observation deployment receipt already exists'
}

$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$Git = (Get-Command git -ErrorAction Stop).Source
$ForbiddenPredictionKeys = [Collections.Generic.HashSet[string]]::new(
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
    $ForbiddenPredictionKeys.Add($Key) | Out-Null
}

function Invoke-Gcloud {
    param([string[]]$Arguments)

    $PreviousPythonPath = $env:PYTHONPATH
    $PreviousWhatIfPreference = $WhatIfPreference
    $PreviousErrorAction = $ErrorActionPreference
    try {
        # gcloud.cmd invokes a PowerShell wrapper whose environment setup also
        # honors WhatIf. Read-only preflight must still execute so the outer
        # ShouldProcess decision can be made from real service state.
        $WhatIfPreference = $false
        $ErrorActionPreference = 'Continue'
        $env:PYTHONPATH = $null
        $Output = & $Gcloud @Arguments 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $env:PYTHONPATH = $PreviousPythonPath
        $WhatIfPreference = $PreviousWhatIfPreference
        $ErrorActionPreference = $PreviousErrorAction
    }
    if ($ExitCode -ne 0) {
        throw "gcloud command failed with exit code ${ExitCode}: $($Output | Out-String)"
    }
    return ($Output | Out-String)
}

function Get-Service {
    return Invoke-Gcloud @(
        'run', 'services', 'describe', $Service,
        '--project', $Project,
        '--region', $Region,
        '--format=json'
    ) | ConvertFrom-Json
}

function Get-Revision {
    param([string]$Revision)
    return Invoke-Gcloud @(
        'run', 'revisions', 'describe', $Revision,
        '--project', $Project, '--region', $Region, '--format=json'
    ) | ConvertFrom-Json
}

function Get-EnvironmentMap {
    param([object[]]$Environment)

    $Map = @{}
    foreach ($Entry in @($Environment)) {
        if ($Entry.name -and $null -ne $Entry.value) {
            $Map[[string]$Entry.name] = [string]$Entry.value
        }
    }
    return $Map
}

function Assert-ObservationEnvironment {
    param([object]$ServiceInfo)

    $Environment = Get-EnvironmentMap `
        $ServiceInfo.spec.template.spec.containers[0].env
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
            throw "Observation environment mismatch: $($Property.Key)"
        }
    }
    foreach ($PrefixName in @(
        'ABSORB_PREVIEW_CANDIDATE_PREFIX',
        'PREVIEW_CANDIDATE_PREFIX'
    )) {
        if ($Environment.ContainsKey($PrefixName)) {
            throw "Preview prefix remains on Production revision: $PrefixName"
        }
    }
}

function Assert-NoPredictionKeys {
    param(
        [object]$Value,
        [string]$Path = '$'
    )

    if ($null -eq $Value) { return }
    if ($Value -is [Collections.IDictionary]) {
        foreach ($Key in $Value.Keys) {
            if ($ForbiddenPredictionKeys.Contains([string]$Key)) {
                throw "Prediction key is forbidden at ${Path}: $Key"
            }
            Assert-NoPredictionKeys -Value $Value[$Key] -Path "$Path.$Key"
        }
        return
    }
    if ($Value -is [Management.Automation.PSCustomObject]) {
        foreach ($Property in $Value.PSObject.Properties) {
            if ($ForbiddenPredictionKeys.Contains([string]$Property.Name)) {
                throw "Prediction key is forbidden at ${Path}: $($Property.Name)"
            }
            Assert-NoPredictionKeys `
                -Value $Property.Value `
                -Path "$Path.$($Property.Name)"
        }
        return
    }
    if ($Value -is [Collections.IEnumerable] -and $Value -isnot [string]) {
        $Index = 0
        foreach ($Item in $Value) {
            Assert-NoPredictionKeys -Value $Item -Path "$Path[$Index]"
            $Index += 1
        }
    }
}

function Get-TextSha256 {
    param([string]$Text)

    $Bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $Digest = $Hasher.ComputeHash($Bytes)
    } finally {
        $Hasher.Dispose()
    }
    return ([BitConverter]::ToString($Digest) -replace '-', '').ToLowerInvariant()
}

function Invoke-ObservationSmoke {
    param([string]$BaseUrl)

    $Results = New-Object System.Collections.Generic.List[object]
    $ReportsHtml = $null
    $HomeHtml = $null
    foreach ($Path in @(
        '/health',
        '/health/data',
        '/',
        '/market',
        '/industries',
        '/stocks',
        '/ask',
        '/learn',
        '/api/dashboard',
        '/reports',
        '/market-map',
        '/stock/2330'
        '/us',
        '/us/market',
        '/us/industries',
        '/us/stocks',
        '/reports/us',
        '/stock/AAPL'
    )) {
        $Response = Invoke-WebRequest `
            -Uri ($BaseUrl.TrimEnd('/') + $Path) `
            -UseBasicParsing `
            -MaximumRedirection 5 `
            -TimeoutSec 45
        if ([int]$Response.StatusCode -ne 200) {
            throw "Observation smoke failed for ${Path}: $($Response.StatusCode)"
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
            Assert-NoPredictionKeys $Document
        }
        if ($Path -eq '/health/data') {
            try { $Freshness = $Response.Content | ConvertFrom-Json }
            catch { throw 'Data health endpoint did not return valid JSON' }
            if (
                $Freshness.service.status -ne 'ok' -or
                $null -eq $Freshness.markets.TW -or
                $null -eq $Freshness.markets.US
            ) { throw 'Data health endpoint is missing TW or US identity' }
        }
        if ($Path -like '/us*' -or $Path -eq '/reports/us' -or $Path -eq '/stock/AAPL') {
            if ([string]$Response.Content -notmatch 'data-market="US"') {
                throw "US smoke lost market identity: $Path"
            }
        }
        if ($Path -eq '/reports') {
            $ReportsHtml = [string]$Response.Content
        }
        if ($Path -eq '/') {
            $HomeHtml = [string]$Response.Content
        }
        $Results.Add([ordered]@{
            path = $Path
            status = [int]$Response.StatusCode
            body_sha256 = Get-TextSha256 ([string]$Response.Content)
        }) | Out-Null
    }

    $AssetVersion = $null
    foreach ($AssetPrefix in @('/static/app.css?v=', '/static/app.js?v=')) {
        $Pattern = [regex]::Escape($AssetPrefix) + '(?<version>[0-9a-f]{12})'
        $Match = [regex]::Match([string]$HomeHtml, $Pattern)
        if (-not $Match.Success) {
            throw "Revisioned static asset is unavailable: $AssetPrefix"
        }
        $Version = [string]$Match.Groups['version'].Value
        if ($null -ne $AssetVersion -and $AssetVersion -ne $Version) {
            throw 'Static asset versions are inconsistent'
        }
        $AssetVersion = $Version
        $AssetPath = $AssetPrefix + $Version
        $Response = Invoke-WebRequest `
            -Uri ($BaseUrl.TrimEnd('/') + $AssetPath) `
            -UseBasicParsing `
            -MaximumRedirection 0 `
            -TimeoutSec 45
        if ([int]$Response.StatusCode -ne 200) {
            throw "Static asset smoke failed: $AssetPath"
        }
        $Results.Add([ordered]@{
            path = $AssetPath
            status = [int]$Response.StatusCode
            body_sha256 = Get-TextSha256 ([string]$Response.Content)
            asset_version = $Version
        }) | Out-Null
    }

    $CanonicalReportPaths = [ordered]@{
        'post-close' = 'market-actuals-title'
        'pre-market' = 'overnight-title'
    }
    foreach ($ReportType in $CanonicalReportPaths.Keys) {
        $Pattern = 'href="(?<path>/reports/[0-9]{4}-[0-9]{2}-[0-9]{2}/' +
            [regex]::Escape($ReportType) + ')"'
        $Match = [regex]::Match([string]$ReportsHtml, $Pattern)
        if (-not $Match.Success) {
            throw "Observation report link is unavailable: $ReportType"
        }
        $Path = [string]$Match.Groups['path'].Value
        $Response = Invoke-WebRequest `
            -Uri ($BaseUrl.TrimEnd('/') + $Path) `
            -UseBasicParsing `
            -MaximumRedirection 0 `
            -TimeoutSec 45
        $Body = [string]$Response.Content
        $HasExpectedContent = if ($ReportType -eq 'post-close') {
            $Body -like '*market-actuals-title*' -or $Body -like '*professional-report*' -or $Body -like '*report-masthead*'
        } else {
            $Body -like "*$($CanonicalReportPaths[$ReportType])*"
        }
        if (
            [int]$Response.StatusCode -ne 200 -or
            -not $HasExpectedContent -or
            $Body -match 'forecast_probability|ranking_score|model_version|backtest_version'
        ) {
            throw "Observation report smoke failed: $ReportType"
        }
        $Results.Add([ordered]@{
            path = $Path
            status = [int]$Response.StatusCode
            body_sha256 = Get-TextSha256 $Body
        }) | Out-Null
    }
    $UsReports = Invoke-WebRequest -Uri ($BaseUrl.TrimEnd('/') + '/reports/us') `
        -UseBasicParsing -MaximumRedirection 0 -TimeoutSec 45
    $UsMatch = [regex]::Match(
        [string]$UsReports.Content,
        'href="(?<path>/reports/us/[0-9]{4}-[0-9]{2}-[0-9]{2}/post-close)"'
    )
    if (-not $UsMatch.Success) { throw 'US canonical report link is unavailable' }
    $UsPath = [string]$UsMatch.Groups['path'].Value
    $UsCanonical = Invoke-WebRequest -Uri ($BaseUrl.TrimEnd('/') + $UsPath) `
        -UseBasicParsing -MaximumRedirection 0 -TimeoutSec 45
    if (
        [int]$UsCanonical.StatusCode -ne 200 -or
        [string]$UsCanonical.Content -notmatch 'data-market="US"' -or
        [string]$UsCanonical.Content -match 'data-market="TW"'
    ) { throw 'US canonical report market identity smoke failed' }
    $Results.Add([ordered]@{
        path = $UsPath
        status = [int]$UsCanonical.StatusCode
        body_sha256 = Get-TextSha256 ([string]$UsCanonical.Content)
    }) | Out-Null
    return $Results.ToArray()
}

function Invoke-ObservationCutoverVerification {
    param([string]$BaseUrl)

    $WindowsPowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
    $Output = & $WindowsPowerShell `
        -NoProfile `
        -NonInteractive `
        -ExecutionPolicy Bypass `
        -File (Join-Path $PSScriptRoot 'verify_cutover.ps1') `
        -Project $Project `
        -Bucket ([string]$ObservationLkg.bucket) `
        -Service $Service `
        -Region $Region `
        -DataRoot $DataRoot `
        -ObservationOnly `
        -BaseUrl $BaseUrl
    $ExitCode = $LASTEXITCODE
    $Text = ($Output | Out-String).Trim()
    if ($ExitCode -ne 0) {
        throw "Observation cutover verification was BLOCKED: $Text"
    }
    try {
        $Verification = $Text | ConvertFrom-Json
    } catch {
        throw 'Observation cutover verification did not return valid JSON'
    }
    if (
        $Verification.overall -ne 'READY' -or
        $Verification.mode -ne 'observation'
    ) {
        throw 'Observation cutover verification did not return READY'
    }
    return $Verification
}

function Write-DeploymentReceipt {
    param([object]$Receipt)

    $Temporary = "$DeploymentReceiptPath.tmp"
    [IO.File]::WriteAllText(
        $Temporary,
        ($Receipt | ConvertTo-Json -Depth 12),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $Temporary -Destination $DeploymentReceiptPath -Force
}

function Restore-PreviousTraffic {
    param([string]$PreviousTrafficSpec)

    if (-not $PreviousTrafficSpec) {
        throw 'Previous Cloud Run traffic specification is empty'
    }
    Invoke-Gcloud @(
        'run', 'services', 'update-traffic', $Service,
        '--project', $Project,
        '--region', $Region,
        "--to-revisions=$PreviousTrafficSpec",
        '--quiet'
    ) | Out-Null
}

$Dirty = & $Git -C $RepoRoot status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect repository status' }
if (@($Dirty).Count -gt 0) { throw 'Production deployment requires a clean worktree' }
$Commit = (& $Git -C $RepoRoot rev-parse HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $Commit -notmatch '^[0-9a-f]{40}$') {
    throw 'Unable to determine deployment commit'
}
$SourceTree = (& $Git -C $RepoRoot rev-parse 'HEAD^{tree}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceTree -notmatch '^[0-9a-f]{40}$') {
    throw 'Unable to determine deployment source tree'
}

$Before = Get-Service
if (-not $Before.status.latestReadyRevisionName -or -not $Before.status.url) {
    throw 'Cloud Run service has no ready Production revision'
}
$PreviousTraffic = @(
    $Before.status.traffic |
        Where-Object { [int]$_.percent -gt 0 } |
        ForEach-Object {
            [ordered]@{
                revision = [string]$_.revisionName
                percent = [int]$_.percent
                tag = [string]$_.tag
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
    throw 'Cloud Run Production traffic is not a complete 100 percent assignment'
}
$PreviousTrafficSpec = (
    $PreviousTraffic |
        ForEach-Object { "$($_['revision'])=$($_['percent'])" }
) -join ','
$BeforePath = Join-Path $CaptureRoot 'cloud-run-before.json'
$BeforeJson = $Before | ConvertTo-Json -Depth 20
$PreviousEnvironment = @($Before.spec.template.spec.containers[0].env)

$Receipt = [ordered]@{
    schema_version = 1
    kind = 'absorb-observation-deployment'
    created_at = [DateTimeOffset]::UtcNow.ToString('o')
    project = $Project
    service = $Service
    region = $Region
    source_commit = $Commit
    source_tree = $SourceTree
    previous_service = [ordered]@{
        file = 'cloud-run-before.json'
        sha256 = $null
    }
    previous_revision = [string]$Before.status.latestReadyRevisionName
    previous_traffic = $PreviousTraffic
    previous_traffic_spec = $PreviousTrafficSpec
    previous_environment = $PreviousEnvironment
    observation_lkg_receipt = $ResolvedLkgReceipt
    observation_lkg_sha256 = $ObservationLkgHash
    previous_pointer_state = @($ObservationLkg.pointers)
    candidate_revision = $null
    candidate_url = $null
    candidate_tag = $null
    candidate_provenance = $null
    smoke = @()
    traffic_applied = $false
    status = 'PLANNED'
}

if (-not $PSCmdlet.ShouldProcess(
    "$Project/$Region/$Service",
    'deploy fail-closed Observation Production candidate'
)) {
    $Receipt | ConvertTo-Json -Depth 12
    return
}

[IO.File]::WriteAllText(
    $BeforePath,
    $BeforeJson,
    [Text.UTF8Encoding]::new($false)
)
$Receipt.previous_service.sha256 = (
    Get-FileHash -LiteralPath $BeforePath -Algorithm SHA256
).Hash.ToLowerInvariant()
Write-DeploymentReceipt $Receipt

$MutationStarted = $false
$TrafficApplied = $false
try {
    $Tag = "observation-$($Commit.Substring(0, 12))"
    $EnvironmentUpdates = @(
        'ABSORB_PREDICTION_MODE=research',
        'ABSORB_OBSERVATION_ENABLED=true',
        'ABSORB_PREDICTION_PROBABILITY_ENABLED=false',
        'ABSORB_PREDICTION_RANKING_ENABLED=false',
        'ABSORB_PREDICTION_STRONG_ACTIONS_ENABLED=false',
        'ABSORB_PREDICTION_PERFORMANCE_ENDORSEMENT_ENABLED=false'
        "ABSORB_SOURCE_COMMIT=$Commit"
        "ABSORB_SOURCE_TREE=$SourceTree"
    ) -join ','
    $PreviewPrefixes = @(
        'ABSORB_PREVIEW_CANDIDATE_PREFIX',
        'PREVIEW_CANDIDATE_PREFIX'
    ) -join ','

    $MutationStarted = $true
    Invoke-Gcloud @(
        'run', 'deploy', $Service,
        '--source', $RepoRoot,
        '--project', $Project,
        '--region', $Region,
        '--no-traffic',
        '--tag', $Tag,
        '--labels', "absorb-source-commit=$Commit,absorb-source-tree=$SourceTree",
        '--update-env-vars', $EnvironmentUpdates,
        '--remove-env-vars', $PreviewPrefixes,
        '--quiet'
    ) | Out-Null

    $AfterDeploy = Get-Service
    Assert-ObservationEnvironment $AfterDeploy
    $CandidateRevision = [string]$AfterDeploy.status.latestCreatedRevisionName
    if (-not $CandidateRevision) {
        throw 'Observation candidate revision is unavailable'
    }
    $CandidateInfo = Get-Revision -Revision $CandidateRevision
    $CandidateEnvironment = Get-EnvironmentMap `
        $CandidateInfo.spec.containers[0].env
    $CandidateLabels = $CandidateInfo.metadata.labels
    $Image = [string]$CandidateInfo.spec.containers[0].image
    $ImageDigestEvidence = [string]$CandidateInfo.status.imageDigest
    if (-not $ImageDigestEvidence) {
        $ImageDigestEvidence = [string]$CandidateInfo.status.containerStatuses[0].imageDigest
    }
    $ImageDigest = if ($ImageDigestEvidence -match '(?<digest>sha256:[0-9a-f]{64})$') {
        [string]$Matches.digest
    } else { $null }
    if (-not $ImageDigest -and $Image -match '@(?<digest>sha256:[0-9a-f]{64})$') {
        $ImageDigest = [string]$Matches.digest
    }
    if (
        [string]$CandidateLabels.'absorb-source-commit' -ne $Commit -or
        [string]$CandidateLabels.'absorb-source-tree' -ne $SourceTree -or
        $CandidateEnvironment['ABSORB_SOURCE_COMMIT'] -ne $Commit -or
        $CandidateEnvironment['ABSORB_SOURCE_TREE'] -ne $SourceTree -or
        $ImageDigest -notmatch '^sha256:[0-9a-f]{64}$'
    ) { throw 'Candidate revision provenance verification failed' }
    $CurrentCommit = (& $Git -C $RepoRoot rev-parse HEAD | Out-String).Trim()
    $CurrentTree = (& $Git -C $RepoRoot rev-parse 'HEAD^{tree}' | Out-String).Trim()
    $CurrentDirty = & $Git -C $RepoRoot status --porcelain
    if ($CurrentCommit -ne $Commit -or $CurrentTree -ne $SourceTree -or @($CurrentDirty).Count -gt 0) {
        throw 'Deployment source changed while candidate was building'
    }
    $Receipt.candidate_provenance = [ordered]@{
        source_commit = $Commit
        source_tree = $SourceTree
        image = $Image
        image_digest = $ImageDigest
        label_commit = [string]$CandidateLabels.'absorb-source-commit'
        label_tree = [string]$CandidateLabels.'absorb-source-tree'
    }
    $CandidateTraffic = @(
        $AfterDeploy.status.traffic |
            Where-Object { $_.revisionName -eq $CandidateRevision }
    )
    if (@($CandidateTraffic | Where-Object { [int]$_.percent -gt 0 }).Count -gt 0) {
        throw 'Observation candidate unexpectedly received Production traffic'
    }
    $Tagged = @(
        $AfterDeploy.status.traffic |
            Where-Object { $_.tag -eq $Tag }
    ) | Select-Object -First 1
    if (-not $Tagged.url) {
        throw 'Observation candidate tag URL is unavailable'
    }

    $Receipt.candidate_revision = $CandidateRevision
    $Receipt.candidate_url = [string]$Tagged.url
    $Receipt.candidate_tag = $Tag
    $CandidateSmoke = Invoke-ObservationSmoke -BaseUrl $Receipt.candidate_url
    $Receipt.smoke = $CandidateSmoke
    $Receipt.cutover_verification = Invoke-ObservationCutoverVerification `
        -BaseUrl $Receipt.candidate_url
    $Receipt.status = 'NO_TRAFFIC_VERIFIED'
    Write-DeploymentReceipt $Receipt

    if ($ApplyTraffic) {
        if (-not $PSCmdlet.ShouldProcess(
            $CandidateRevision,
            'assign 100 percent Cloud Run Production traffic'
        )) {
            $Receipt | ConvertTo-Json -Depth 12
            return
        }
        Invoke-Gcloud @(
            'run', 'services', 'update-traffic', $Service,
            '--project', $Project,
            '--region', $Region,
            "--to-revisions=$CandidateRevision=100",
            '--quiet'
        ) | Out-Null
        $TrafficApplied = $true

        $AfterTraffic = Get-Service
        $Active = @(
            $AfterTraffic.status.traffic |
                Where-Object {
                    $_.revisionName -eq $CandidateRevision -and
                    [int]$_.percent -eq 100
                }
        )
        if ($Active.Count -ne 1) {
            throw 'Observation candidate did not receive 100 percent traffic'
        }
        Assert-ObservationEnvironment $AfterTraffic
        $ProductionSmoke = Invoke-ObservationSmoke `
            -BaseUrl ([string]$AfterTraffic.status.url)
        $Receipt.production_smoke = $ProductionSmoke
        $Receipt.traffic_applied = $true
        $Receipt.status = 'PRODUCTION_VERIFIED'
        $Receipt.completed_at = [DateTimeOffset]::UtcNow.ToString('o')
        Write-DeploymentReceipt $Receipt
    }

    [ordered]@{
        deployment_receipt = $DeploymentReceiptPath
        candidate_revision = $Receipt.candidate_revision
        candidate_url = $Receipt.candidate_url
        traffic_applied = $Receipt.traffic_applied
        status = $Receipt.status
    } | ConvertTo-Json -Compress
} catch {
    $Failure = $_
    $RollbackErrors = New-Object System.Collections.Generic.List[string]
    if ($RollbackOnFailure -and $MutationStarted -and $TrafficApplied) {
        try {
            Restore-PreviousTraffic $PreviousTrafficSpec
        } catch {
            $RollbackErrors.Add("traffic: $($_.Exception.Message)") | Out-Null
        }
    }
    $Receipt.status = 'FAILED'
    $Receipt.failure = $Failure.Exception.Message
    $Receipt.rollback_errors = $RollbackErrors.ToArray()
    $Receipt.failed_at = [DateTimeOffset]::UtcNow.ToString('o')
    Write-DeploymentReceipt $Receipt
    if ($RollbackErrors.Count -gt 0) {
        throw "$($Failure.Exception.Message); rollback errors: $($RollbackErrors -join '; ')"
    }
    throw
}
