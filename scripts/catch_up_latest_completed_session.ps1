[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$TargetDate,
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$Project = 'line-stock-bot-498908',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($Project -ne 'line-stock-bot-498908') {
    throw 'Project is not allowlisted'
}
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') {
    throw 'Bucket is not allowlisted'
}

$Invariant = [Globalization.CultureInfo]::InvariantCulture
try {
    $ParsedTargetDate = [DateTime]::ParseExact(
        $TargetDate,
        'yyyy-MM-dd',
        $Invariant
    ).Date
}
catch {
    throw 'TargetDate must be YYYY-MM-DD'
}
if ($ParsedTargetDate.ToString('yyyy-MM-dd', $Invariant) -ne $TargetDate) {
    throw 'TargetDate must be canonical YYYY-MM-DD'
}
try {
    $LocalToday = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
        [DateTimeOffset]::UtcNow,
        'Taipei Standard Time'
    ).Date
}
catch {
    throw 'Unable to determine Taipei local today'
}
if ($ParsedTargetDate -ge $LocalToday) {
    throw 'TargetDate must be strictly before local today'
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'python_runtime.ps1')
. (Join-Path $PSScriptRoot 'observation_release_common.ps1')
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot
$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$OldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
$CatchUpLockStream = $null
$ObservationWriterMutexName = 'Global\ABSORB-TW-Observation-Writer'
$ObservationWriterMutex = $null
$ObservationWriterMutexHeld = $false

$ObservationPointerNames = @(
    'quant-latest',
    'reports-v2-index',
    'reports-v2-post-close',
    'dashboard-latest'
)
$AllPointerDefinitions = @(
    [pscustomobject]@{
        name = 'quant-latest'
        uri = "gs://$Bucket/quant/v1/latest-TW.json"
    },
    [pscustomobject]@{
        name = 'quant-insights-latest'
        uri = "gs://$Bucket/quant/v1/latest-insights.json"
    },
    [pscustomobject]@{
        name = 'dashboard-latest'
        uri = "gs://$Bucket/dashboard/v1/latest-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-v1-index'
        uri = "gs://$Bucket/reports/v1/index-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-v1-latest'
        uri = "gs://$Bucket/reports/v1/latest-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-v2-index'
        uri = "gs://$Bucket/reports/v2/index-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-v2-post-close'
        uri = "gs://$Bucket/reports/v2/latest-TW-post_close.json"
    },
    [pscustomobject]@{
        name = 'reports-v2-pre-market'
        uri = "gs://$Bucket/reports/v2/latest-TW-pre_market.json"
    },
    [pscustomobject]@{
        name = 'reports-v2-weekly-model'
        uri = "gs://$Bucket/reports/v2/latest-TW-weekly_model.json"
    }
)
$ObservationDefinitions = @(
    [pscustomobject]@{
        name = 'quant'
        kind = 'quant'
        uri = "gs://$Bucket/quant/v1/latest-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-index'
        kind = 'reports-index'
        uri = "gs://$Bucket/reports/v2/index-TW.json"
    },
    [pscustomobject]@{
        name = 'reports-latest'
        kind = 'reports-latest'
        uri = "gs://$Bucket/reports/v2/latest-TW-post_close.json"
    },
    [pscustomobject]@{
        name = 'dashboard'
        kind = 'dashboard'
        uri = "gs://$Bucket/dashboard/v1/latest-TW.json"
    }
)

function ConvertTo-CanonicalDate {
    param(
        [Parameter(Mandatory)]$Value,
        [Parameter(Mandatory)][string]$Label
    )

    $Text = [string]$Value
    try {
        $Parsed = [DateTime]::ParseExact($Text, 'yyyy-MM-dd', $Invariant).Date
    }
    catch {
        throw "$Label must be canonical YYYY-MM-DD"
    }
    if ($Parsed.ToString('yyyy-MM-dd', $Invariant) -ne $Text) {
        throw "$Label must be canonical YYYY-MM-DD"
    }
    return $Parsed
}

function Read-JsonWithinRoot {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root,
        [int]$MaximumBytes = 5MB
    )

    $Resolved = Assert-PathWithinRoot -Path $Path -Root $Root
    $Item = Get-Item -LiteralPath $Resolved -Force
    if (
        $Item.PSIsContainer -or
        $Item.Length -le 0 -or
        $Item.Length -gt $MaximumBytes
    ) {
        throw 'JSON file size is invalid'
    }
    try {
        $Document = Get-Content -LiteralPath $Resolved -Raw -Encoding utf8 |
            ConvertFrom-Json
    }
    catch {
        throw 'JSON file is invalid'
    }
    if ($null -eq $Document) {
        throw 'JSON document is empty'
    }
    return [pscustomobject]@{
        path = $Resolved
        document = $Document
        size = [long]$Item.Length
        sha256 = (Get-FileHash -LiteralPath $Resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)]$Document
    )

    $Candidate = [IO.Path]::GetFullPath($Path)
    $Parent = [IO.Path]::GetDirectoryName($Candidate)
    if ([string]::IsNullOrWhiteSpace($Parent)) {
        throw 'Evidence path has no parent directory'
    }
    $ResolvedParent = Assert-PathWithinRoot -Path $Parent -Root $Root
    $Leaf = [IO.Path]::GetFileName($Candidate)
    if ([string]::IsNullOrWhiteSpace($Leaf)) {
        throw 'Evidence path has no file name'
    }
    $Resolved = [IO.Path]::Combine($ResolvedParent, $Leaf)
    if ([IO.File]::Exists($Resolved) -or [IO.Directory]::Exists($Resolved)) {
        $Resolved = Assert-PathWithinRoot -Path $Resolved -Root $Root
    }
    $Temporary = "$Resolved.tmp"
    if ([IO.File]::Exists($Temporary)) {
        throw 'Catch-up evidence temporary file already exists'
    }
    try {
        [IO.File]::WriteAllText(
            $Temporary,
            ($Document | ConvertTo-Json -Depth 20),
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $Temporary -Destination $Resolved -Force
    }
    finally {
        if ([IO.File]::Exists($Temporary)) {
            [IO.File]::Delete($Temporary)
        }
    }
}

function Acquire-CatchUpLock {
    try {
        $script:ObservationWriterMutex = [Threading.Mutex]::new(
            $false,
            $ObservationWriterMutexName
        )
        try {
            if (-not $script:ObservationWriterMutex.WaitOne(0)) {
                throw 'Another TW observation writer is active'
            }
        }
        catch [Threading.AbandonedMutexException] {
            throw 'TW observation writer mutex was abandoned; refusing to run'
        }
        $script:ObservationWriterMutexHeld = $true
    }
    catch {
        if ($null -ne $script:ObservationWriterMutex -and
            -not $script:ObservationWriterMutexHeld) {
            $script:ObservationWriterMutex.Dispose()
            $script:ObservationWriterMutex = $null
        }
        throw
    }
    $ReleaseRoot = Join-Path $DataRoot 'release'
    if (-not [IO.Directory]::Exists($ReleaseRoot)) {
        New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
    }
    $ReleaseRoot = Assert-PathWithinRoot -Path $ReleaseRoot -Root $DataRoot
    $LockPath = Join-Path $ReleaseRoot 'tw-catch-up.lock'
    if ([IO.File]::Exists($LockPath)) {
        $null = Assert-PathWithinRoot -Path $LockPath -Root $ReleaseRoot
    }
    try {
        $script:CatchUpLockStream = [IO.File]::Open(
            $LockPath,
            [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    }
    catch {
        throw 'Another TW catch-up or local release operation is active'
    }
    if (
        ([IO.File]::GetAttributes($LockPath) -band
            [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        $script:CatchUpLockStream.Dispose()
        $script:CatchUpLockStream = $null
        throw 'Catch-up lock must not be a reparse point'
    }
}

function Assert-NoPendingPointerJournals {
    $ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
    if (-not [IO.Directory]::Exists($ReceiptRoot)) { return }
    $ReceiptRoot = Assert-PathWithinRoot -Path $ReceiptRoot -Root $DataRoot
    $Directories = New-Object System.Collections.Generic.List[string]
    $Directories.Add($ReceiptRoot) | Out-Null
    foreach ($Child in @(Get-ChildItem -LiteralPath $ReceiptRoot -Directory -Force)) {
        $ChildPath = Assert-PathWithinRoot -Path $Child.FullName -Root $ReceiptRoot
        $Directories.Add($ChildPath) | Out-Null
    }
    foreach ($Directory in $Directories) {
        $Pending = @(
            Get-ChildItem -LiteralPath $Directory -File -Filter '*.pending.json' -Force
        )
        $PendingTemporary = @(
            Get-ChildItem -LiteralPath $Directory -File -Filter '*.pending.json.tmp' -Force
        )
        if ($Pending.Count -gt 0 -or $PendingTemporary.Count -gt 0) {
            throw 'Observation LKG pending pointer journal requires reconciliation'
        }
    }
}

function Assert-PostCloseTaskIdle {
    $Task = Get-ScheduledTask -TaskName 'ABSORB-TW-PostClose' -ErrorAction Stop
    if ([string]$Task.State -ne 'Ready') {
        throw "ABSORB-TW-PostClose is not Ready: $($Task.State)"
    }
}

function Invoke-PythonJson {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $Output = (& $PythonExe @Arguments | Out-String).Trim()
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Python contract command failed with exit code $ExitCode"
    }
    $Lines = @($Output -split "`r?`n" | Where-Object { $_.Trim() })
    for ($Index = $Lines.Count - 1; $Index -ge 0; $Index--) {
        try {
            return ($Lines[$Index] | ConvertFrom-Json)
        }
        catch {
            continue
        }
    }
    throw 'Python contract command did not return JSON'
}

function Invoke-PowerShellScript {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [string[]]$Arguments = @()
    )

    $PowerShellExecutable = Join-Path $PSHOME 'powershell.exe'
    if (-not [IO.File]::Exists($PowerShellExecutable)) {
        throw 'PowerShell executable for child script is unavailable'
    }
    $ChildArguments = @(
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        $ScriptPath
    ) + $Arguments
    return Invoke-NativeProcessCaptured `
        -FilePath $PowerShellExecutable `
        -Arguments $ChildArguments `
        -AllowFailure
}

function Get-CalendarPaths {
    $CalendarRoot = Assert-PathWithinRoot `
        -Path (Join-Path $DataRoot 'publish\calendars\v1') `
        -Root $DataRoot
    $Year = $ParsedTargetDate.Year
    $Primary = if ($env:TWSE_CALENDAR_ARTIFACT) {
        $env:TWSE_CALENDAR_ARTIFACT
    }
    else {
        Join-Path $CalendarRoot "TW-$Year.json"
    }
    $Primary = Assert-PathWithinRoot -Path $Primary -Root $DataRoot
    $Paths = New-Object System.Collections.Generic.List[string]
    $Paths.Add($Primary) | Out-Null
    foreach ($CandidateYear in @(($Year - 1), ($Year + 1))) {
        $Candidate = Join-Path $CalendarRoot "TW-$CandidateYear.json"
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $ResolvedCandidate = Assert-PathWithinRoot `
                -Path $Candidate `
                -Root $DataRoot
            if (-not $Paths.Contains($ResolvedCandidate)) {
                $Paths.Add($ResolvedCandidate) | Out-Null
            }
        }
    }
    return $Paths.ToArray()
}

function Get-GcloudJsonAtGeneration {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)]$State,
        [Parameter(Mandatory)][string]$Label,
        [int]$MaximumBytes = 1MB
    )

    if (-not $State.exists -or [string]$State.generation -notmatch '^\d+$') {
        throw "GCS object is missing or has an invalid generation: $Label"
    }
    $Temporary = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ('absorb-catch-up-' + [Guid]::NewGuid().ToString('N') + '.json')
    try {
        Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
            'storage', 'cp', '--quiet',
            "$Uri#$([string]$State.generation)",
            $Temporary
        ) | Out-Null
        $Item = Get-Item -LiteralPath $Temporary -Force
        if (
            $Item.PSIsContainer -or
            $Item.Length -le 0 -or
            $Item.Length -gt $MaximumBytes
        ) {
            throw "GCS JSON object size is invalid: $Label"
        }
        try {
            $Document = Get-Content -LiteralPath $Temporary -Raw -Encoding utf8 |
                ConvertFrom-Json
        }
        catch {
            throw "GCS JSON object is invalid: $Label"
        }
        if ($null -eq $Document -or $Document -is [array]) {
            throw "GCS JSON object must be a top-level object: $Label"
        }
        return [pscustomobject]@{
            document = $Document
            generation = [string]$State.generation
            size = [long]$Item.Length
            sha256 = (Get-FileHash -LiteralPath $Temporary -Algorithm SHA256).Hash.ToLowerInvariant()
            uri = $Uri
        }
    }
    finally {
        if ([IO.File]::Exists($Temporary)) {
            [IO.File]::Delete($Temporary)
        }
    }
}

function Get-RemotePointer {
    param([Parameter(Mandatory)]$Definition)

    $State = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Definition.uri
    $Payload = Get-GcloudJsonAtGeneration `
        -Uri $Definition.uri `
        -State $State `
        -Label $Definition.name
    $Document = $Payload.document
    $EffectiveDate = $null
    $Identity = $null

    if ($Definition.kind -eq 'quant') {
        if (
            [int]$Document.schema_version -ne 3 -or
            [string]$Document.market -ne 'TW'
        ) {
            throw 'Live TW quant pointer schema is invalid'
        }
        $ManifestRelative = [string]$Document.manifest
        $ManifestHash = ([string]$Document.manifest_sha256).ToLowerInvariant()
        if (
            $ManifestRelative -notmatch
                '^manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
            $ManifestHash -notmatch '^[0-9a-f]{64}$' -or
            -not $ManifestRelative.EndsWith("-$($ManifestHash.Substring(0, 12)).json")
        ) {
            throw 'Live TW quant manifest identity is invalid'
        }
        $ManifestUri = "gs://$Bucket/quant/v1/$ManifestRelative"
        $ManifestState = Get-GcloudObjectState -Gcloud $Gcloud -Uri $ManifestUri
        $ManifestPayload = Get-GcloudJsonAtGeneration `
            -Uri $ManifestUri `
            -State $ManifestState `
            -Label 'live TW quant manifest' `
            -MaximumBytes 5MB
        if ($ManifestPayload.sha256 -ne $ManifestHash) {
            throw 'Live TW quant manifest hash mismatch'
        }
        $Manifest = $ManifestPayload.document
        if (
            [int]$Manifest.schema_version -ne 3 -or
            [string]$Manifest.market -ne 'TW' -or
            [string]$Manifest.observation_as_of -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Manifest.target_market_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Manifest.observation_as_of -ne [string]$Manifest.target_market_date
        ) {
            throw 'Live TW quant manifest date contract is invalid'
        }
        $EffectiveDate = ConvertTo-CanonicalDate `
            -Value $Manifest.observation_as_of `
            -Label 'live TW quant source date'
        $Identity = [ordered]@{
            source_date = $EffectiveDate.ToString('yyyy-MM-dd', $Invariant)
            manifest = $ManifestRelative
            manifest_sha256 = $ManifestHash
        }
    }
    elseif ($Definition.kind -eq 'dashboard') {
        if (
            [int]$Document.schema_version -ne 2 -or
            [string]$Document.kind -ne 'absorb-observation-dashboard' -or
            [string]$Document.product_mode -ne 'observation' -or
            [string]$Document.market -ne 'TW'
        ) {
            throw 'Live TW dashboard pointer schema is invalid'
        }
        $Manifest = [string]$Document.source_manifest
        $ManifestHash = ([string]$Document.source_manifest_sha256).ToLowerInvariant()
        $Object = [string]$Document.path
        $ObjectHash = ([string]$Document.sha256).ToLowerInvariant()
        if (
            $Manifest -notmatch '^quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
            $ManifestHash -notmatch '^[0-9a-f]{64}$' -or
            $Object -notmatch '^objects/[0-9a-f]{64}\.json$' -or
            $ObjectHash -notmatch '^[0-9a-f]{64}$' -or
            $Object -ne "objects/$ObjectHash.json" -or
            [long]$Document.size -le 0
        ) {
            throw 'Live TW dashboard pointer identity is invalid'
        }
        $EffectiveDate = ConvertTo-CanonicalDate `
            -Value $Document.observation_as_of `
            -Label 'live TW dashboard source date'
        $Identity = [ordered]@{
            source_date = $EffectiveDate.ToString('yyyy-MM-dd', $Invariant)
            source_manifest = $Manifest
            source_manifest_sha256 = $ManifestHash
            path = $Object
            sha256 = $ObjectHash
        }
    }
    elseif ($Definition.kind -eq 'reports-latest') {
        if (
            [int]$Document.schema_version -ne 2 -or
            [string]$Document.kind -ne 'absorb-report' -or
            [string]$Document.market -ne 'TW' -or
            [string]$Document.product_mode -ne 'observation' -or
            [string]$Document.report_type -ne 'post_close'
        ) {
            throw 'Live TW report latest pointer schema is invalid'
        }
        $MetadataRelative = [string]$Document.metadata
        $MetadataHash = ([string]$Document.metadata_sha256).ToLowerInvariant()
        if (
            $MetadataRelative -notmatch '^metadata/[0-9a-f]{64}\.json$' -or
            $MetadataHash -notmatch '^[0-9a-f]{64}$' -or
            $MetadataRelative -ne "metadata/$MetadataHash.json"
        ) {
            throw 'Live TW report metadata identity is invalid'
        }
        $MetadataUri = "gs://$Bucket/reports/v2/$MetadataRelative"
        $MetadataState = Get-GcloudObjectState -Gcloud $Gcloud -Uri $MetadataUri
        $MetadataPayload = Get-GcloudJsonAtGeneration `
            -Uri $MetadataUri `
            -State $MetadataState `
            -Label 'live TW report metadata' `
            -MaximumBytes 5MB
        if ($MetadataPayload.sha256 -ne $MetadataHash) {
            throw 'Live TW report metadata hash mismatch'
        }
        $Metadata = $MetadataPayload.document
        if (
            [string]$Metadata.product_mode -ne 'observation' -or
            [string]$Metadata.market -ne 'TW' -or
            [string]$Metadata.report_type -ne 'post_close' -or
            [string]$Metadata.source_market_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Metadata.applicable_trading_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
            [string]$Metadata.source_manifest -notmatch
                '^quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
            ([string]$Metadata.source_manifest_sha256).ToLowerInvariant() -notmatch
                '^[0-9a-f]{64}$'
        ) {
            throw 'Live TW report metadata schema is invalid'
        }
        $PointerSourceDate = ConvertTo-CanonicalDate `
            -Value $Document.source_market_date `
            -Label 'live TW report pointer source date'
        $PointerApplicableDate = ConvertTo-CanonicalDate `
            -Value $Document.applicable_trading_date `
            -Label 'live TW report pointer applicable date'
        $MetadataSourceDate = ConvertTo-CanonicalDate `
            -Value $Metadata.source_market_date `
            -Label 'live TW report metadata source date'
        $MetadataApplicableDate = ConvertTo-CanonicalDate `
            -Value $Metadata.applicable_trading_date `
            -Label 'live TW report metadata applicable date'
        if (
            $PointerSourceDate -ne $MetadataSourceDate -or
            $PointerApplicableDate -ne $MetadataApplicableDate
        ) {
            throw 'Live TW report pointer date identity mismatch'
        }
        $EffectiveDate = ConvertTo-CanonicalDate `
            -Value $Document.source_market_date `
            -Label 'live TW report source date'
        $Identity = [ordered]@{
            source_date = $EffectiveDate.ToString('yyyy-MM-dd', $Invariant)
            applicable_trading_date = $PointerApplicableDate.ToString('yyyy-MM-dd', $Invariant)
            source_manifest = [string]$Metadata.source_manifest
            source_manifest_sha256 = ([string]$Metadata.source_manifest_sha256).ToLowerInvariant()
            metadata = $MetadataRelative
            metadata_sha256 = $MetadataHash
        }
    }
    elseif ($Definition.kind -eq 'reports-index') {
        if (
            [int]$Document.schema_version -ne 2 -or
            [string]$Document.kind -ne 'absorb-report-index' -or
            [string]$Document.market -ne 'TW'
        ) {
            throw 'Live TW report index schema is invalid'
        }
        $Entries = @($Document.reports | Where-Object {
            [string]$_.report_type -eq 'post_close' -and
            [string]$_.product_mode -eq 'observation'
        })
        if ($Entries.Count -eq 0) {
            throw 'Live TW report index has no observation post-close entry'
        }
        $ParsedEntries = New-Object System.Collections.Generic.List[object]
        foreach ($Entry in $Entries) {
            $EntryDate = ConvertTo-CanonicalDate `
                -Value $Entry.source_market_date `
                -Label 'live TW report index source date'
            $EntryApplicableDate = ConvertTo-CanonicalDate `
                -Value $Entry.applicable_trading_date `
                -Label 'live TW report index applicable date'
            $EntryMetadata = [string]$Entry.metadata
            $EntryMetadataHash = ([string]$Entry.metadata_sha256).ToLowerInvariant()
            if (
                $EntryMetadata -notmatch '^metadata/[0-9a-f]{64}\.json$' -or
                $EntryMetadataHash -notmatch '^[0-9a-f]{64}$' -or
                $EntryMetadata -ne "metadata/$EntryMetadataHash.json"
            ) {
                throw 'Live TW report index metadata hash is invalid'
            }
            $ParsedEntries.Add([pscustomobject]@{
                date = $EntryDate
                applicable_trading_date = $EntryApplicableDate
                metadata = $EntryMetadata
                metadata_sha256 = $EntryMetadataHash
            }) | Out-Null
        }
        $LatestEntry = $ParsedEntries | Sort-Object date | Select-Object -Last 1
        $LatestEntries = @($ParsedEntries | Where-Object {
            $_.date -eq $LatestEntry.date
        })
        if ($LatestEntries.Count -ne 1) {
            throw 'Live TW report index has duplicate or conflicting latest entries'
        }
        $EffectiveDate = [DateTime]$LatestEntry.date
        $Identity = [ordered]@{
            source_date = $EffectiveDate.ToString('yyyy-MM-dd', $Invariant)
            applicable_trading_date = $LatestEntry.applicable_trading_date.ToString('yyyy-MM-dd', $Invariant)
            metadata = [string]$LatestEntry.metadata
            metadata_sha256 = [string]$LatestEntry.metadata_sha256
        }
    }
    else {
        throw 'Unknown observation pointer kind'
    }

    return [pscustomobject]@{
        name = [string]$Definition.name
        kind = [string]$Definition.kind
        uri = [string]$Definition.uri
        exists = [bool]$State.exists
        generation = [string]$State.generation
        document = $Document
        size = [long]$Payload.size
        sha256 = [string]$Payload.sha256
        effective_date = $EffectiveDate
        identity = $Identity
    }
}

function Get-ObservationSnapshot {
    $Pointers = New-Object System.Collections.Generic.List[object]
    foreach ($Definition in $ObservationDefinitions) {
        $Pointers.Add((Get-RemotePointer -Definition $Definition)) | Out-Null
    }
    $Quant = $Pointers | Where-Object { $_.name -eq 'quant' }
    $Dashboard = $Pointers | Where-Object { $_.name -eq 'dashboard' }
    $ReportLatest = $Pointers | Where-Object { $_.name -eq 'reports-latest' }
    $ReportIndex = $Pointers | Where-Object { $_.name -eq 'reports-index' }
    if (
        [string]$Dashboard.identity.source_manifest -ne
            "quant/v1/$([string]$Quant.identity.manifest)" -or
        [string]$Dashboard.identity.source_manifest_sha256 -ne
            [string]$Quant.identity.manifest_sha256 -or
        [string]$ReportLatest.identity.source_manifest -ne
            "quant/v1/$([string]$Quant.identity.manifest)" -or
        [string]$ReportLatest.identity.source_manifest_sha256 -ne
            [string]$Quant.identity.manifest_sha256 -or
        [string]$Dashboard.identity.source_date -ne
            [string]$Quant.identity.source_date -or
        [string]$ReportLatest.identity.source_date -ne
            [string]$Quant.identity.source_date -or
        [string]$ReportLatest.identity.metadata_sha256 -ne
            [string]$ReportIndex.identity.metadata_sha256 -or
        [string]$ReportLatest.identity.applicable_trading_date -ne
            [string]$ReportIndex.identity.applicable_trading_date -or
        $ReportLatest.effective_date -ne $ReportIndex.effective_date
    ) {
        throw 'Live Observation pointers are not bound to one source identity'
    }
    return $Pointers.ToArray()
}

function Get-LocalQuantState {
    $QuantRoot = Assert-PathWithinRoot `
        -Path (Join-Path $DataRoot 'publish\quant\v1') `
        -Root $DataRoot
    $LatestInfo = Read-JsonWithinRoot `
        -Path (Join-Path $QuantRoot 'latest-TW.json') `
        -Root $QuantRoot `
        -MaximumBytes 100KB
    $Latest = $LatestInfo.document
    if (
        [int]$Latest.schema_version -ne 3 -or
        [string]$Latest.market -ne 'TW'
    ) {
        throw 'Local TW quant latest pointer schema is invalid'
    }
    $ManifestRelative = [string]$Latest.manifest
    $ManifestHash = ([string]$Latest.manifest_sha256).ToLowerInvariant()
    if (
        $ManifestRelative -notmatch
            '^manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$' -or
        $ManifestHash -notmatch '^[0-9a-f]{64}$' -or
        -not $ManifestRelative.EndsWith("-$($ManifestHash.Substring(0, 12)).json")
    ) {
        throw 'Local TW quant manifest identity is invalid'
    }
    $ManifestInfo = Read-JsonWithinRoot `
        -Path (Join-Path $QuantRoot $ManifestRelative.Replace('/', '\')) `
        -Root $QuantRoot `
        -MaximumBytes 5MB
    if ($ManifestInfo.sha256 -ne $ManifestHash) {
        throw 'Local TW quant manifest hash mismatch'
    }
    $Manifest = $ManifestInfo.document
    if (
        [int]$Manifest.schema_version -ne 3 -or
        [string]$Manifest.market -ne 'TW' -or
        [string]$Manifest.generated_at -ne [string]$Latest.generated_at -or
        [string]$Manifest.observation_as_of -ne $TargetDate -or
        [string]$Manifest.target_market_date -ne $TargetDate -or
        [string]$Manifest.observation_as_of -ne [string]$Manifest.target_market_date
    ) {
        throw 'Local TW quant manifest date or generation contract is invalid'
    }
    return [pscustomobject]@{
        latest_path = $LatestInfo.path
        latest = $Latest
        manifest_path = $ManifestInfo.path
        manifest = $Manifest
        manifest_relative = $ManifestRelative
        manifest_sha256 = $ManifestHash
        identity = [ordered]@{
            source_date = $TargetDate
            manifest = $ManifestRelative
            manifest_sha256 = $ManifestHash
        }
    }
}

function Assert-TerminalCheckpoint {
    $CheckpointInfo = Read-JsonWithinRoot `
        -Path (Join-Path $DataRoot 'checkpoints\progress.json') `
        -Root (Join-Path $DataRoot 'checkpoints') `
        -MaximumBytes 1MB
    $Checkpoint = $CheckpointInfo.document
    $Failures = @($Checkpoint.failed)
    if (
        [string]$Checkpoint.stage -ne 'market_batch' -or
        [string]$Checkpoint.market -ne 'TW' -or
        $null -eq $Checkpoint.batch_identity -or
        [string]$Checkpoint.batch_identity.target_market_date -ne $TargetDate -or
        [string]$Checkpoint.batch_identity.product_mode -ne 'observation' -or
        $Failures.Count -ne 0 -or
        $null -eq $Checkpoint.next_index
    ) {
        throw 'TW post-close terminal checkpoint contract is incomplete'
    }
    return $CheckpointInfo
}

function Assert-ObservationCandidate {
    param(
        [Parameter(Mandatory)][string]$CandidatePath,
        [Parameter(Mandatory)]$Quant
    )

    $CandidateRoot = Assert-PathWithinRoot `
        -Path (Join-Path $DataRoot 'outputs\observation\candidates') `
        -Root $DataRoot
    $CandidateInfo = Read-JsonWithinRoot `
        -Path (Join-Path $CandidatePath 'candidate.json') `
        -Root $CandidatePath `
        -MaximumBytes 1MB
    $Candidate = $CandidateInfo.document
    if (
        [int]$Candidate.schema_version -ne 1 -or
        [string]$Candidate.kind -ne 'absorb-observation-candidate' -or
        [string]$Candidate.product_mode -ne 'observation' -or
        [string]$Candidate.observation_as_of -ne $TargetDate
    ) {
        throw 'Observation candidate manifest identity is invalid'
    }
    $ExpectedManifest = "quant/v1/$($Quant.manifest_relative)"
    $DashboardInfo = Read-JsonWithinRoot `
        -Path (Join-Path $CandidatePath 'dashboard-snapshot.json') `
        -Root $CandidatePath `
        -MaximumBytes 5MB
    $ReportInfo = Read-JsonWithinRoot `
        -Path (Join-Path $CandidatePath 'post-close-report-v2.json') `
        -Root $CandidatePath `
        -MaximumBytes 5MB
    foreach ($FileName in @('dashboard-snapshot.json', 'post-close-report-v2.json')) {
        $Expected = $Candidate.files.$FileName
        $Actual = if ($FileName -eq 'dashboard-snapshot.json') {
            $DashboardInfo
        }
        else {
            $ReportInfo
        }
        if (
            $null -eq $Expected -or
            [long]$Expected.size -ne [long]$Actual.size -or
            ([string]$Expected.sha256).ToLowerInvariant() -ne $Actual.sha256
        ) {
            throw "Observation candidate file hash mismatch: $FileName"
        }
    }
    $Dashboard = $DashboardInfo.document
    $Report = $ReportInfo.document
    if (
        [string]$Dashboard.observation_as_of -ne $TargetDate -or
        [string]$Dashboard.source_manifest -ne $ExpectedManifest -or
        [string]$Dashboard.source_manifest_sha256 -ne $Quant.manifest_sha256 -or
        [string]$Dashboard.product_mode -ne 'observation' -or
        [string]$Report.source_market_date -ne $TargetDate -or
        [string]$Report.source_manifest -ne $ExpectedManifest -or
        [string]$Report.source_manifest_sha256 -ne $Quant.manifest_sha256 -or
        [string]$Report.product_mode -ne 'observation' -or
        [string]$Report.market -ne 'TW' -or
        [string]$Report.report_type -ne 'post_close'
    ) {
        throw 'Observation candidate is not bound to the exact TW quant manifest'
    }
    return [pscustomobject]@{
        path = (Assert-PathWithinRoot -Path $CandidatePath -Root $CandidateRoot)
        manifest = $Candidate
        dashboard = $Dashboard
        report = $Report
        dashboard_path = $DashboardInfo.path
        report_path = $ReportInfo.path
    }
}

function Get-LocalObservationPointers {
    param([Parameter(Mandatory)]$Quant)

    $DashboardRoot = Assert-PathWithinRoot `
        -Path (Join-Path $DataRoot 'publish\dashboard\v1') `
        -Root $DataRoot
    $ReportRoot = Assert-PathWithinRoot `
        -Path (Join-Path $DataRoot 'publish\reports\v2') `
        -Root $DataRoot
    $DashboardInfo = Read-JsonWithinRoot `
        -Path (Join-Path $DashboardRoot 'latest-TW.json') `
        -Root $DashboardRoot `
        -MaximumBytes 100KB
    $Dashboard = $DashboardInfo.document
    if (
        [int]$Dashboard.schema_version -ne 2 -or
        [string]$Dashboard.kind -ne 'absorb-observation-dashboard' -or
        [string]$Dashboard.product_mode -ne 'observation' -or
        [string]$Dashboard.market -ne 'TW' -or
        [string]$Dashboard.observation_as_of -ne $TargetDate -or
        [string]$Dashboard.source_manifest -ne "quant/v1/$($Quant.manifest_relative)" -or
        [string]$Dashboard.source_manifest_sha256 -ne $Quant.manifest_sha256
    ) {
        throw 'Local dashboard pointer is not bound to the target quant manifest'
    }
    $ReportLatestInfo = Read-JsonWithinRoot `
        -Path (Join-Path $ReportRoot 'latest-TW-post_close.json') `
        -Root $ReportRoot `
        -MaximumBytes 100KB
    $ReportLatest = $ReportLatestInfo.document
    if (
        [int]$ReportLatest.schema_version -ne 2 -or
        [string]$ReportLatest.kind -ne 'absorb-report' -or
        [string]$ReportLatest.market -ne 'TW' -or
        [string]$ReportLatest.product_mode -ne 'observation' -or
        [string]$ReportLatest.report_type -ne 'post_close'
    ) {
        throw 'Local report latest pointer schema is invalid'
    }
    $ReportLatestSourceDate = ConvertTo-CanonicalDate `
        -Value $ReportLatest.source_market_date `
        -Label 'local TW report pointer source date'
    $ReportLatestApplicableDate = ConvertTo-CanonicalDate `
        -Value $ReportLatest.applicable_trading_date `
        -Label 'local TW report pointer applicable date'
    $MetadataRelative = [string]$ReportLatest.metadata
    $MetadataHash = ([string]$ReportLatest.metadata_sha256).ToLowerInvariant()
    if (
        $MetadataRelative -notmatch '^metadata/[0-9a-f]{64}\.json$' -or
        $MetadataHash -notmatch '^[0-9a-f]{64}$' -or
        $MetadataRelative -ne "metadata/$MetadataHash.json"
    ) {
        throw 'Local report metadata identity is invalid'
    }
    $MetadataInfo = Read-JsonWithinRoot `
        -Path (Join-Path $ReportRoot $MetadataRelative.Replace('/', '\')) `
        -Root $ReportRoot `
        -MaximumBytes 5MB
    if ($MetadataInfo.sha256 -ne $MetadataHash) {
        throw 'Local report metadata hash mismatch'
    }
    $Metadata = $MetadataInfo.document
    if (
        [string]$Metadata.source_market_date -ne $TargetDate -or
        [string]$Metadata.applicable_trading_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
        [string]$Metadata.source_manifest -ne "quant/v1/$($Quant.manifest_relative)" -or
        [string]$Metadata.source_manifest_sha256 -ne $Quant.manifest_sha256 -or
        [string]$Metadata.product_mode -ne 'observation' -or
        [string]$Metadata.market -ne 'TW' -or
        [string]$Metadata.report_type -ne 'post_close'
    ) {
        throw 'Local report metadata is not bound to the target quant manifest'
    }
    $MetadataSourceDate = ConvertTo-CanonicalDate `
        -Value $Metadata.source_market_date `
        -Label 'local TW report metadata source date'
    $MetadataApplicableDate = ConvertTo-CanonicalDate `
        -Value $Metadata.applicable_trading_date `
        -Label 'local TW report metadata applicable date'
    if (
        $ReportLatestSourceDate -ne $MetadataSourceDate -or
        $ReportLatestApplicableDate -ne $MetadataApplicableDate -or
        $ReportLatestSourceDate.ToString('yyyy-MM-dd', $Invariant) -ne $TargetDate
    ) {
        throw 'Local report pointer date identity mismatch'
    }
    $IndexInfo = Read-JsonWithinRoot `
        -Path (Join-Path $ReportRoot 'index-TW.json') `
        -Root $ReportRoot `
        -MaximumBytes 1MB
    $Index = $IndexInfo.document
    $IndexEntries = @($Index.reports | Where-Object {
        [string]$_.report_type -eq 'post_close' -and
        [string]$_.product_mode -eq 'observation' -and
        [string]$_.source_market_date -eq $TargetDate -and
        [string]$_.metadata_sha256 -eq $MetadataHash
    })
    if (
        [int]$Index.schema_version -ne 2 -or
        [string]$Index.kind -ne 'absorb-report-index' -or
        [string]$Index.market -ne 'TW' -or
        $IndexEntries.Count -ne 1
    ) {
        throw 'Local report index is not bound to the target report'
    }
    $IndexApplicableDate = ConvertTo-CanonicalDate `
        -Value $IndexEntries[0].applicable_trading_date `
        -Label 'local TW report index applicable date'
    if ($IndexApplicableDate -ne $ReportLatestApplicableDate) {
        throw 'Local report index date identity mismatch'
    }
    return [ordered]@{
        quant = [ordered]@{
            path = $Quant.latest_path
            identity = $Quant.identity
        }
        dashboard = [ordered]@{
            path = $DashboardInfo.path
            identity = [ordered]@{
                source_date = $TargetDate
                source_manifest = [string]$Dashboard.source_manifest
                source_manifest_sha256 = [string]$Dashboard.source_manifest_sha256
                path = [string]$Dashboard.path
                sha256 = ([string]$Dashboard.sha256).ToLowerInvariant()
            }
        }
        reports_latest = [ordered]@{
            path = $ReportLatestInfo.path
            identity = [ordered]@{
                source_date = $TargetDate
                applicable_trading_date = $ReportLatestApplicableDate.ToString('yyyy-MM-dd', $Invariant)
                source_manifest = [string]$Metadata.source_manifest
                source_manifest_sha256 = ([string]$Metadata.source_manifest_sha256).ToLowerInvariant()
                metadata = $MetadataRelative
                metadata_sha256 = $MetadataHash
            }
        }
        reports_index = [ordered]@{
            path = $IndexInfo.path
            identity = [ordered]@{
                source_date = $TargetDate
                applicable_trading_date = $IndexApplicableDate.ToString('yyyy-MM-dd', $Invariant)
                metadata = [string]$IndexEntries[0].metadata
                metadata_sha256 = $MetadataHash
            }
        }
    }
}

function Assert-ObservationReportIndexCatchUpDelta {
    param(
        [Parameter(Mandatory)]$CapturedIndex,
        [Parameter(Mandatory)]$LocalPointers
    )

    if (
        [int]$CapturedIndex.schema_version -ne 2 -or
        [string]$CapturedIndex.kind -ne 'absorb-report-index' -or
        [string]$CapturedIndex.market -ne 'TW' -or
        $null -eq $CapturedIndex.reports
    ) {
        throw 'Captured report index is invalid for catch-up delta validation'
    }
    $IndexInfo = Read-JsonWithinRoot `
        -Path $LocalPointers.reports_index.path `
        -Root (Split-Path -Parent $LocalPointers.reports_index.path) `
        -MaximumBytes 1MB
    $LocalIndex = $IndexInfo.document
    if (
        [int]$LocalIndex.schema_version -ne 2 -or
        [string]$LocalIndex.kind -ne 'absorb-report-index' -or
        [string]$LocalIndex.market -ne 'TW' -or
        $null -eq $LocalIndex.reports
    ) {
        throw 'Local report index is invalid for catch-up delta validation'
    }

    $CapturedEntries = @($CapturedIndex.reports)
    $LocalEntries = @($LocalIndex.reports)
    $TargetEntries = @($LocalEntries | Where-Object {
        [string]$_.report_type -eq 'post_close' -and
        [string]$_.product_mode -eq 'observation' -and
        [string]$_.source_market_date -eq $TargetDate
    })
    if ($TargetEntries.Count -ne 1) {
        throw 'Catch-up report index must contain exactly one TargetDate observation entry'
    }
    $CapturedTargetEntries = @($CapturedEntries | Where-Object {
        [string]$_.report_type -eq 'post_close' -and
        [string]$_.product_mode -eq 'observation' -and
        [string]$_.source_market_date -eq $TargetDate
    })
    if ($CapturedTargetEntries.Count -ne 0) {
        throw 'Captured report index already contains TargetDate observation entry'
    }
    $TargetEntry = $TargetEntries[0]
    if (
        [string]$TargetEntry.metadata -ne
            [string]$LocalPointers.reports_latest.identity.metadata -or
        [string]$TargetEntry.metadata_sha256 -ne
            [string]$LocalPointers.reports_latest.identity.metadata_sha256 -or
        [string]$TargetEntry.applicable_trading_date -ne
            [string]$LocalPointers.reports_latest.identity.applicable_trading_date
    ) {
        throw 'TargetDate report index entry is not bound to the promoted report'
    }

    $ExpectedCounts = @{}
    foreach ($Entry in $CapturedEntries) {
        $Key = Get-CanonicalJson -Value $Entry
        $ExpectedCounts[$Key] = 1 + [int]($ExpectedCounts[$Key])
    }
    $ActualCounts = @{}
    foreach ($Entry in @($LocalEntries | Where-Object {
        -not (
            [string]$_.report_type -eq 'post_close' -and
            [string]$_.product_mode -eq 'observation' -and
            [string]$_.source_market_date -eq $TargetDate
        )
    })) {
        $Key = Get-CanonicalJson -Value $Entry
        $ActualCounts[$Key] = 1 + [int]($ActualCounts[$Key])
    }
    if ($ExpectedCounts.Count -ne $ActualCounts.Count) {
        throw 'Catch-up report index contains an unauthorized entry delta'
    }
    foreach ($Key in $ExpectedCounts.Keys) {
        if (
            -not $ActualCounts.ContainsKey($Key) -or
            [int]$ActualCounts[$Key] -ne [int]$ExpectedCounts[$Key]
        ) {
            throw 'Catch-up report index changed a captured entry'
        }
    }
}

function Assert-IdentityMatch {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)]$Expected,
        [Parameter(Mandatory)]$Actual
    )

    $Properties = if ($Expected -is [Collections.IDictionary]) {
        @($Expected.Keys)
    }
    else {
        @($Expected.PSObject.Properties.Name)
    }
    foreach ($Property in $Properties) {
        $ExpectedValue = if ($Expected -is [Collections.IDictionary]) {
            $Expected[$Property]
        }
        else {
            $Expected.$Property
        }
        $ActualValue = if ($Actual -is [Collections.IDictionary]) {
            $Actual[$Property]
        }
        else {
            $Actual.$Property
        }
        if ([string]$ExpectedValue -ne [string]$ActualValue) {
            throw "$Label identity mismatch: $Property"
        }
    }
}

function ConvertTo-CanonicalJsonValue {
    param($Value)

    if ($null -eq $Value) { return $null }
    if ($Value -is [Collections.IDictionary]) {
        $Normalized = [ordered]@{}
        foreach ($Key in @($Value.Keys | Sort-Object)) {
            $Normalized[[string]$Key] = ConvertTo-CanonicalJsonValue `
                -Value $Value[$Key]
        }
        return $Normalized
    }
    if ($Value -is [pscustomobject]) {
        $Normalized = [ordered]@{}
        foreach ($Property in @($Value.PSObject.Properties | Sort-Object Name)) {
            $Normalized[[string]$Property.Name] = ConvertTo-CanonicalJsonValue `
                -Value $Property.Value
        }
        return $Normalized
    }
    if ($Value -is [array]) {
        return ,@($Value | ForEach-Object {
            ConvertTo-CanonicalJsonValue -Value $_
        })
    }
    return $Value
}

function Get-CanonicalJson {
    param([Parameter(Mandatory)]$Value)

    $Normalized = ConvertTo-CanonicalJsonValue -Value $Value
    return ($Normalized | ConvertTo-Json -Depth 50 -Compress)
}

function Convert-PointerEvidence {
    param([Parameter(Mandatory)]$Pointer)

    return [ordered]@{
        name = $Pointer.name
        uri = $Pointer.uri
        generation = $Pointer.generation
        size = $Pointer.size
        sha256 = $Pointer.sha256
        effective_date = $Pointer.effective_date.ToString('yyyy-MM-dd', $Invariant)
        identity = $Pointer.identity
    }
}

function Assert-LkgMatchesSnapshot {
    param(
        [Parameter(Mandatory)]$Receipt,
        [Parameter(Mandatory)]$BeforeStates
    )

    foreach ($Definition in $AllPointerDefinitions) {
        $Matches = @($Receipt.pointers | Where-Object {
            [string]$_.uri -eq [string]$Definition.uri
        })
        if ($Matches.Count -ne 1) {
            throw "Observation LKG receipt is missing or duplicates $($Definition.name)"
        }
        $ReceiptPointer = $Matches[0]
        $Before = $BeforeStates | Where-Object { $_.uri -eq $Definition.uri }
        if ($null -eq $Before) {
            throw "Before-state is missing $($Definition.name)"
        }
        $BeforeGeneration = if ($Before.exists) {
            [string]$Before.generation
        }
        else {
            $null
        }
        $ReceiptGeneration = if ($ReceiptPointer.exists) {
            [string]$ReceiptPointer.generation
        }
        else {
            $null
        }
        if (
            [bool]$ReceiptPointer.exists -ne [bool]$Before.exists -or
            $ReceiptGeneration -ne $BeforeGeneration
        ) {
            throw "Observation LKG capture raced with the before-state: $($Definition.name)"
        }
    }
}

function Assert-NonObservationPointersUnchanged {
    param(
        [Parameter(Mandatory)]$BeforeStates,
        [Parameter(Mandatory)]$AfterStates
    )

    foreach ($Definition in $AllPointerDefinitions) {
        $Before = $BeforeStates | Where-Object { $_.uri -eq $Definition.uri }
        $After = $AfterStates | Where-Object { $_.uri -eq $Definition.uri }
        if ($null -eq $Before -or $null -eq $After) {
            throw "Pointer state is missing: $($Definition.name)"
        }
        if ($ObservationPointerNames -contains $Definition.name) {
            if (-not $After.exists) {
                throw "Observation pointer disappeared: $($Definition.name)"
            }
            continue
        }
        if (
            [bool]$Before.exists -ne [bool]$After.exists -or
            [string]$Before.generation -ne [string]$After.generation
        ) {
            throw "Unrelated pointer was mutated: $($Definition.name)"
        }
    }
}

function Assert-AllPointerStatesUnchanged {
    param(
        [Parameter(Mandatory)]$BeforeStates,
        [Parameter(Mandatory)]$AfterStates
    )

    foreach ($Definition in $AllPointerDefinitions) {
        $Before = $BeforeStates | Where-Object { $_.uri -eq $Definition.uri }
        $After = $AfterStates | Where-Object { $_.uri -eq $Definition.uri }
        if (
            $null -eq $Before -or
            $null -eq $After -or
            [bool]$Before.exists -ne [bool]$After.exists -or
            [string]$Before.generation -ne [string]$After.generation
        ) {
            throw "Idempotent catch-up observed a pointer mutation: $($Definition.name)"
        }
    }
}

function Get-AllPointerStates {
    $States = New-Object System.Collections.Generic.List[object]
    foreach ($Definition in $AllPointerDefinitions) {
        $State = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Definition.uri
        $States.Add([pscustomobject]@{
            name = $Definition.name
            uri = $Definition.uri
            exists = [bool]$State.exists
            generation = if ($State.exists) { [string]$State.generation } else { $null }
        }) | Out-Null
    }
    return $States.ToArray()
}

function Assert-RemoteReadbackMatchesLocal {
    param(
        [Parameter(Mandatory)]$Snapshot,
        [Parameter(Mandatory)]$LocalPointers
    )

    $Mapping = @(
        @('quant', 'quant'),
        @('reports-index', 'reports_index'),
        @('reports-latest', 'reports_latest'),
        @('dashboard', 'dashboard')
    )
    foreach ($Pair in $Mapping) {
        $Remote = $Snapshot | Where-Object { $_.name -eq $Pair[0] }
        $Local = $LocalPointers[$Pair[1]]
        Assert-GcloudFileMatches `
            -Gcloud $Gcloud `
            -LocalPath $Local.path `
            -Uri "$($Remote.uri)#$($Remote.generation)"
        Assert-IdentityMatch `
            -Label $Pair[0] `
            -Expected $Local.identity `
            -Actual $Remote.identity
    }
}

try {
    Acquire-CatchUpLock
    Assert-NoPendingPointerJournals
    Assert-PostCloseTaskIdle
    $CalendarPaths = Get-CalendarPaths
    $LiveBefore = Get-ObservationSnapshot
    $AllStatesBeforeNoop = Get-AllPointerStates
    $ContractArguments = @(
        '-m', 'stock_papi.batch.catch_up_latest_completed_session',
        '--target-date', $TargetDate,
        '--local-today', $LocalToday.ToString('yyyy-MM-dd', $Invariant)
    )
    foreach ($Path in $CalendarPaths) {
        $ContractArguments += @('--calendar-artifact', $Path)
    }
    foreach ($Pointer in $LiveBefore) {
        $ContractArguments += @(
            '--live-pointer',
            "$($Pointer.name)=$($Pointer.effective_date.ToString('yyyy-MM-dd', $Invariant))"
        )
    }
    $Contract = Invoke-PythonJson -Arguments $ContractArguments
    if ([string]$Contract.live.mode -notin @('publish', 'idempotent')) {
        throw 'Catch-up live pointer contract did not return a supported mode'
    }

    if ([string]$Contract.live.mode -eq 'idempotent') {
        $LiveAfter = Get-ObservationSnapshot
        foreach ($Remote in $LiveBefore) {
            $After = $LiveAfter | Where-Object { $_.name -eq $Remote.name }
            if (
                $null -eq $After -or
                [string]$After.generation -ne [string]$Remote.generation
            ) {
                throw "Idempotent catch-up observed a pointer mutation: $($Remote.name)"
            }
            Assert-IdentityMatch `
                -Label $Remote.name `
                -Expected $Remote.identity `
                -Actual $After.identity
        }
        Assert-AllPointerStatesUnchanged `
            -BeforeStates $AllStatesBeforeNoop `
            -AfterStates (Get-AllPointerStates)
        $ReleaseRoot = Join-Path $DataRoot 'release\catch-up'
        if (-not [IO.Directory]::Exists($ReleaseRoot)) {
            New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
        }
        $ReleaseRoot = Assert-PathWithinRoot -Path $ReleaseRoot -Root $DataRoot
        $RunId = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ-') +
            [Guid]::NewGuid().ToString('N').Substring(0, 8)
        $EvidencePath = Join-Path $ReleaseRoot "$TargetDate-$RunId.json"
        $Evidence = [ordered]@{
            schema_version = 1
            kind = 'absorb-tw-latest-completed-session-catch-up'
            mode = 'idempotent'
            target_date = $TargetDate
            local_today = $LocalToday.ToString('yyyy-MM-dd', $Invariant)
            normal_pipeline_validated = $false
            terminal_checkpoint_path = $null
            candidate_path = $null
            before = @($LiveBefore | ForEach-Object { Convert-PointerEvidence $_ })
            after = @($LiveAfter | ForEach-Object { Convert-PointerEvidence $_ })
            lkg_receipt = $null
        }
        Write-AtomicJson -Path $EvidencePath -Root $DataRoot -Document $Evidence
        $Evidence | ConvertTo-Json -Depth 20 -Compress
        exit 0
    }

    $PipelinePath = Join-Path $PSScriptRoot 'run_tw_post_close_pipeline.ps1'
    $PipelineResult = Invoke-PowerShellScript `
        -ScriptPath $PipelinePath `
        -Arguments @('-DataRoot', $DataRoot, '-TargetDate', $TargetDate)
    if ($PipelineResult.exit_code -ne 0) {
        throw 'Normal TW post-close validation pipeline failed'
    }
    $TerminalCheckpoint = Assert-TerminalCheckpoint
    $Quant = Get-LocalQuantState
    $CandidateArguments = @(
        '-m', 'stock_papi.batch.observation_products_cli',
        'build',
        '--root', $DataRoot,
        '--source-market-date', $TargetDate,
        '--source-validation-date', $TargetDate,
        '--source-manifest', "quant/v1/$($Quant.manifest_relative)",
        '--source-manifest-sha256', $Quant.manifest_sha256
    )
    foreach ($Path in $CalendarPaths) {
        $CandidateArguments += @('--calendar-artifact', $Path)
    }
    $CandidateResult = Invoke-PythonJson -Arguments $CandidateArguments
    $CandidatePath = Assert-PathWithinRoot `
        -Path ([string]$CandidateResult.candidate_path) `
        -Root (Join-Path $DataRoot 'outputs\observation\candidates')
    $Candidate = Assert-ObservationCandidate `
        -CandidatePath $CandidatePath `
        -Quant $Quant
    $PromoteArguments = @(
        '-m', 'stock_papi.batch.observation_products_cli',
        'promote',
        '--root', $DataRoot,
        '--candidate', $Candidate.path
    )
    [void](Invoke-PythonJson -Arguments $PromoteArguments)
    $LocalPointers = Get-LocalObservationPointers -Quant $Quant
    $CapturedReportIndex = $LiveBefore | Where-Object { $_.name -eq 'reports-index' }
    Assert-ObservationReportIndexCatchUpDelta `
        -CapturedIndex $CapturedReportIndex.document `
        -LocalPointers $LocalPointers

    if ([string]$Contract.live.mode -ne 'publish') {
        throw 'Catch-up publish mode was not proven by the live contract'
    }
    $LivePreCapture = Get-ObservationSnapshot
    $PreCaptureArguments = @(
        '-m', 'stock_papi.batch.catch_up_latest_completed_session',
        '--target-date', $TargetDate,
        '--local-today', $LocalToday.ToString('yyyy-MM-dd', $Invariant)
    )
    foreach ($Path in $CalendarPaths) {
        $PreCaptureArguments += @('--calendar-artifact', $Path)
    }
    foreach ($Pointer in $LivePreCapture) {
        $PreCaptureArguments += @(
            '--live-pointer',
            "$($Pointer.name)=$($Pointer.effective_date.ToString('yyyy-MM-dd', $Invariant))"
        )
    }
    $PreCaptureContract = Invoke-PythonJson -Arguments $PreCaptureArguments
    if ([string]$PreCaptureContract.live.mode -ne 'publish') {
        throw 'Live pointer state changed before LKG capture; rerun catch-up'
    }
    foreach ($Pointer in $LiveBefore) {
        $Current = $LivePreCapture | Where-Object { $_.name -eq $Pointer.name }
        if (
            $null -eq $Current -or
            [string]$Current.generation -ne [string]$Pointer.generation
        ) {
            throw "Live pointer generation changed before LKG capture: $($Pointer.name)"
        }
        Assert-IdentityMatch `
            -Label $Pointer.name `
            -Expected $Pointer.identity `
            -Actual $Current.identity
    }
    $BeforeStates = Get-AllPointerStates
    $CapturePath = Join-Path $PSScriptRoot 'capture_observation_lkg.ps1'
    $CaptureResult = Invoke-PowerShellScript `
        -ScriptPath $CapturePath `
        -Arguments @('-DataRoot', $DataRoot, '-Bucket', $Bucket)
    if ($CaptureResult.exit_code -ne 0) {
        throw 'Observation LKG capture failed'
    }
    try {
        $Capture = $CaptureResult.text.Trim() | ConvertFrom-Json
    }
    catch {
        throw 'Observation LKG capture did not return valid JSON'
    }
    $ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
    $LkgReceiptPath = Assert-PathWithinRoot `
        -Path ([string]$Capture.receipt) `
        -Root $ReceiptRoot
    $ReceiptInfo = Read-JsonWithinRoot `
        -Path $LkgReceiptPath `
        -Root $ReceiptRoot `
        -MaximumBytes 1MB
    $Receipt = $ReceiptInfo.document
    Assert-LkgMatchesSnapshot -Receipt $Receipt -BeforeStates $BeforeStates
    $PendingJournalPath = "$LkgReceiptPath.pending.json"
    if (
        [IO.File]::Exists($PendingJournalPath) -or
        [IO.File]::Exists("$PendingJournalPath.tmp")
    ) {
        throw 'Observation LKG pending pointer journal already exists'
    }

    $ReleaseRoot = Join-Path $DataRoot 'release\catch-up'
    if (-not [IO.Directory]::Exists($ReleaseRoot)) {
        New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
    }
    $ReleaseRoot = Assert-PathWithinRoot -Path $ReleaseRoot -Root $DataRoot
    $RunId = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ-') +
        [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $EvidencePath = Join-Path $ReleaseRoot "$TargetDate-$RunId.json"
    $BeforeReceiptCopy = Join-Path $ReleaseRoot "$TargetDate-$RunId-lkg-before.json"
    [IO.File]::Copy($LkgReceiptPath, $BeforeReceiptCopy, $false)
    $BeforeEvidence = [ordered]@{
        schema_version = 1
        kind = 'absorb-tw-latest-completed-session-catch-up-before'
        target_date = $TargetDate
        local_today = $LocalToday.ToString('yyyy-MM-dd', $Invariant)
        normal_pipeline_validated = $true
        terminal_checkpoint_path = $TerminalCheckpoint.path
        candidate_path = $Candidate.path
        lkg_receipt = $LkgReceiptPath
        lkg_receipt_before_copy = $BeforeReceiptCopy
        pointers = @($LiveBefore | ForEach-Object { Convert-PointerEvidence $_ })
    }
    Write-AtomicJson `
        -Path (Join-Path $ReleaseRoot "$TargetDate-$RunId-before.json") `
        -Root $DataRoot `
        -Document $BeforeEvidence

    $UploadPath = Join-Path $PSScriptRoot 'upload_local_quant.ps1'
    $UploadResult = Invoke-PowerShellScript `
        -ScriptPath $UploadPath `
        -Arguments @(
            '-DataRoot',
            $DataRoot,
            '-Bucket',
            $Bucket,
            '-RequireReportV2',
            '-RequireDashboard',
            '-ObservationOnly',
            '-LkgReceiptPath',
            $LkgReceiptPath
        )
    if ($UploadResult.exit_code -ne 0) {
        throw 'Observation catch-up upload failed; inspect LKG receipt and pending journal'
    }
    if (
        [IO.File]::Exists($PendingJournalPath) -or
        [IO.File]::Exists("$PendingJournalPath.tmp")
    ) {
        throw 'Observation catch-up upload left a pending pointer journal'
    }
    $LiveAfter = Get-ObservationSnapshot
    $AfterPointerArguments = @(
        '-m', 'stock_papi.batch.catch_up_latest_completed_session',
        '--target-date', $TargetDate,
        '--local-today', $LocalToday.ToString('yyyy-MM-dd', $Invariant)
    )
    foreach ($Path in $CalendarPaths) {
        $AfterPointerArguments += @('--calendar-artifact', $Path)
    }
    foreach ($Pointer in $LiveAfter) {
        $AfterPointerArguments += @(
            '--live-pointer',
            "$($Pointer.name)=$($Pointer.effective_date.ToString('yyyy-MM-dd', $Invariant))"
        )
    }
    $AfterContract = Invoke-PythonJson -Arguments $AfterPointerArguments
    if ([string]$AfterContract.live.mode -ne 'idempotent') {
        throw 'Catch-up upload did not advance every live pointer to TargetDate'
    }
    foreach ($Pointer in $LiveAfter) {
        if ($Pointer.effective_date -ne $ParsedTargetDate) {
            throw "Catch-up pointer date mismatch: $($Pointer.name)"
        }
    }
    Assert-RemoteReadbackMatchesLocal `
        -Snapshot $LiveAfter `
        -LocalPointers $LocalPointers
    $AfterStates = Get-AllPointerStates
    Assert-NonObservationPointersUnchanged `
        -BeforeStates $BeforeStates `
        -AfterStates $AfterStates
    foreach ($Definition in $AllPointerDefinitions | Where-Object {
        $ObservationPointerNames -contains $_.name
    }) {
        $Before = $BeforeStates | Where-Object { $_.uri -eq $Definition.uri }
        $After = $AfterStates | Where-Object { $_.uri -eq $Definition.uri }
        if ($Before.exists -and [string]$Before.generation -eq [string]$After.generation) {
            throw "Expected observation pointer generation did not change: $($Definition.name)"
        }
    }
    $FinalEvidence = [ordered]@{
        schema_version = 1
        kind = 'absorb-tw-latest-completed-session-catch-up'
        mode = 'published'
        target_date = $TargetDate
        local_today = $LocalToday.ToString('yyyy-MM-dd', $Invariant)
        normal_pipeline_validated = $true
        terminal_checkpoint_path = $TerminalCheckpoint.path
        candidate_path = $Candidate.path
        lkg_receipt = $LkgReceiptPath
        lkg_receipt_before_copy = $BeforeReceiptCopy
        before = @($LiveBefore | ForEach-Object { Convert-PointerEvidence $_ })
        after = @($LiveAfter | ForEach-Object { Convert-PointerEvidence $_ })
        all_pointer_states_before = $BeforeStates
        all_pointer_states_after = $AfterStates
    }
    Write-AtomicJson -Path $EvidencePath -Root $DataRoot -Document $FinalEvidence
    $FinalEvidence | ConvertTo-Json -Depth 20 -Compress
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
finally {
    if ($null -ne $CatchUpLockStream) {
        $CatchUpLockStream.Dispose()
        $CatchUpLockStream = $null
    }
    if ($null -ne $ObservationWriterMutex) {
        if ($ObservationWriterMutexHeld) {
            try { $ObservationWriterMutex.ReleaseMutex() } catch { }
        }
        $ObservationWriterMutex.Dispose()
        $ObservationWriterMutex = $null
        $ObservationWriterMutexHeld = $false
    }
    $env:PYTHONPATH = $OldPythonPath
}
