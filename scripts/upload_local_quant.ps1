[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots',
    [switch]$RequireReportV2,
    [switch]$RequireDashboard,
    [switch]$ObservationOnly,
    [string]$LkgReceiptPath,
    [string]$PreflightDataRoot,
    [switch]$ReportV2Only,
    [string]$Market = ''
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) { throw 'Data root is not allowlisted' }
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') { throw 'Bucket is not allowlisted' }
if (
    $ReportV2Only -and
    (-not $ObservationOnly -or -not $RequireReportV2 -or $RequireDashboard)
) { throw 'ReportV2Only requires ObservationOnly and RequireReportV2 only' }
. (Join-Path $PSScriptRoot 'observation_release_common.ps1')

$PublishRoot = Join-Path `
    $(if ($PreflightDataRoot) { $PreflightDataRoot } else { $DataRoot }) `
    'publish\quant\v1'
$ResolvedRoot = (Resolve-Path -LiteralPath $PublishRoot).Path
if (((Get-Item -LiteralPath $ResolvedRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw 'Publish root must not be a reparse point'
}
$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$ObjectBatchSize = 100

$Global:VerifiedDirs = @{}
$Global:PointerUpdates = New-Object System.Collections.Generic.List[object]
$ExpectedPointerGenerations = @{}
$PendingPointerJournalPath = $null
$Global:PointerStagingRunPath = $null
$Global:LkgPointerLockStream = $null
$ObservationOnlyPointerAllowlist = @(
    "gs://$Bucket/quant/v1/latest-TW.json",
    "gs://$Bucket/quant/v1/latest-US.json",
    "gs://$Bucket/reports/v2/index-TW.json",
    "gs://$Bucket/reports/v2/index-US.json",
    "gs://$Bucket/reports/v2/latest-TW-post_close.json",
    "gs://$Bucket/reports/v2/latest-US-post_close.json",
    "gs://$Bucket/reports/v2/latest-TW-pre_market.json",
    "gs://$Bucket/reports/v2/latest-US-pre_market.json",
    "gs://$Bucket/dashboard/v1/latest-TW.json",
    "gs://$Bucket/dashboard/v1/latest-US.json"
)
$ReceiptUpdated = $false

function Send-ReportUploadFailureNotification {
    param([string]$Message)
    $AdminUserId = [string]$env:REPORT_ADMIN_USER_ID
    if ($AdminUserId -notmatch '^U[0-9a-f]{32}$') { return }
    $OldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $null
        $Token = (& $Gcloud secrets versions access latest `
            --secret=stock-papi-line-channel-access-token `
            --project=line-stock-bot-498908 2>$null).Trim()
        if (-not $Token) { return }
        $Headers = @{ Authorization = "Bearer $Token"; 'Content-Type' = 'application/json' }
        $Body = @{
            to = $AdminUserId
            messages = @(@{ type = 'text'; text = $Message })
        } | ConvertTo-Json -Depth 5 -Compress
        Invoke-RestMethod -Method Post -Uri 'https://api.line.me/v2/bot/message/push' `
            -Headers $Headers -Body $Body -TimeoutSec 10 | Out-Null
    } catch {
        Write-Warning 'LINE administrator notification failed'
    } finally {
        $env:PYTHONPATH = $OldPythonPath
    }
}

function Assert-AllowlistedPath {
    param([string]$Path)
    return Assert-PathWithinRoot `
        -Path $Path `
        -Root $ResolvedRoot `
        -VerifiedDirs $Global:VerifiedDirs
}

function Read-VerifiedGzipJson {
    param(
        [string]$Path,
        [long]$ExpectedSize
    )
    if ($ExpectedSize -le 0 -or $ExpectedSize -gt 20MB) {
        throw 'Invalid uncompressed object size'
    }
    $InputStream = [IO.File]::OpenRead($Path)
    try {
        $Gzip = [IO.Compression.GzipStream]::new(
            $InputStream,
            [IO.Compression.CompressionMode]::Decompress
        )
        try {
            $Output = [IO.MemoryStream]::new()
            try {
                $Buffer = New-Object byte[] 81920
                while (($Read = $Gzip.Read($Buffer, 0, $Buffer.Length)) -gt 0) {
                    $Output.Write($Buffer, 0, $Read)
                    if ($Output.Length -gt 20MB) {
                        throw 'Object expands beyond limit'
                    }
                }
                if ($Output.Length -ne $ExpectedSize) {
                    throw 'Object uncompressed size mismatch'
                }
                $Utf8 = [Text.UTF8Encoding]::new($false, $true)
                return $Utf8.GetString($Output.ToArray()) | ConvertFrom-Json
            } finally {
                $Output.Dispose()
            }
        } finally {
            $Gzip.Dispose()
        }
    } finally {
        $InputStream.Dispose()
    }
}

function Test-JsonInteger {
    param($Value)
    return $Value -is [byte] -or $Value -is [int16] -or
        $Value -is [int32] -or $Value -is [int64]
}

function Invoke-GcloudCopy {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$NoClobber
    )
    $Arguments = @('storage', 'cp', '--quiet')
    if ($NoClobber) { $Arguments += '--no-clobber' }
    $Arguments += @($Source, $Destination)

    $Result = Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments $Arguments
    if ($Result.text) { Write-Output $Result.text.TrimEnd() }
}

function Invoke-GcloudCopyBatch {
    param(
        [string[]]$Sources,
        [string]$Destination
    )
    if (-not $Sources -or $Sources.Count -eq 0) { return }
    $Arguments = @('storage', 'cp', '--quiet', '--no-clobber')
    $Arguments += $Sources
    $Arguments += $Destination

    $Result = Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments $Arguments
    if ($Result.text) { Write-Output $Result.text.TrimEnd() }
}

function Set-GcloudMutablePointer {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$ExpectedSha256
    )
    if (
        $ObservationOnly -and
        $ObservationOnlyPointerAllowlist -notcontains $Destination
    ) {
        throw 'Observation-only pointer destination is not allowlisted'
    }
    $ExpectedGeneration = $null
    if ($LkgReceiptPath) {
        if (-not $ExpectedPointerGenerations.ContainsKey($Destination)) {
            throw 'Observation LKG receipt is missing pointer'
        }
        $ExpectedGeneration = [string]$ExpectedPointerGenerations[$Destination]
    }
    $Source = Assert-PathWithinRoot -Path $Source -Root $DataRoot
    if (-not $Global:PointerStagingRunPath) {
        $ReleaseRoot = Join-Path $DataRoot 'release'
        if (-not [IO.Directory]::Exists($ReleaseRoot)) {
            New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
        }
        $ReleaseRoot = Assert-PathWithinRoot -Path $ReleaseRoot -Root $DataRoot
        $StagingParent = Join-Path $ReleaseRoot 'pointer-staging'
        if (-not [IO.Directory]::Exists($StagingParent)) {
            New-Item -ItemType Directory -Path $StagingParent -Force | Out-Null
        }
        $StagingParent = Assert-PathWithinRoot -Path $StagingParent -Root $DataRoot
        $RunName = 'run-' + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ-') +
            [Guid]::NewGuid().ToString('N')
        $Global:PointerStagingRunPath = Assert-PathWithinRoot `
            -Path (New-Item `
                -ItemType Directory `
                -Path (Join-Path $StagingParent $RunName) `
                -Force).FullName `
            -Root $StagingParent
    }
    $Snapshot = New-VerifiedPointerSnapshot `
        -Source $Source `
        -StagingRunPath $Global:PointerStagingRunPath `
        -ExpectedSha256 $ExpectedSha256
    $Update = Invoke-GcloudConditionalCopy `
        -Gcloud $Gcloud `
        -Source $Snapshot `
        -Destination $Destination `
        -ExpectedGeneration $ExpectedGeneration `
        -ExpectedSourceSha256 $ExpectedSha256 `
        -PendingJournalPath $PendingPointerJournalPath `
        -SkipIfMatches
    if ($Update.changed) {
        $Global:PointerUpdates.Add([pscustomobject]$Update) | Out-Null
    }
    return $Update
}

function Release-LkgPointerLock {
    if ($null -ne $Global:LkgPointerLockStream) {
        $Global:LkgPointerLockStream.Dispose()
        $Global:LkgPointerLockStream = $null
    }
}

function Update-ObservationLkgReceipt {
    if (-not $LkgReceiptPath) { return }
    $ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
    $ResolvedReceipt = Assert-PathWithinRoot `
        -Path $LkgReceiptPath `
        -Root $ReceiptRoot
    $Receipt = Get-Content -LiteralPath $ResolvedReceipt -Raw -Encoding utf8 |
        ConvertFrom-Json
    if (
        $Receipt.schema_version -ne 1 -or
        $Receipt.kind -ne 'absorb-observation-lkg' -or
        $Receipt.bucket -ne $Bucket -or
        -not ($Receipt.pointers -is [array])
    ) {
        throw 'Observation LKG receipt is invalid'
    }
    foreach ($Update in $Global:PointerUpdates) {
        $Matches = @($Receipt.pointers | Where-Object {
            [string]$_.uri -eq [string]$Update.uri
        })
        if ($Matches.Count -eq 0) { throw 'Observation LKG receipt is missing pointer' }
        if ($Matches.Count -ne 1) {
            throw 'Observation LKG receipt contains duplicate pointers'
        }
        $Pointer = $Matches[0]
        $ExpectedBefore = if (
            [string]$Pointer.applied_generation -match '^\d+$'
        ) {
            [string]$Pointer.applied_generation
        } elseif ($Pointer.exists) {
            [string]$Pointer.generation
        } else {
            '0'
        }
        if ([string]$Update.before_generation -ne $ExpectedBefore) {
            throw 'Observation LKG generation changed before update'
        }
        $Pointer | Add-Member `
            -NotePropertyName applied_generation `
            -NotePropertyValue ([string]$Update.after_generation) `
            -Force
    }
    $Receipt | Add-Member `
        -NotePropertyName applied_at `
        -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o')) `
        -Force
    $Temporary = "$ResolvedReceipt.tmp"
    [IO.File]::WriteAllText(
        $Temporary,
        ($Receipt | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $Temporary -Destination $ResolvedReceipt -Force
    Remove-GcloudPendingPointerJournal -Path $PendingPointerJournalPath
}

function Assert-ObservationReportIndexPreservesLkg {
    param([object]$LocalIndex)
    if (-not $ObservationOnly -or -not $LkgReceiptPath) { return }
    $IndexUri = "gs://$Bucket/reports/v2/index-TW.json"
    $CapturedPointers = @($ReceiptPreflight.pointers | Where-Object {
        [string]$_.uri -eq $IndexUri
    })
    if ($CapturedPointers.Count -ne 1) {
        throw 'Observation LKG receipt is missing report index capture'
    }
    $CapturedPointer = $CapturedPointers[0]
    if (-not $CapturedPointer.exists) { return }
    $CaptureRoot = Split-Path -Parent $LkgReceiptPath
    $PreviousFile = [string]$CapturedPointer.previous_file
    if ($PreviousFile -notmatch '^[^\\/:*?"<>|]+$') {
        throw 'Observation LKG report index capture path is invalid'
    }
    $CapturedPath = Assert-PathWithinRoot `
        -Path (Join-Path $CaptureRoot $PreviousFile) `
        -Root $CaptureRoot
    $CapturedFile = Get-Item -LiteralPath $CapturedPath
    if (
        $CapturedFile.PSIsContainer -or
        [long]$CapturedPointer.previous_size -ne $CapturedFile.Length -or
        [string]$CapturedPointer.previous_sha256 -notmatch '^[0-9a-f]{64}$' -or
        (Get-FileHash -LiteralPath $CapturedPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            [string]$CapturedPointer.previous_sha256
    ) {
        throw 'Observation LKG captured report index checksum mismatch'
    }
    try {
        $CapturedIndex = Get-Content `
            -LiteralPath $CapturedPath `
            -Raw `
            -Encoding utf8 | ConvertFrom-Json
    } catch {
        throw 'Observation LKG captured report index is invalid'
    }
    if (
        $CapturedIndex.schema_version -ne 2 -or
        $CapturedIndex.kind -notin @('absorb-report-index', 'stock-papi-report-index') -or
        $CapturedIndex.market -notin @('TW', 'US') -or
        $null -eq $CapturedIndex.reports
    ) {
        throw 'Observation LKG captured report index is invalid'
    }
    $LocalReports = @($LocalIndex.reports)
    foreach ($CapturedEntry in @($CapturedIndex.reports)) {
        $Matches = @($LocalReports | Where-Object {
            [string]$_.report_type -eq [string]$CapturedEntry.report_type -and
            [string]$_.source_market_date -eq [string]$CapturedEntry.source_market_date -and
            [string]$_.applicable_trading_date -eq [string]$CapturedEntry.applicable_trading_date -and
            [string]$_.metadata -eq [string]$CapturedEntry.metadata -and
            [string]$_.metadata_sha256 -eq [string]$CapturedEntry.metadata_sha256 -and
            [string]$_.content_sha256 -eq [string]$CapturedEntry.content_sha256
        })
        if ($Matches.Count -ne 1) {
            throw 'Observation report index would clobber captured entry'
        }
    }
}

function Publish-ReportsV2 {
    param([string]$Root)
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    $Resolved = (Resolve-Path -LiteralPath $Root).Path
    if (((Get-Item -LiteralPath $Resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Report v2 publish root must not be a reparse point'
    }
    function Assert-V2Path {
        param([string]$Path)
        return Assert-PathWithinRoot -Path $Path -Root $Resolved
    }
    function Publish-VerifiedReportObject {
        param(
            [object]$Pointer,
            [string]$Kind,
            [string]$SourceManifest,
            [string]$SourceManifestHash,
            [string]$SourceMarketDate,
            [string]$ApplicableTradingDate
        )
        $ExpectedKeys = @(
            'object', 'sha256', 'content_sha256', 'schema_version',
            'generator_version', 'code_commit_sha'
        )
        if (
            $null -eq $Pointer -or
            (@($Pointer.PSObject.Properties.Name | Sort-Object) -join '|') -ne
                (@($ExpectedKeys | Sort-Object) -join '|')
        ) {
            throw "Invalid $Kind report object pointer"
        }
        $ObjectRelative = [string]$Pointer.object
        $Prefix = if ($Kind -eq 'canonical') {
            'objects/canonical/'
        } else {
            'objects/regression/'
        }
        if (
            $ObjectRelative -notmatch "^$([regex]::Escape($Prefix))[0-9a-f]{64}\.json$"
        ) {
            throw "Invalid $Kind report object path"
        }
        $ObjectHash = ([string]$Pointer.sha256).ToLowerInvariant()
        if (
            $ObjectHash -notmatch '^[0-9a-f]{64}$' -or
            $ObjectRelative -ne "$Prefix$ObjectHash.json" -or
            [int]$Pointer.schema_version -ne 1 -or
            [string]$Pointer.content_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$Pointer.generator_version -notmatch '^.{1,100}$' -or
            [string]$Pointer.code_commit_sha -notmatch '^[0-9a-f]{40}$'
        ) {
            throw "Invalid $Kind report object pointer"
        }
        $ObjectPath = Assert-V2Path (Join-Path $Resolved $ObjectRelative)
        $ObjectFile = Get-Item -LiteralPath $ObjectPath
        $MaxBytes = if ($Kind -eq 'canonical') { 5MB } else { 2MB }
        if ($ObjectFile.Length -le 0 -or $ObjectFile.Length -gt $MaxBytes) {
            throw "Invalid $Kind report object size"
        }
        if (
            (Get-FileHash -LiteralPath $ObjectPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
                $ObjectHash
        ) {
            throw "$Kind report object hash mismatch"
        }
        try {
            $ObjectDocument = Get-Content -LiteralPath $ObjectPath -Raw -Encoding utf8 |
                ConvertFrom-Json
        } catch {
            throw "Invalid $Kind report object JSON"
        }
        $Identity = $ObjectDocument.identity
        $ExpectedKind = if ($Kind -eq 'canonical') {
            'absorb-professional-post-close-report'
        } else {
            'absorb-regression-research-artifact'
        }
        if (
            [int]$ObjectDocument.schema_version -ne 1 -or
            [string]$ObjectDocument.kind -ne $ExpectedKind -or
            $null -eq $Identity -or
            [string]$Identity.market -notin @('TW', 'US') -or
            [string]$Identity.source_market_date -ne $SourceMarketDate -or
            [string]$Identity.applicable_trading_date -ne $ApplicableTradingDate -or
            [string]$Identity.source_manifest -ne $SourceManifest -or
            [string]$Identity.source_manifest_sha256 -ne $SourceManifestHash -or
            [string]$Identity.content_sha256 -ne [string]$Pointer.content_sha256 -or
            [string]$Identity.generator_version -ne [string]$Pointer.generator_version -or
            [string]$Identity.code_commit_sha -ne [string]$Pointer.code_commit_sha
        ) {
            throw "Report object schema or lineage mismatch: $Kind"
        }
        Invoke-GcloudCopy `
            $ObjectPath `
            "gs://$Bucket/reports/v2/$ObjectRelative" `
            -NoClobber | Out-Null
        Assert-GcloudFileMatches `
            -Gcloud $Gcloud `
            -LocalPath $ObjectPath `
            -Uri "gs://$Bucket/reports/v2/$ObjectRelative"
    }

    $Uploaded = New-Object System.Collections.Generic.List[string]
    $ReportMarkets = if ($Market) { @($Market) } else { @('TW', 'US') }
    foreach ($ReportMarket in $ReportMarkets) {
        $IndexPathCandidate = Join-Path $Resolved "index-$ReportMarket.json"
        if (-not (Test-Path -LiteralPath $IndexPathCandidate -PathType Leaf)) { continue }
        $IndexPath = Assert-V2Path $IndexPathCandidate
        $IndexFile = Get-Item -LiteralPath $IndexPath
        if ($IndexFile.Length -le 0 -or $IndexFile.Length -gt 1MB) { throw "Invalid report v2 index size for $ReportMarket" }
        $IndexPointerHash = (Get-FileHash -LiteralPath $IndexPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $Index = Get-Content -LiteralPath $IndexPath -Raw -Encoding utf8 | ConvertFrom-Json
        if ($Index.schema_version -ne 2 -or $Index.kind -notin @('absorb-report-index', 'stock-papi-report-index') -or $Index.market -ne $ReportMarket) {
            throw "Invalid report v2 index for $ReportMarket"
        }
        $Reports = @($Index.reports)
        if ($Reports.Count -gt 180) { throw "Report v2 index contains too many entries for $ReportMarket" }
        $Seen = @{}
        foreach ($Entry in $Reports) {
            $Type = [string]$Entry.report_type
            if ($Type -notin @('post_close', 'pre_market', 'weekly_model')) { throw 'Invalid report v2 type' }
            $LogicalKey = "$Type|$($Entry.source_market_date)|$($Entry.applicable_trading_date)"
            if ($Seen.ContainsKey($LogicalKey)) { throw 'Duplicate report v2 logical key' }
            $Seen[$LogicalKey] = $true
            $MetadataRelative = [string]$Entry.metadata
            if ($MetadataRelative -notmatch '^metadata/[0-9a-f]{64}\.json$') { throw 'Invalid report v2 metadata path' }
            $MetadataPath = Assert-V2Path (Join-Path $Resolved $MetadataRelative)
            $MetadataHash = (Get-FileHash -LiteralPath $MetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($MetadataHash -ne [string]$Entry.metadata_sha256) { throw 'Report v2 metadata hash mismatch' }
            $Metadata = Get-Content -LiteralPath $MetadataPath -Raw -Encoding utf8 | ConvertFrom-Json
            if (
                $Metadata.schema_version -ne 2 -or $Metadata.kind -notin @('absorb-report', 'stock-papi-report') -or
                $Metadata.market -ne $ReportMarket -or [string]$Metadata.report_type -ne $Type -or
                [string]$Metadata.source_market_date -ne [string]$Entry.source_market_date -or
                [string]$Metadata.applicable_trading_date -ne [string]$Entry.applicable_trading_date
            ) { throw 'Report v2 metadata identity mismatch' }
            $SourceManifest = [string]$Metadata.source_manifest
            if ($SourceManifest -notmatch '^quant/v1/manifests/(?:TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$') {
                throw 'Invalid report v2 source manifest path'
            }
            $SourceRelative = $SourceManifest.Substring('quant/v1/'.Length)
            $SourcePath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $SourceRelative)
            if ((Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$Metadata.source_manifest_sha256) {
                throw 'Report v2 source manifest hash mismatch'
            }
            $CanonicalPointer = $Metadata.professional_report
            $RegressionPointer = $Metadata.regression_research
            if (
                $ObservationOnly -and
                $Type -eq 'post_close' -and
                [string]$Metadata.product_mode -eq 'observation' -and
                $null -eq $CanonicalPointer
            ) {
                throw 'Observation post-close canonical report object is missing'
            }
            if ($null -ne $CanonicalPointer) {
                if ($Type -ne 'post_close') {
                    throw 'Canonical report object is only allowed for post-close'
                }
                Publish-VerifiedReportObject `
                    -Pointer $CanonicalPointer `
                    -Kind 'canonical' `
                    -SourceManifest $SourceManifest `
                    -SourceManifestHash ([string]$Metadata.source_manifest_sha256) `
                    -SourceMarketDate ([string]$Entry.source_market_date) `
                    -ApplicableTradingDate ([string]$Entry.applicable_trading_date)
            }
            if ($null -ne $RegressionPointer) {
                if ($Type -ne 'post_close') {
                    throw 'Regression report object is only allowed for post-close'
                }
                Publish-VerifiedReportObject `
                    -Pointer $RegressionPointer `
                    -Kind 'regression' `
                    -SourceManifest $SourceManifest `
                    -SourceManifestHash ([string]$Metadata.source_manifest_sha256) `
                    -SourceMarketDate ([string]$Entry.source_market_date) `
                    -ApplicableTradingDate ([string]$Entry.applicable_trading_date)
            }
            $ContentHash = [string]$Metadata.content_sha256
            if ($ContentHash -notmatch '^[0-9a-f]{64}$' -or $ContentHash -ne [string]$Entry.content_sha256) {
                throw 'Report v2 content hash mismatch'
            }
            $HasPdf = $null -ne $Metadata.pdf_path
            if ($Type -eq 'pre_market' -and $HasPdf) { throw 'Pre-market report v2 must not contain PDF' }
            if ($HasPdf) {
                $PdfRelative = [string]$Metadata.pdf_path
                if ($PdfRelative -notmatch '^objects/[0-9a-f]{64}\.pdf$' -or [long]$Metadata.pdf_size -le 0 -or [long]$Metadata.pdf_size -gt 15MB) {
                    throw 'Invalid report v2 PDF metadata'
                }
                $PdfPath = Assert-V2Path (Join-Path $Resolved $PdfRelative)
                $Pdf = Get-Item -LiteralPath $PdfPath
                if ($Pdf.Length -ne [long]$Metadata.pdf_size) { throw 'Report v2 PDF size mismatch' }
                $PdfHash = (Get-FileHash -LiteralPath $PdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($PdfHash -ne [string]$Metadata.pdf_sha256 -or $PdfRelative -ne "objects/$PdfHash.pdf") {
                    throw 'Report v2 PDF hash mismatch'
                }
                Invoke-GcloudCopy $PdfPath "gs://$Bucket/reports/v2/$PdfRelative" -NoClobber
                Assert-GcloudFileMatches `
                    -Gcloud $Gcloud `
                    -LocalPath $PdfPath `
                    -Uri "gs://$Bucket/reports/v2/$PdfRelative"
            }
            Invoke-GcloudCopy $MetadataPath "gs://$Bucket/reports/v2/$MetadataRelative" -NoClobber
            Assert-GcloudFileMatches `
                -Gcloud $Gcloud `
                -LocalPath $MetadataPath `
                -Uri "gs://$Bucket/reports/v2/$MetadataRelative"
        }

        # All immutable objects and metadata are verified and uploaded before mutable pointers.
        if ($ReportMarket -eq 'TW') {
            Assert-ObservationReportIndexPreservesLkg -LocalIndex $Index
        }
        if ((Get-FileHash -LiteralPath $IndexPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $IndexPointerHash) {
            throw "Report v2 index changed during validation for $ReportMarket"
        }
        $IndexDestination = if ($ReportMarket -eq 'TW') {
            "gs://$Bucket/reports/v2/index-TW.json"
        } else {
            "gs://$Bucket/reports/v2/index-$ReportMarket.json"
        }
        Set-GcloudMutablePointer `
            -Source $IndexPath `
            -Destination $IndexDestination `
            -ExpectedSha256 $IndexPointerHash | Out-Null
        $RemoteIndex = Get-GcloudJson `
            -Gcloud $Gcloud `
            -Uri $IndexDestination
        if ($RemoteIndex.schema_version -ne 2 -or $RemoteIndex.market -ne $ReportMarket -or @($RemoteIndex.reports).Count -ne $Reports.Count) {
            throw "Report v2 remote index read-back mismatch for $ReportMarket"
        }
        $ReportV2Types = @(
            Get-ObservationReportV2Types -ObservationOnly:$ObservationOnly
        )
        foreach ($Type in $ReportV2Types) {
            $LatestName = "latest-$ReportMarket-$Type.json"
            $LatestCandidate = Join-Path $Resolved $LatestName
            if (-not (Test-Path -LiteralPath $LatestCandidate -PathType Leaf)) { continue }
            $LatestPath = Assert-V2Path $LatestCandidate
            $LatestPointerHash = (Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
            if ($Latest.schema_version -ne 2 -or $Latest.kind -notin @('absorb-report', 'stock-papi-report') -or $Latest.market -ne $ReportMarket -or [string]$Latest.report_type -ne $Type) {
                throw "Invalid report v2 latest pointer for $ReportMarket"
            }
            if (
                $ObservationOnly -and
                (
                    [string]$Latest.product_mode -ne 'observation' -or
                    (
                        $null -ne $Latest.model_versions -and
                        @($Latest.model_versions.PSObject.Properties).Count -ne 0
                    )
                )
            ) {
                throw 'Observation report latest pointer contains prediction state'
            }
            $Match = @($Reports | Where-Object {
                [string]$_.report_type -eq $Type -and
                [string]$_.metadata -eq [string]$Latest.metadata -and
                [string]$_.metadata_sha256 -eq [string]$Latest.metadata_sha256
            })
            if ($Match.Count -ne 1) { throw "Report v2 latest pointer is not present in index for $ReportMarket" }
            if ((Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $LatestPointerHash) {
                throw "Report v2 latest pointer changed during validation for $ReportMarket"
            }
            Set-GcloudMutablePointer `
                -Source $LatestPath `
                -Destination "gs://$Bucket/reports/v2/$LatestName" `
                -ExpectedSha256 $LatestPointerHash | Out-Null
            $RemoteLatest = Get-GcloudJson `
                -Gcloud $Gcloud `
                -Uri "gs://$Bucket/reports/v2/$LatestName"
            if (
                [string]$RemoteLatest.report_type -ne $Type -or
                [string]$RemoteLatest.metadata -ne [string]$Latest.metadata -or
                [string]$RemoteLatest.metadata_sha256 -ne [string]$Latest.metadata_sha256
            ) { throw "Report v2 remote latest read-back mismatch for $ReportMarket" }
            $Uploaded.Add("$ReportMarket-$Type") | Out-Null
        }
    }
    return $Uploaded.ToArray()
}

function Publish-DashboardV1 {
    $Root = Join-Path $DataRoot 'publish\dashboard\v1'
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return $false }
    $Resolved = (Resolve-Path -LiteralPath $Root).Path
    if (((Get-Item -LiteralPath $Resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Dashboard root must not be a reparse point' }
    $DashboardMarkets = if ($Market) { @($Market) } else { @('TW', 'US') }
    $PublishedAny = $false
    foreach ($DashboardMarket in $DashboardMarkets) {
        $LatestCandidate = Join-Path $Resolved "latest-$DashboardMarket.json"
        if (-not (Test-Path -LiteralPath $LatestCandidate -PathType Leaf)) { continue }
        $LatestPath = Assert-PathWithinRoot `
            -Path $LatestCandidate `
            -Root $Resolved
        $LatestPointerHash = (Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
        $Relative = [string]$Latest.path
        if (
            $Latest.schema_version -ne 2 -or
            $Latest.kind -ne 'absorb-observation-dashboard' -or
            $Latest.product_mode -ne 'observation' -or
            $Latest.market -ne $DashboardMarket -or $Relative -notmatch '^objects/[0-9a-f]{64}\.json$' -or
            [string]$Latest.sha256 -notmatch '^[0-9a-f]{64}$' -or [long]$Latest.size -le 0 -or [long]$Latest.size -gt 5MB
        ) { throw "Invalid dashboard latest pointer for $DashboardMarket" }
        $ObjectPath = Assert-PathWithinRoot `
            -Path (Join-Path $Resolved $Relative) `
            -Root $Resolved
        $Object = Get-Item -LiteralPath $ObjectPath
        if (($Object.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or $Object.Length -ne [long]$Latest.size) { throw "Invalid dashboard object for $DashboardMarket" }
        $Digest = (Get-FileHash -LiteralPath $ObjectPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Digest -ne [string]$Latest.sha256 -or $Relative -ne "objects/$Digest.json") { throw "Dashboard object hash mismatch for $DashboardMarket" }
        $Document = Get-Content -LiteralPath $ObjectPath -Raw -Encoding utf8 | ConvertFrom-Json
        if (
            $Document.schema_version -ne 2 -or
            $Document.kind -ne 'absorb-observation-dashboard' -or
            $Document.product_mode -ne 'observation' -or
            $Document.market -ne $DashboardMarket -or
            [string]$Document.observation_as_of -ne [string]$Latest.observation_as_of -or
            $Document.prediction_capability.mode -ne 'research' -or
            $Document.prediction_capability.probability_allowed -ne $false -or
            $Document.prediction_capability.ranking_allowed -ne $false -or
            $Document.prediction_capability.strong_action_allowed -ne $false -or
            $Document.prediction_capability.performance_endorsement_allowed -ne $false -or
            $null -eq $Document.market_observation -or
            $null -eq $Document.industry_observations -or
            $null -eq $Document.heatmap -or
            $null -eq $Document.daily_focus -or
            $null -eq $Document.stock_events -or
            $null -eq $Document.etf_observations
        ) { throw "Dashboard object schema mismatch for $DashboardMarket" }
        $SourceManifest = [string]$Document.source_manifest
        if ($SourceManifest -notmatch '^quant/v1/manifests/(?:TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$') { throw "Invalid dashboard source manifest for $DashboardMarket" }
        $SourcePath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $SourceManifest.Substring('quant/v1/'.Length))
        if ((Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$Document.source_manifest_sha256) { throw "Dashboard source manifest hash mismatch for $DashboardMarket" }
        Invoke-GcloudCopy $ObjectPath "gs://$Bucket/dashboard/v1/$Relative" -NoClobber
        Assert-GcloudFileMatches `
            -Gcloud $Gcloud `
            -LocalPath $ObjectPath `
            -Uri "gs://$Bucket/dashboard/v1/$Relative"
        if ((Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $LatestPointerHash) {
            throw "Dashboard latest pointer changed during validation for $DashboardMarket"
        }
        Set-GcloudMutablePointer `
            -Source $LatestPath `
            -Destination "gs://$Bucket/dashboard/v1/latest-$DashboardMarket.json" `
            -ExpectedSha256 $LatestPointerHash | Out-Null
        $Remote = Get-GcloudJson `
            -Gcloud $Gcloud `
            -Uri "gs://$Bucket/dashboard/v1/latest-$DashboardMarket.json"
        if (
            [string]$Remote.sha256 -ne $Digest -or
            [string]$Remote.observation_as_of -ne
            [string]$Document.observation_as_of -or
            [string]$Remote.product_mode -ne 'observation'
        ) { throw "Dashboard remote read-back mismatch for $DashboardMarket" }
        $PublishedAny = $true
    }
    return $PublishedAny
}

if ($ObservationOnly -and -not $LkgReceiptPath) {
    $CaptureText = (& (Join-Path $PSScriptRoot 'capture_observation_lkg.ps1') `
        -DataRoot $DataRoot `
        -Bucket $Bucket | Out-String).Trim()
    try {
        $Capture = $CaptureText | ConvertFrom-Json
    } catch {
        throw 'Observation LKG capture output is invalid'
    }
    $LkgReceiptPath = [string]$Capture.receipt
}
if ($LkgReceiptPath) {
    $ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
    $LkgReceiptPath = Assert-PathWithinRoot `
        -Path $LkgReceiptPath `
        -Root $ReceiptRoot
    $PointerLockPath = Join-Path $ReceiptRoot 'pointer-update.lock'
    try {
        $Global:LkgPointerLockStream = [IO.File]::Open(
            $PointerLockPath,
            [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    } catch {
        throw 'Observation LKG pointer lock is held'
    }
    $ReceiptPreflight = Get-Content `
        -LiteralPath $LkgReceiptPath `
        -Raw `
        -Encoding utf8 | ConvertFrom-Json
    if (
        $ReceiptPreflight.schema_version -ne 1 -or
        $ReceiptPreflight.kind -ne 'absorb-observation-lkg' -or
        $ReceiptPreflight.bucket -ne $Bucket -or
        -not ($ReceiptPreflight.pointers -is [array])
    ) {
        throw 'Observation LKG receipt preflight is invalid'
    }
    $PendingPointerJournalPath = "$LkgReceiptPath.pending.json"
    if (
        [IO.File]::Exists($PendingPointerJournalPath) -or
        [IO.File]::Exists("$PendingPointerJournalPath.tmp")
    ) {
        throw 'Observation LKG pending pointer journal exists'
    }
    $ExpectedPointerGenerations = @{}
    foreach ($Pointer in @($ReceiptPreflight.pointers)) {
        $PointerUri = [string]$Pointer.uri
        if (
            [string]::IsNullOrWhiteSpace($PointerUri) -or
            -not $PointerUri.StartsWith(
                "gs://$Bucket/",
                [StringComparison]::Ordinal
            ) -or
            $ExpectedPointerGenerations.ContainsKey($PointerUri)
        ) {
            throw 'Observation LKG receipt contains invalid or duplicate pointer'
        }
        $CapturedGeneration = [string]$Pointer.generation
        $AppliedGeneration = [string]$Pointer.applied_generation
        if (
            ($Pointer.exists -and $CapturedGeneration -notmatch '^\d+$') -or
            ($AppliedGeneration -and $AppliedGeneration -notmatch '^\d+$')
        ) {
            throw 'Observation LKG receipt contains invalid generation'
        }
        $ExpectedGeneration = if ($AppliedGeneration -match '^\d+$') {
            $AppliedGeneration
        } elseif ($Pointer.exists) {
            $CapturedGeneration
        } else {
            '0'
        }
        $ExpectedPointerGenerations[$PointerUri] = $ExpectedGeneration
    }
}

try {
    $InsightsUploaded = $false
    $InsightsLatestPath = Join-Path $ResolvedRoot 'latest-insights.json'
    if (-not $ObservationOnly -and (Test-Path -LiteralPath $InsightsLatestPath -PathType Leaf)) {
        $InsightsLatestPath = Assert-AllowlistedPath $InsightsLatestPath
        $InsightsPointerHash = (Get-FileHash -LiteralPath $InsightsLatestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $Insights = Get-Content -LiteralPath $InsightsLatestPath -Raw -Encoding utf8 | ConvertFrom-Json
        if ($Insights.schema_version -ne 1 -or $Insights.kind -ne 'market-insights') {
            throw 'Invalid market-insights latest pointer'
        }
        $InsightsObjectRelative = [string]$Insights.path
        if ($InsightsObjectRelative -notmatch '^objects/[0-9a-f]{64}\.json\.gz$') {
            throw 'Invalid market-insights object path'
        }
        $InsightsObjectPath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $InsightsObjectRelative)
        $InsightsObject = Get-Item -LiteralPath $InsightsObjectPath
        if ($InsightsObject.Length -ne [long]$Insights.size) { throw 'Market-insights object size mismatch' }
        if ((Get-FileHash -LiteralPath $InsightsObjectPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Insights.sha256) {
            throw 'Market-insights object hash mismatch'
        }
        if ((Get-FileHash -LiteralPath $InsightsLatestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $InsightsPointerHash) {
            throw 'Market-insights latest pointer changed during validation'
        }
        Invoke-GcloudCopy $InsightsObjectPath "gs://$Bucket/quant/v1/$InsightsObjectRelative" -NoClobber
        Set-GcloudMutablePointer `
            -Source $InsightsLatestPath `
            -Destination "gs://$Bucket/quant/v1/latest-insights.json" `
            -ExpectedSha256 $InsightsPointerHash | Out-Null
        $InsightsUploaded = $true
    }

    $UploadedMarkets = @()
    $Markets = if ($ReportV2Only) {
        @()
    } elseif ($Market) {
        @($Market)
    } elseif ($ObservationOnly) {
        @('TW')
    } else {
        @('TW', 'US')
    }
    foreach ($Market in $Markets) {
        $LatestPath = Join-Path $ResolvedRoot "latest-$Market.json"
        if (-not (Test-Path -LiteralPath $LatestPath -PathType Leaf)) { continue }
        $LatestPath = Assert-AllowlistedPath $LatestPath
        $LatestPointerHash = (Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $Latest = Get-Content -LiteralPath $LatestPath -Raw -Encoding utf8 | ConvertFrom-Json
        $LatestSchema = [int]$Latest.schema_version
        if ($LatestSchema -notin @(2, 3, 4) -or $Latest.market -ne $Market) {
            throw "Invalid latest pointer for $Market"
        }
        if ($LatestSchema -in @(3, 4) -and $Market -notin @('TW', 'US')) {
            throw 'Manifest v3/v4 is TW or US only'
        }
        $ManifestRelative = [string]$Latest.manifest
        if ($ManifestRelative -notmatch '^manifests/[A-Z]+-[0-9TZ]+-[0-9a-f]{12}\.json$') {
            throw "Invalid manifest path for $Market"
        }
        if (
            [string]$Latest.manifest_sha256 -notmatch '^[0-9a-f]{64}$' -or
            -not $ManifestRelative.EndsWith(
                "-$(([string]$Latest.manifest_sha256).Substring(0, 12)).json"
            )
        ) { throw "Invalid content-addressed manifest for $Market" }
        $ManifestPath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $ManifestRelative)
        if ((Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Latest.manifest_sha256) {
            throw "Manifest hash mismatch for $Market"
        }
        $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
        if (
            [int]$Manifest.schema_version -ne $LatestSchema -or
            $Manifest.market -ne $Market -or
            [string]$Manifest.generated_at -ne [string]$Latest.generated_at
        ) {
            throw "Invalid manifest for $Market"
        }

        $SymbolProperties = @($Manifest.symbols.PSObject.Properties)
        if ($LatestSchema -eq 2) {
            $FailureThreshold = if ($Market -eq 'TW') { 0.05 } else { 0.25 }
            if (
                -not (Test-JsonInteger $Manifest.universe_count) -or
                -not (Test-JsonInteger $Manifest.symbol_count) -or
                -not (Test-JsonInteger $Manifest.failure_count) -or
                [long]$Manifest.universe_count -lt 1 -or
                [long]$Manifest.symbol_count -ne $SymbolProperties.Count -or
                [long]$Manifest.failure_count -ne
                    ([long]$Manifest.universe_count - [long]$Manifest.symbol_count) -or
                @($Manifest.failed_symbols).Count -ne [long]$Manifest.failure_count -or
                [Math]::Abs(
                    [double]$Manifest.coverage -
                    ([double]$Manifest.symbol_count / [double]$Manifest.universe_count)
                ) -gt 1e-12 -or
                [Math]::Abs(
                    [double]$Manifest.failure_rate -
                    ([double]$Manifest.failure_count / [double]$Manifest.universe_count)
                ) -gt 1e-12 -or
                [double]$Manifest.failure_rate -ge $FailureThreshold
            ) { throw "Invalid manifest arithmetic for $Market" }
        } elseif ($LatestSchema -eq 3) {
            $ExpectedProperties = @(
                $Manifest.expected_non_price_symbols.PSObject.Properties
            )
            $OperationalFailures = @($Manifest.operational_failed_symbols)
            $CountFields = @(
                $Manifest.universe_count,
                $Manifest.observation_count,
                $Manifest.regular_price_symbol_count,
                $Manifest.expected_non_price_symbol_count,
                $Manifest.operational_failure_count,
                $Manifest.regular_price_denominator
            )
            if (
                $Manifest.PSObject.Properties['market_as_of'] -or
                [string]$Manifest.target_market_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
                [string]$Manifest.observation_as_of -ne
                    [string]$Manifest.target_market_date -or
                @($CountFields | Where-Object {
                    -not (Test-JsonInteger $_) -or [long]$_ -lt 0
                }).Count -ne 0 -or
                [long]$Manifest.universe_count -lt 1 -or
                [long]$Manifest.regular_price_denominator -lt 1 -or
                [long]$Manifest.observation_count -ne $SymbolProperties.Count -or
                [long]$Manifest.expected_non_price_symbol_count -ne
                    $ExpectedProperties.Count -or
                [long]$Manifest.operational_failure_count -ne
                    $OperationalFailures.Count -or
                [long]$Manifest.regular_price_symbol_count +
                    [long]$Manifest.expected_non_price_symbol_count -ne
                    [long]$Manifest.observation_count -or
                [long]$Manifest.observation_count +
                    [long]$Manifest.operational_failure_count -ne
                    [long]$Manifest.universe_count -or
                [long]$Manifest.regular_price_denominator -ne
                    ([long]$Manifest.universe_count -
                    [long]$Manifest.expected_non_price_symbol_count) -or
                @($OperationalFailures | Select-Object -Unique).Count -ne
                    $OperationalFailures.Count -or
                [Math]::Abs(
                    [double]$Manifest.regular_price_coverage -
                    ([double]$Manifest.regular_price_symbol_count /
                    [double]$Manifest.regular_price_denominator)
                ) -gt 1e-12 -or
                [Math]::Abs(
                    [double]$Manifest.observation_coverage -
                    ([double]$Manifest.observation_count /
                    [double]$Manifest.universe_count)
                ) -gt 1e-12 -or
                [Math]::Abs(
                    [double]$Manifest.operational_failure_rate -
                    ([double]$Manifest.operational_failure_count /
                    [double]$Manifest.universe_count)
                ) -gt 1e-12 -or
                [double]$Manifest.operational_failure_rate -ge 0.05
            ) { throw 'Invalid manifest v3 arithmetic' }
        } elseif ($LatestSchema -eq 4) {
            $ExpectedProperties = @(
                $Manifest.expected_non_price_symbols.PSObject.Properties
            )
            $OperationalFailures = @($Manifest.operational_failed_symbols)
            $UnavailableSymbols = @($Manifest.unavailable_symbols)
            $CountFields = @(
                $Manifest.active_universe_count,
                $Manifest.observation_count,
                $Manifest.regular_price_symbol_count,
                $Manifest.verified_non_price_symbol_count,
                $Manifest.unavailable_count,
                $Manifest.operational_failure_count,
                $Manifest.regular_price_denominator
            )
            if (
                $Manifest.PSObject.Properties['market_as_of'] -or
                [string]$Manifest.target_market_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or
                [string]$Manifest.observation_as_of -ne
                    [string]$Manifest.target_market_date -or
                @($CountFields | Where-Object {
                    -not (Test-JsonInteger $_) -or [long]$_ -lt 0
                }).Count -ne 0 -or
                [long]$Manifest.active_universe_count -lt 1 -or
                [long]$Manifest.regular_price_denominator -lt 1 -or
                [long]$Manifest.observation_count -ne $SymbolProperties.Count -or
                [long]$Manifest.verified_non_price_symbol_count -ne
                    $ExpectedProperties.Count -or
                [long]$Manifest.unavailable_count -ne $UnavailableSymbols.Count -or
                [long]$Manifest.operational_failure_count -ne 0 -or
                $OperationalFailures.Count -ne 0 -or
                [long]$Manifest.regular_price_symbol_count +
                    [long]$Manifest.verified_non_price_symbol_count -ne
                    [long]$Manifest.observation_count -or
                [long]$Manifest.observation_count +
                    [long]$Manifest.unavailable_count -ne
                    [long]$Manifest.active_universe_count -or
                [long]$Manifest.regular_price_denominator -ne
                    ([long]$Manifest.observation_count -
                    [long]$Manifest.verified_non_price_symbol_count) -or
                @($UnavailableSymbols | Select-Object -Unique).Count -ne
                    $UnavailableSymbols.Count -or
                [Math]::Abs(
                    [double]$Manifest.regular_price_coverage -
                    ([double]$Manifest.regular_price_symbol_count /
                    [double]$Manifest.regular_price_denominator)
                ) -gt 1e-12 -or
                [Math]::Abs(
                    [double]$Manifest.observation_coverage -
                    ([double]$Manifest.observation_count /
                    [double]$Manifest.active_universe_count)
                ) -gt 1e-12 -or
                [long]$Manifest.observation_count * 100 -le
                    ([long]$Manifest.active_universe_count * 95)
            ) { throw 'Invalid manifest v4 arithmetic' }
            $SymbolPattern = if ($Market -eq 'US') { '^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?$' } else { '^\d{4,6}$' }
            foreach ($Symbol in $UnavailableSymbols) {
                if ([string]$Symbol -notmatch $SymbolPattern) {
                    throw 'Invalid unavailable_symbols entry'
                }
            }
        }
            $SymbolPattern = if ($Market -eq 'US') { '^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?$' } else { '^\d{4,6}$' }
            $ExpectedBySymbol = @{}
            foreach ($Property in $ExpectedProperties) {
                $Symbol = [string]$Property.Name
                $Status = $Property.Value
                if (
                    $Symbol -notmatch $SymbolPattern -or
                    $ExpectedBySymbol.ContainsKey($Symbol) -or
                    [string]$Status.status -notin @(
                        'official_no_regular_trade',
                        'officially_suspended'
                    ) -or
                    [string]$Status.evidence_sha256 -notmatch '^[0-9a-f]{64}$' -or
                    [string]$Status.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or
                    [string]$Status.latest_regular_price_date -notmatch
                        '^\d{4}-\d{2}-\d{2}$'
                ) { throw 'Invalid expected_non_price_symbols entry' }
                $ExpectedBySymbol[$Symbol] = $Status
            }
            foreach ($Symbol in $OperationalFailures) {
                if (
                    $Symbol -notmatch $SymbolPattern -or
                    $ExpectedBySymbol.ContainsKey([string]$Symbol)
                ) { throw 'Invalid operational_failed_symbols entry' }
            }
            if ($LatestSchema -eq 4) {
                foreach ($Symbol in @($Manifest.unavailable_symbols)) {
                    if (
                        $Symbol -notmatch $SymbolPattern -or
                        $ExpectedBySymbol.ContainsKey([string]$Symbol) -or
                        $SymbolProperties.Name -contains [string]$Symbol
                    ) { throw 'Invalid unavailable_symbols entry' }
                }
            }

        # Upload objects only after validating every object in this manifest.
        $ValidatedObjectPaths = New-Object System.Collections.Generic.List[string]
        $ValidatedObjectRelatives = New-Object System.Collections.Generic.List[string]
        foreach ($Property in $SymbolProperties) {
            $Symbol = [string]$Property.Name
            $Entry = $Property.Value
            $ObjectRelative = [string]$Entry.path
            if ($ObjectRelative -notmatch '^objects/[0-9a-f]{64}\.json\.gz$') {
                throw "Invalid object path for $Market"
            }
            if ($ObjectRelative -ne "objects/$([string]$Entry.sha256).json.gz") {
                throw "Object path hash mismatch for $Market"
            }
            $ObjectPath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $ObjectRelative)
            $Object = Get-Item -LiteralPath $ObjectPath
            if ($Object.Length -ne [long]$Entry.size) { throw "Object size mismatch for $Market" }
            if ((Get-FileHash -LiteralPath $ObjectPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Entry.sha256) {
                throw "Object hash mismatch for $Market"
            }
            $Document = Read-VerifiedGzipJson `
                -Path $ObjectPath `
                -ExpectedSize ([long]$Entry.uncompressed_size)
            if (
                [string]$Document.market -ne $Market -or
                [string]$Document.symbol -ne $Symbol -or
                [string]$Document.as_of -ne [string]$Entry.as_of -or
                [string]$Document.model_version -ne [string]$Entry.model_version -or
                @($Document.daily).Count -eq 0
            ) { throw "Object schema mismatch for $Market" }
            $LatestDailyDate = [string]@($Document.daily)[-1].Date
            if ($LatestDailyDate.Contains('T')) {
                $LatestDailyDate = $LatestDailyDate.Split('T')[0]
            }
            if ($LatestDailyDate -ne [string]$Entry.as_of) {
                throw "Object daily date mismatch for $Market"
            }
            if ($LatestSchema -eq 2) {
                if (
                    [int]$Document.schema_version -ne 1 -or
                    [string]$Entry.as_of -ne [string]$Manifest.market_as_of -or
                    $Entry.PSObject.Properties['observation_kind'] -or
                    $Entry.PSObject.Properties['evidence_sha256'] -or
                    $Document.PSObject.Properties['trading_status_evidence']
                ) { throw "Object v2 contract mismatch for $Market" }
            } else {
                $LatestSummaryDate = [string]$Document.latest.Date
                if ($LatestSummaryDate.Contains('T')) {
                    $LatestSummaryDate = $LatestSummaryDate.Split('T')[0]
                }
                if (
                    [int]$Document.schema_version -ne 2 -or
                    $LatestSummaryDate -ne [string]$Entry.as_of -or
                    [string]$Document.target_market_date -ne
                        [string]$Manifest.target_market_date -or
                    [string]$Document.observation_as_of -ne
                        [string]$Manifest.observation_as_of -or
                    [string]$Document.latest_regular_price_date -ne
                        [string]$Entry.latest_regular_price_date -or
                    [string]$Document.latest_regular_price_date -ne
                        [string]$Entry.as_of -or
                    [string]$Document.observation_kind -ne
                        [string]$Entry.observation_kind
                ) { throw 'Object v3 observation mismatch' }
                $Expected = if ($ExpectedBySymbol.ContainsKey($Symbol)) {
                    $ExpectedBySymbol[$Symbol]
                } else { $null }
                if ($null -eq $Expected) {
                    if (
                        [string]$Entry.observation_kind -ne 'regular_price' -or
                        [string]$Entry.as_of -ne
                            [string]$Manifest.target_market_date -or
                        $null -ne $Document.trading_status_evidence -or
                        $Entry.PSObject.Properties['evidence_sha256']
                    ) { throw 'Regular price object v3 mismatch' }
                } else {
                    $Evidence = $Document.trading_status_evidence
                    if (
                        $null -eq $Evidence -or
                        [int]$Evidence.schema_version -ne 1 -or
                        [string]$Evidence.status -ne [string]$Expected.status -or
                        [string]$Evidence.market -ne 'TW' -or
                        [string]$Evidence.symbol -ne $Symbol -or
                        [string]$Evidence.target_market_date -ne
                            [string]$Manifest.target_market_date -or
                        [string]$Evidence.evidence_sha256 -notmatch
                            '^[0-9a-f]{64}$' -or
                        [string]$Evidence.evidence_sha256 -ne
                            [string]$Entry.evidence_sha256 -or
                        [string]$Evidence.evidence_sha256 -ne
                            [string]$Expected.evidence_sha256 -or
                        [string]$Entry.sha256 -ne
                            [string]$Expected.artifact_sha256 -or
                        [string]$Entry.latest_regular_price_date -ne
                            [string]$Expected.latest_regular_price_date -or
                        [string]$Entry.observation_kind -ne
                            [string]$Expected.status
                    ) { throw 'Status object evidence mismatch' }
                }
            }
            $ValidatedObjectPaths.Add($ObjectPath) | Out-Null
            $ValidatedObjectRelatives.Add($ObjectRelative) | Out-Null
        }
        # Upload objects in bounded batches
        for ($Offset = 0; $Offset -lt $ValidatedObjectPaths.Count; $Offset += $ObjectBatchSize) {
            if ($PreflightDataRoot) { break }
            $Last = [Math]::Min($Offset + $ObjectBatchSize - 1, $ValidatedObjectPaths.Count - 1)
            Invoke-GcloudCopyBatch `
                -Sources $ValidatedObjectPaths[$Offset..$Last] `
                -Destination "gs://$Bucket/quant/v1/objects/"
        }
        if (-not $PreflightDataRoot -and $ValidatedObjectPaths.Count -gt 0) {
            $SampleIndex = 0
            $ObjectRelative = $ValidatedObjectRelatives[$SampleIndex]
            Assert-GcloudFileMatches `
                -Gcloud $Gcloud `
                -LocalPath $ValidatedObjectPaths[$SampleIndex] `
                -Uri "gs://$Bucket/quant/v1/$ObjectRelative"
        }

        # Upload manifest
        if ($PreflightDataRoot) {
            $UploadedMarkets += $Market
            continue
        }
        Invoke-GcloudCopy $ManifestPath "gs://$Bucket/quant/v1/$ManifestRelative" -NoClobber
        Assert-GcloudFileMatches `
            -Gcloud $Gcloud `
            -LocalPath $ManifestPath `
            -Uri "gs://$Bucket/quant/v1/$ManifestRelative"

        # Upload latest pointer
        if ((Get-FileHash -LiteralPath $LatestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $LatestPointerHash) {
            throw "Latest pointer changed during validation for $Market"
        }
        Set-GcloudMutablePointer `
            -Source $LatestPath `
            -Destination "gs://$Bucket/quant/v1/latest-$Market.json" `
            -ExpectedSha256 $LatestPointerHash | Out-Null
        $UploadedMarkets += $Market
    }

    if ($PreflightDataRoot) {
        Write-Output "Validated quant snapshots: $($UploadedMarkets -join ',')"
        return
    }

    $ReportUploaded = $false
    $ReportUploadError = $null
    $ReportPublishRoot = Join-Path $DataRoot 'publish\reports\v1'
    if (-not $ObservationOnly -and (Test-Path -LiteralPath $ReportPublishRoot -PathType Container)) {
        try {
            $ResolvedReportRoot = (Resolve-Path -LiteralPath $ReportPublishRoot).Path
            if (((Get-Item -LiteralPath $ResolvedReportRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Report publish root must not be a reparse point'
            }
            function Assert-ReportPath {
                param([string]$Path)
                return Assert-PathWithinRoot -Path $Path -Root $ResolvedReportRoot
            }

            $ReportLatestPath = Assert-ReportPath (Join-Path $ResolvedReportRoot 'latest-TW.json')
            $ReportLatestPointerHash = (Get-FileHash -LiteralPath $ReportLatestPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $ReportLatest = Get-Content -LiteralPath $ReportLatestPath -Raw -Encoding utf8 | ConvertFrom-Json
            if (
                $ReportLatest.schema_version -ne 1 -or
                $ReportLatest.kind -ne 'daily-industry-report' -or
                $ReportLatest.market -ne 'TW' -or
                [string]$ReportLatest.report_date -notmatch '^\d{4}-\d{2}-\d{2}$'
            ) { throw 'Invalid report latest pointer' }
            $ReportMetadataRelative = [string]$ReportLatest.metadata
            if ($ReportMetadataRelative -notmatch '^metadata/[0-9a-f]{64}\.json$') {
                throw 'Invalid report metadata path'
            }
            $ReportMetadataPath = Assert-ReportPath (Join-Path $ResolvedReportRoot $ReportMetadataRelative)
            $ReportMetadataHash = (Get-FileHash -LiteralPath $ReportMetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($ReportMetadataHash -ne [string]$ReportLatest.metadata_sha256) {
                throw 'Report metadata hash mismatch'
            }
            $ReportMetadata = Get-Content -LiteralPath $ReportMetadataPath -Raw -Encoding utf8 | ConvertFrom-Json
            $ReportPdfRelative = [string]$ReportMetadata.pdf_path
            if (
                $ReportMetadata.schema_version -ne 1 -or
                $ReportMetadata.kind -ne 'daily-industry-report' -or
                $ReportMetadata.market -ne 'TW' -or
                [string]$ReportMetadata.report_date -ne [string]$ReportLatest.report_date -or
                $ReportPdfRelative -notmatch '^objects/[0-9a-f]{64}\.pdf$' -or
                [long]$ReportMetadata.pdf_size -le 0 -or
                [long]$ReportMetadata.pdf_size -gt 15MB
            ) { throw 'Invalid report metadata' }
            $ReportPdfPath = Assert-ReportPath (Join-Path $ResolvedReportRoot $ReportPdfRelative)
            $ReportPdf = Get-Item -LiteralPath $ReportPdfPath
            if ($ReportPdf.Length -ne [long]$ReportMetadata.pdf_size) { throw 'Report PDF size mismatch' }
            if ((Get-FileHash -LiteralPath $ReportPdfPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$ReportMetadata.pdf_sha256) {
                throw 'Report PDF hash mismatch'
            }
            if ($ReportPdfRelative -ne "objects/$($ReportMetadata.pdf_sha256).pdf") {
                throw 'Report PDF content address mismatch'
            }
            $SourceManifest = [string]$ReportMetadata.source_manifest
            if ($SourceManifest -notmatch '^quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json$') {
                throw 'Invalid report source manifest path'
            }
            $SourceManifestRelative = $SourceManifest.Substring('quant/v1/'.Length)
            $SourceManifestPath = Assert-AllowlistedPath (Join-Path $ResolvedRoot $SourceManifestRelative)
            if ((Get-FileHash -LiteralPath $SourceManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$ReportMetadata.source_manifest_sha256) {
                throw 'Report source manifest hash mismatch'
            }
            $ReportIndexPath = Assert-ReportPath (Join-Path $ResolvedReportRoot 'index-TW.json')
            $ReportIndexFile = Get-Item -LiteralPath $ReportIndexPath
            if ($ReportIndexFile.Length -le 0 -or $ReportIndexFile.Length -gt 1MB) { throw 'Invalid report index size' }
            $ReportIndexPointerHash = (Get-FileHash -LiteralPath $ReportIndexPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $ReportIndex = Get-Content -LiteralPath $ReportIndexPath -Raw -Encoding utf8 | ConvertFrom-Json
            if ($ReportIndex.schema_version -ne 1 -or $ReportIndex.market -ne 'TW') { throw 'Invalid report index' }
            $ReportIndexEntry = @($ReportIndex.reports | Where-Object {
                [string]$_.report_date -eq [string]$ReportLatest.report_date -and
                [string]$_.metadata -eq $ReportMetadataRelative -and
                [string]$_.metadata_sha256 -eq $ReportMetadataHash -and
                [string]$_.pdf_path -eq $ReportPdfRelative -and
                [string]$_.pdf_sha256 -eq [string]$ReportMetadata.pdf_sha256
            })
            if ($ReportIndexEntry.Count -ne 1) { throw 'Report index entry mismatch' }
            if ((Get-FileHash -LiteralPath $ReportIndexPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ReportIndexPointerHash) {
                throw 'Report index changed during validation'
            }
            if ((Get-FileHash -LiteralPath $ReportLatestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ReportLatestPointerHash) {
                throw 'Report latest pointer changed during validation'
            }

            Invoke-GcloudCopy $ReportPdfPath "gs://$Bucket/reports/v1/$ReportPdfRelative" -NoClobber
            Invoke-GcloudCopy $ReportMetadataPath "gs://$Bucket/reports/v1/$ReportMetadataRelative" -NoClobber
            Assert-GcloudFileMatches `
                -Gcloud $Gcloud `
                -LocalPath $ReportPdfPath `
                -Uri "gs://$Bucket/reports/v1/$ReportPdfRelative"
            Assert-GcloudFileMatches `
                -Gcloud $Gcloud `
                -LocalPath $ReportMetadataPath `
                -Uri "gs://$Bucket/reports/v1/$ReportMetadataRelative"
            Set-GcloudMutablePointer `
                -Source $ReportIndexPath `
                -Destination "gs://$Bucket/reports/v1/index-TW.json" `
                -ExpectedSha256 $ReportIndexPointerHash | Out-Null
            Set-GcloudMutablePointer `
                -Source $ReportLatestPath `
                -Destination "gs://$Bucket/reports/v1/latest-TW.json" `
                -ExpectedSha256 $ReportLatestPointerHash | Out-Null
            $ReportUploaded = $true
        } catch {
            $ReportUploadError = $_.Exception.Message
            Write-Warning "日報上傳失敗：$ReportUploadError"
            Send-ReportUploadFailureNotification "日報上傳失敗：$ReportUploadError"
        }
    }

    $ReportV2UploadedTypes = @()
    $ReportV2UploadError = $null
    try {
        $ReportV2UploadedTypes = @(Publish-ReportsV2 (Join-Path $DataRoot 'publish\reports\v2'))
    } catch {
        $ReportV2UploadError = $_.Exception.Message
        Write-Warning "報告 v2 上傳失敗：$ReportV2UploadError"
        Send-ReportUploadFailureNotification "報告 v2 上傳失敗：$ReportV2UploadError"
    }

    $DashboardUploaded = $false
    $DashboardUploadError = $null
    if (-not $ReportV2Only) {
        try {
            $DashboardUploaded = Publish-DashboardV1
        } catch {
            $DashboardUploadError = $_.Exception.Message
            Write-Warning "Dashboard 上傳失敗：$DashboardUploadError"
            Send-ReportUploadFailureNotification "Dashboard 上傳失敗：$DashboardUploadError"
        }
    }

    $Status = @{
        uploaded_at = [DateTimeOffset]::Now.ToString('o')
        markets = $UploadedMarkets
        market_insights = $InsightsUploaded
        report_uploaded = $ReportUploaded
        report_error = $ReportUploadError
        report_v2_uploaded_types = $ReportV2UploadedTypes
        report_v2_error = $ReportV2UploadError
        dashboard_uploaded = $DashboardUploaded
        dashboard_error = $DashboardUploadError
        pointer_updates = $Global:PointerUpdates.ToArray()
        lkg_receipt = $LkgReceiptPath
        bucket = $Bucket
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath (Join-Path $DataRoot 'logs\upload-status.json') -Value $Status -Encoding utf8
    if ($RequireReportV2 -and ($ReportV2UploadError -or $ReportV2UploadedTypes.Count -eq 0)) {
        throw 'Required report v2 upload or remote verification failed'
    }
    if ($RequireDashboard -and ($DashboardUploadError -or -not $DashboardUploaded)) {
        throw 'Required dashboard upload or remote verification failed'
    }
    Update-ObservationLkgReceipt
    $ReceiptUpdated = $true
    Release-LkgPointerLock
    Write-Output "Uploaded quant snapshots: $($UploadedMarkets -join ',')"

} catch {
    $OriginalError = $_
    try {
        if (
            -not $ReceiptUpdated -and
            $LkgReceiptPath -and
            $Global:PointerUpdates.Count -gt 0
        ) {
            if (
                $PendingPointerJournalPath -and
                [IO.File]::Exists($PendingPointerJournalPath)
            ) {
                throw 'Upload failed after pointer mutation; pending pointer journal requires reconciliation'
            }
            try {
                Update-ObservationLkgReceipt
                $ReceiptUpdated = $true
            } catch {
                throw 'Upload failed after pointer mutation and LKG receipt update failed'
            }
        }
        throw $OriginalError
    } finally {
        Release-LkgPointerLock
    }
}
