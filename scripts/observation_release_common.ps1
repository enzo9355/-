. (Join-Path $PSScriptRoot 'native_process.ps1')
Import-Module Microsoft.PowerShell.Utility -MaximumVersion 5.1 -ErrorAction Stop

function Read-StrictUtf8JsonFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [ValidateRange(1, 10485760)][long]$MaximumBytes = 1MB
    )

    if (-not [IO.File]::Exists($Path)) {
        throw 'JSON file does not exist'
    }
    $File = Get-Item -LiteralPath $Path -Force
    if (
        $File.PSIsContainer -or
        $File.Length -le 0 -or
        $File.Length -gt $MaximumBytes -or
        ($File.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw 'JSON file size or attributes are invalid'
    }
    try {
        $Utf8 = [Text.UTF8Encoding]::new($false, $true)
        $Text = [IO.File]::ReadAllText($File.FullName, $Utf8)
    } catch {
        throw 'JSON file is not valid UTF-8'
    }
    try {
        return $Text | ConvertFrom-Json
    } catch {
        throw 'JSON file is invalid'
    }
}

function Get-ObservationReportV2Types {
    param([switch]$ObservationOnly)

    if ($ObservationOnly) {
        return @('post_close', 'pre_market')
    }
    return @('post_close', 'pre_market', 'weekly_model')
}

function Assert-PathWithinRoot {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root,
        [hashtable]$VerifiedDirs
    )

    # Use only .NET path APIs. PowerShell provider reads can be suppressed by
    # an outer -WhatIf even though this guard is intentionally read-only.
    $ResolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    )
    $Resolved = [IO.Path]::GetFullPath($Path)
    if (
        -not [IO.Directory]::Exists($ResolvedRoot) -or
        (-not [IO.File]::Exists($Resolved) -and
            -not [IO.Directory]::Exists($Resolved))
    ) {
        throw 'Release path does not exist'
    }
    $RootPrefix = $ResolvedRoot + [IO.Path]::DirectorySeparatorChar
    if (
        -not $Resolved.Equals(
            $ResolvedRoot,
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        -not $Resolved.StartsWith(
            $RootPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw 'Release path escaped allowlisted root'
    }
    if (
        ([IO.File]::GetAttributes($ResolvedRoot) -band
            [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw 'Release root contains a reparse point'
    }

    if (
        ([IO.File]::GetAttributes($Resolved) -band
            [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw 'Release path contains a reparse point'
    }
    $CurrentPath = if ([IO.Directory]::Exists($Resolved)) {
        $Resolved
    } else {
        [IO.Path]::GetDirectoryName($Resolved)
    }
    while (
        $null -ne $CurrentPath -and
        -not $CurrentPath.Equals(
            $ResolvedRoot,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        if (
            $null -ne $VerifiedDirs -and
            $VerifiedDirs.ContainsKey($CurrentPath)
        ) {
            break
        }
        if (
            ([IO.File]::GetAttributes($CurrentPath) -band
                [IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw 'Release path contains a reparse point'
        }
        $Parent = [IO.Directory]::GetParent($CurrentPath)
        $CurrentPath = if ($null -eq $Parent) { $null } else { $Parent.FullName }
    }
    if ($null -eq $CurrentPath) {
        throw 'Release path escaped allowlisted root'
    }

    if ($null -ne $VerifiedDirs) {
        $CurrentPath = if ([IO.Directory]::Exists($Resolved)) {
            $Resolved
        } else {
            [IO.Path]::GetDirectoryName($Resolved)
        }
        while (
            $null -ne $CurrentPath -and
            -not $CurrentPath.Equals(
                $ResolvedRoot,
                [StringComparison]::OrdinalIgnoreCase
            )
        ) {
            if ($VerifiedDirs.ContainsKey($CurrentPath)) { break }
            $VerifiedDirs[$CurrentPath] = $true
            $Parent = [IO.Directory]::GetParent($CurrentPath)
            $CurrentPath = if ($null -eq $Parent) { $null } else { $Parent.FullName }
        }
        if ($null -eq $CurrentPath) {
            throw 'Release path escaped allowlisted root'
        }
    }
    return $Resolved
}

function Invoke-GcloudCaptured {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure
    )

    $PreviousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $null
        return Invoke-NativeProcessCaptured `
            -FilePath $Gcloud `
            -Arguments $Arguments `
            -AllowFailure:$AllowFailure
    } finally {
        $env:PYTHONPATH = $PreviousPythonPath
    }
}

function Get-GcloudObjectState {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string]$Uri
    )

    $PreviousWhatIfPreference = $WhatIfPreference
    try {
        # Object metadata is a read-only preflight and must remain observable
        # when the caller uses WhatIf to prove rollback readiness.
        $WhatIfPreference = $false
        $Result = Invoke-GcloudCaptured -Gcloud $Gcloud -AllowFailure -Arguments @(
            'storage', 'objects', 'describe', $Uri, '--format=json'
        )
    } finally {
        $WhatIfPreference = $PreviousWhatIfPreference
    }
    if ($Result.exit_code -ne 0) {
        if ($Result.text -match '(?i)(not found|no urls matched|404)') {
            return [pscustomobject]@{
                exists = $false
                generation = $null
                uri = $Uri
            }
        }
        throw "Unable to inspect GCS object state: $Uri"
    }
    try {
        $Metadata = $Result.text | ConvertFrom-Json
    } catch {
        throw "Invalid GCS object metadata: $Uri"
    }
    $Generation = [string]$Metadata.generation
    if ($Generation -notmatch '^\d+$') {
        throw "GCS object generation is invalid: $Uri"
    }
    return [pscustomobject]@{
        exists = $true
        generation = $Generation
        uri = $Uri
    }
}

function Get-GcloudJson {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string]$Uri,
        [ValidateRange(1, 10485760)][long]$MaximumBytes = 1MB
    )

    $Temporary = Join-Path (
        [IO.Path]::GetTempPath()
    ) ('absorb-gcloud-json-' + [Guid]::NewGuid().ToString('N'))
    try {
        Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
            'storage', 'cp', '--quiet', $Uri, $Temporary
        ) | Out-Null
        try {
            return Read-StrictUtf8JsonFile `
                -Path $Temporary `
                -MaximumBytes $MaximumBytes
        } catch {
            throw "GCS JSON read-back is invalid: $Uri"
        }
    } finally {
        if ([IO.File]::Exists($Temporary)) {
            [IO.File]::Delete($Temporary)
        }
    }
}

function Invoke-GcloudConditionalCopy {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Destination,
        [string]$ExpectedGeneration,
        [string]$ExpectedSourceSha256,
        [string]$PendingJournalPath,
        [switch]$SkipIfMatches
    )

    $Before = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Destination
    $ActualGeneration = if ($Before.exists) {
        [string]$Before.generation
    } else {
        '0'
    }
    if (
        $ExpectedGeneration -and
        $ExpectedGeneration -ne $ActualGeneration
    ) {
        throw "Conditional GCS pointer generation mismatch: $Destination"
    }
    if ($SkipIfMatches -and $Before.exists) {
        try {
            Assert-GcloudFileMatches `
                -Gcloud $Gcloud `
                -LocalPath $Source `
                -Uri $Destination
            return [ordered]@{
                uri = $Destination
                before_exists = $true
                before_generation = $ActualGeneration
                after_generation = $ActualGeneration
                changed = $false
            }
        } catch {
            if (
                $_.Exception.Message -notlike
                'GCS read-back hash or size mismatch:*'
            ) {
                throw
            }
            # A verified hash mismatch means a conditional update is required.
        }
    }
    if (
        $ExpectedSourceSha256 -and
        (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $ExpectedSourceSha256
    ) {
        throw "GCS pointer source changed after validation: $Destination"
    }
    if ($PendingJournalPath) {
        Write-GcloudPendingPointerJournal `
            -Path $PendingJournalPath `
            -Entry ([ordered]@{
                uri = $Destination
                source = $Source
                expected_generation = $ActualGeneration
                source_sha256 = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
                created_at = [DateTimeOffset]::UtcNow.ToString('o')
            })
    }
    Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
        'storage', 'cp', '--quiet',
        "--if-generation-match=$ActualGeneration",
        $Source,
        $Destination
    ) | Out-Null
    $After = $null
    $LastVerificationError = $null
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
            $Candidate = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Destination
            if (
                -not $Candidate.exists -or
                [string]$Candidate.generation -notmatch '^\d+$' -or
                ($Before.exists -and $Candidate.generation -eq $Before.generation)
            ) {
                throw "Conditional GCS pointer update was not applied: $Destination"
            }
            $After = $Candidate
            break
        } catch {
            $LastVerificationError = $_.Exception.Message
            if ($Attempt -eq 3) {
                throw "Conditional GCS pointer update could not be verified: $Destination ($LastVerificationError)"
            }
            Start-Sleep -Milliseconds (250 * $Attempt)
        }
    }
    if ($null -eq $After) {
        throw "Conditional GCS pointer update could not be verified: $Destination"
    }
    Assert-GcloudFileMatches `
        -Gcloud $Gcloud `
        -LocalPath $Source `
        -Uri "$Destination#$([string]$After.generation)"
    return [ordered]@{
        uri = $Destination
        before_exists = [bool]$Before.exists
        before_generation = $ActualGeneration
        after_generation = [string]$After.generation
        changed = $true
    }
}

function New-VerifiedPointerSnapshot {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$StagingRunPath,
        [Parameter(Mandatory)][string]$ExpectedSha256
    )

    if ($ExpectedSha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'Mutable pointer validation hash is invalid'
    }
    if (-not [IO.File]::Exists($Source)) {
        throw 'Mutable pointer source does not exist'
    }
    $SourceAttributes = [IO.File]::GetAttributes($Source)
    if (($SourceAttributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Mutable pointer source must not be a reparse point'
    }
    if (-not [IO.Directory]::Exists($StagingRunPath)) {
        throw 'Mutable pointer staging path does not exist'
    }
    if (
        ([IO.File]::GetAttributes($StagingRunPath) -band
            [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw 'Mutable pointer staging path must not be a reparse point'
    }
    $Snapshot = Join-Path `
        $StagingRunPath `
        ('pointer-' + [Guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::Copy($Source, $Snapshot, $false)
    try {
        $SnapshotAttributes = [IO.File]::GetAttributes($Snapshot)
        if (($SnapshotAttributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Mutable pointer snapshot must not be a reparse point'
        }
        $SnapshotHash = (Get-FileHash -LiteralPath $Snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($SnapshotHash -ne $ExpectedSha256) {
            throw 'Mutable pointer source changed during snapshot'
        }
        [IO.File]::SetAttributes(
            $Snapshot,
            $SnapshotAttributes -bor [IO.FileAttributes]::ReadOnly
        )
        return $Snapshot
    } catch {
        if ([IO.File]::Exists($Snapshot)) {
            [IO.File]::SetAttributes($Snapshot, [IO.FileAttributes]::Normal)
            [IO.File]::Delete($Snapshot)
        }
        throw
    }
}

function Write-GcloudPendingPointerJournal {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][object]$Entry
    )

    $Entries = New-Object System.Collections.Generic.List[object]
    if ([IO.File]::Exists($Path)) {
        foreach ($ExistingEntry in @(Read-GcloudPendingPointerJournal -Path $Path)) {
            $Entries.Add($ExistingEntry) | Out-Null
        }
    }
    Assert-GcloudPendingPointerJournalEntry -Entry $Entry
    $Entries.Add([pscustomobject]$Entry) | Out-Null
    $Temporary = "$Path.tmp"
    if ([IO.File]::Exists($Temporary)) {
        throw 'Pending pointer journal temporary file exists'
    }
    try {
        [IO.File]::WriteAllText(
            $Temporary,
            (ConvertTo-Json -InputObject $Entries.ToArray() -Depth 8),
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $Temporary -Destination $Path -Force
    } catch {
        if ([IO.File]::Exists($Temporary)) {
            [IO.File]::Delete($Temporary)
        }
        throw
    }
}

function Assert-GcloudPendingPointerJournalEntry {
    param([Parameter(Mandatory)][object]$Entry)

    if (
        $null -eq $Entry -or
        [string]$Entry.uri -notmatch '^gs://[^/]+/.+$' -or
        [string]::IsNullOrWhiteSpace([string]$Entry.source) -or
        [string]$Entry.expected_generation -notmatch '^\d+$' -or
        [string]$Entry.source_sha256 -notmatch '^[0-9a-f]{64}$'
    ) {
        throw 'Pending pointer journal entry is invalid'
    }
    try {
        [DateTimeOffset]::Parse([string]$Entry.created_at) | Out-Null
    } catch {
        throw 'Pending pointer journal entry is invalid'
    }
}

function Read-GcloudPendingPointerJournal {
    param([Parameter(Mandatory)][string]$Path)

    try {
        $Document = Read-StrictUtf8JsonFile -Path $Path -MaximumBytes 1MB
    } catch {
        throw 'Pending pointer journal is invalid'
    }
    $Entries = New-Object System.Collections.Generic.List[object]
    foreach ($Item in @($Document)) {
        $ValueProperty = $Item.PSObject.Properties['value']
        $CountProperty = $Item.PSObject.Properties['Count']
        if ($null -ne $ValueProperty -or $null -ne $CountProperty) {
            if (
                $null -eq $ValueProperty -or
                $null -eq $CountProperty -or
                @($Item.PSObject.Properties).Count -ne 2 -or
                [int]$Item.Count -ne @($Item.value).Count
            ) {
                throw 'Pending pointer journal legacy wrapper is invalid'
            }
            foreach ($LegacyEntry in @($Item.value)) {
                Assert-GcloudPendingPointerJournalEntry -Entry $LegacyEntry
                $Entries.Add($LegacyEntry) | Out-Null
            }
            continue
        }
        Assert-GcloudPendingPointerJournalEntry -Entry $Item
        $Entries.Add($Item) | Out-Null
    }
    if ($Entries.Count -eq 0) {
        throw 'Pending pointer journal is empty'
    }
    $Seen = @{}
    foreach ($Entry in $Entries) {
        $Uri = [string]$Entry.uri
        if ($Seen.ContainsKey($Uri)) {
            throw 'Pending pointer journal contains duplicate pointers'
        }
        $Seen[$Uri] = $true
    }
    return $Entries.ToArray()
}

function Get-VerifiedSuccessorPointerEvidence {
    param(
        [Parameter(Mandatory)][object]$Entry,
        [Parameter(Mandatory)][string[]]$SuccessorReceiptPaths,
        [Parameter(Mandatory)][string]$ReceiptRoot,
        [Parameter(Mandatory)][string]$Bucket
    )

    Assert-GcloudPendingPointerJournalEntry -Entry $Entry
    $Source = Get-Item -LiteralPath ([string]$Entry.source) -Force
    if (
        $Source.PSIsContainer -or
        (Get-FileHash -LiteralPath $Source.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            [string]$Entry.source_sha256
    ) {
        throw 'Pending pointer journal source hash mismatch'
    }
    $CreatedAt = [DateTimeOffset]::Parse([string]$Entry.created_at)
    $Candidates = New-Object System.Collections.Generic.List[object]
    foreach ($ReceiptPath in $SuccessorReceiptPaths) {
        $ResolvedReceipt = Assert-PathWithinRoot `
            -Path $ReceiptPath `
            -Root $ReceiptRoot
        $Receipt = Read-StrictUtf8JsonFile `
            -Path $ResolvedReceipt `
            -MaximumBytes 1MB
        if (
            $Receipt.schema_version -ne 1 -or
            $Receipt.kind -ne 'absorb-observation-lkg' -or
            $Receipt.bucket -ne $Bucket -or
            -not ($Receipt.pointers -is [array])
        ) {
            throw 'Successor Observation LKG receipt is invalid'
        }
        $CapturedAt = [DateTimeOffset]::Parse([string]$Receipt.captured_at)
        if ($CapturedAt -le $CreatedAt) { continue }
        $Matches = @($Receipt.pointers | Where-Object {
            [string]$_.uri -eq [string]$Entry.uri -and
            $_.exists -eq $true -and
            [string]$_.generation -match '^\d+$' -and
            [string]$_.generation -ne [string]$Entry.expected_generation -and
            [string]$_.previous_sha256 -eq [string]$Entry.source_sha256
        })
        foreach ($Pointer in $Matches) {
            $PreviousFile = [string]$Pointer.previous_file
            if ($PreviousFile -notmatch '^[^\\/:*?"<>|]+$') {
                throw 'Successor pointer evidence path is invalid'
            }
            $CaptureRoot = Split-Path -Parent $ResolvedReceipt
            $PreviousPath = Assert-PathWithinRoot `
                -Path (Join-Path $CaptureRoot $PreviousFile) `
                -Root $CaptureRoot
            $Previous = Get-Item -LiteralPath $PreviousPath -Force
            $PreviousHash = (
                Get-FileHash -LiteralPath $PreviousPath -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            if (
                $Previous.PSIsContainer -or
                $Previous.Length -ne [long]$Pointer.previous_size -or
                $PreviousHash -ne [string]$Pointer.previous_sha256 -or
                $PreviousHash -ne [string]$Entry.source_sha256
            ) {
                throw 'Successor pointer evidence hash mismatch'
            }
            $Candidates.Add([pscustomobject]@{
                generation = [string]$Pointer.generation
                sha256 = $PreviousHash
                receipt = $ResolvedReceipt
                evidence_file = $PreviousPath
                captured_at = $CapturedAt.ToString('o')
            }) | Out-Null
        }
    }
    if ($Candidates.Count -ne 1) {
        throw 'Successor pointer evidence is unavailable or ambiguous'
    }
    return $Candidates[0]
}

function Assert-ObservationLkgPointerCapture {
    param(
        [Parameter(Mandatory)][object]$Pointer,
        [Parameter(Mandatory)][string]$CaptureRoot,
        [Parameter(Mandatory)][string]$Bucket
    )

    if (
        [string]$Pointer.uri -notlike "gs://$Bucket/*" -or
        ([string]$Pointer.applied_generation -and
            (
                [string]$Pointer.applied_generation -notmatch '^\d+$' -or
                [string]$Pointer.applied_generation -eq '0'
            ))
    ) {
        throw 'Observation LKG pointer capture is invalid'
    }
    if (-not $Pointer.exists) {
        if (
            $null -ne $Pointer.generation -or
            $null -ne $Pointer.previous_file -or
            $null -ne $Pointer.previous_sha256 -or
            [long]$Pointer.previous_size -ne 0
        ) {
            throw 'Observation LKG absent pointer capture is invalid'
        }
        return
    }
    $PreviousFile = [string]$Pointer.previous_file
    if (
        [string]$Pointer.generation -notmatch '^\d+$' -or
        $PreviousFile -notmatch '^[^\\/:*?"<>|]+$' -or
        [string]$Pointer.previous_sha256 -notmatch '^[0-9a-f]{64}$' -or
        [long]$Pointer.previous_size -le 0 -or
        [long]$Pointer.previous_size -gt 1MB
    ) {
        throw 'Observation LKG pointer capture is invalid'
    }
    $PreviousPath = Assert-PathWithinRoot `
        -Path (Join-Path $CaptureRoot $PreviousFile) `
        -Root $CaptureRoot
    $Previous = Get-Item -LiteralPath $PreviousPath -Force
    $PreviousHash = (
        Get-FileHash -LiteralPath $PreviousPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $Previous.PSIsContainer -or
        $Previous.Length -ne [long]$Pointer.previous_size -or
        $PreviousHash -ne [string]$Pointer.previous_sha256
    ) {
        throw 'Observation LKG pointer capture hash mismatch'
    }
}

function Remove-GcloudPendingPointerJournal {
    param([string]$Path)
    if ($Path -and [IO.File]::Exists($Path)) {
        [IO.File]::Delete($Path)
    }
}

function Invoke-GcloudConditionalDelete {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$ExpectedGeneration
    )

    if ($ExpectedGeneration -notmatch '^\d+$' -or $ExpectedGeneration -eq '0') {
        throw 'Expected generation for conditional delete is invalid'
    }
    $Current = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Uri
    if (-not $Current.exists -or $Current.generation -ne $ExpectedGeneration) {
        throw "Conditional delete generation mismatch: $Uri"
    }
    Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
        'storage', 'rm',
        "--if-generation-match=$ExpectedGeneration",
        $Uri
    ) | Out-Null
    $After = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Uri
    if ($After.exists) {
        throw "Conditional delete verification failed: $Uri"
    }
}

function Assert-GcloudFileMatches {
    param(
        [Parameter(Mandatory)][string]$Gcloud,
        [Parameter(Mandatory)][string]$LocalPath,
        [Parameter(Mandatory)][string]$Uri
    )

    $Temporary = Join-Path (
        [IO.Path]::GetTempPath()
    ) ("absorb-readback-" + [Guid]::NewGuid().ToString('N'))
    try {
        Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
            'storage', 'cp', '--quiet', $Uri, $Temporary
        ) | Out-Null
        $Local = Get-Item -LiteralPath $LocalPath
        $Remote = Get-Item -LiteralPath $Temporary
        if (
            $Local.Length -ne $Remote.Length -or
            (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $Temporary -Algorithm SHA256).Hash
        ) {
            throw "GCS read-back hash or size mismatch: $Uri"
        }
    } finally {
        if (Test-Path -LiteralPath $Temporary -PathType Leaf) {
            Remove-Item -LiteralPath $Temporary -Force
        }
    }
}
