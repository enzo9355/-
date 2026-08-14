. (Join-Path $PSScriptRoot 'native_process.ps1')
Import-Module Microsoft.PowerShell.Utility -MaximumVersion 5.1 -ErrorAction Stop

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

    $Existing = @()
    if ([IO.File]::Exists($Path)) {
        try {
            $Existing = @(Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json)
        } catch {
            throw 'Pending pointer journal is invalid'
        }
    }
    $Temporary = "$Path.tmp"
    if ([IO.File]::Exists($Temporary)) {
        throw 'Pending pointer journal temporary file exists'
    }
    try {
        [IO.File]::WriteAllText(
            $Temporary,
            (@($Existing) + @($Entry) | ConvertTo-Json -Depth 8),
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
