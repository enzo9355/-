[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory)][string]$ReceiptPath,
    [string[]]$SuccessorReceiptPath = @(),
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots'
)

$ErrorActionPreference = 'Stop'
$RequestedWhatIf = [bool]$WhatIfPreference
$WhatIfPreference = $false
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') {
    throw 'Bucket is not allowlisted'
}
. (Join-Path $PSScriptRoot 'observation_release_common.ps1')

$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$ReceiptRoot = Join-Path $DataRoot 'release\observation-lkg'
$ResolvedReceipt = Assert-PathWithinRoot -Path $ReceiptPath -Root $ReceiptRoot
$ResolvedSuccessors = @(
    foreach ($Path in $SuccessorReceiptPath) {
        Assert-PathWithinRoot -Path $Path -Root $ReceiptRoot
    }
)
$JournalPath = "$ResolvedReceipt.pending.json"
if (-not [IO.File]::Exists($JournalPath)) {
    throw 'Observation LKG pending pointer journal does not exist'
}
$TemporaryJournalPath = "$JournalPath.tmp"
if ([IO.File]::Exists($TemporaryJournalPath)) {
    throw 'Observation LKG pending pointer journal temporary file exists'
}

$PointerLockPath = Join-Path $ReceiptRoot 'pointer-update.lock'
$PointerLock = $null
try {
    $PointerLock = [IO.File]::Open(
        $PointerLockPath,
        [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None
    )
} catch {
    throw 'Observation LKG pointer lock is held'
}

try {
    $Receipt = Read-StrictUtf8JsonFile `
        -Path $ResolvedReceipt `
        -MaximumBytes 1MB
    if (
        $Receipt.schema_version -ne 1 -or
        $Receipt.kind -ne 'absorb-observation-lkg' -or
        $Receipt.bucket -ne $Bucket -or
        -not ($Receipt.pointers -is [array])
    ) {
        throw 'Observation LKG receipt is invalid'
    }
    $CaptureRoot = Split-Path -Parent $ResolvedReceipt
    $Entries = @(Read-GcloudPendingPointerJournal -Path $JournalPath)
    $PointerStagingRoot = Join-Path $DataRoot 'release\pointer-staging'
    $Results = New-Object System.Collections.Generic.List[object]

    foreach ($Entry in $Entries) {
        $SourcePath = Assert-PathWithinRoot `
            -Path ([string]$Entry.source) `
            -Root $PointerStagingRoot
        $Source = Get-Item -LiteralPath $SourcePath -Force
        $SourceHash = (
            Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if (
            $Source.PSIsContainer -or
            $Source.Length -le 0 -or
            $Source.Length -gt 1MB -or
            $SourceHash -ne [string]$Entry.source_sha256
        ) {
            throw 'Pending pointer journal source hash mismatch'
        }

        $Matches = @($Receipt.pointers | Where-Object {
            [string]$_.uri -eq [string]$Entry.uri
        })
        if ($Matches.Count -ne 1) {
            throw 'Observation LKG receipt pointer is missing or duplicated'
        }
        $Pointer = $Matches[0]
        Assert-ObservationLkgPointerCapture `
            -Pointer $Pointer `
            -CaptureRoot $CaptureRoot `
            -Bucket $Bucket
        $CapturedGeneration = if ($Pointer.exists) {
            [string]$Pointer.generation
        } else {
            '0'
        }
        if (
            $CapturedGeneration -notmatch '^\d+$' -or
            $CapturedGeneration -ne [string]$Entry.expected_generation
        ) {
            throw 'Pending pointer journal generation does not match receipt'
        }

        $AppliedGeneration = [string]$Pointer.applied_generation
        $EvidenceKind = $null
        if ($AppliedGeneration -match '^\d+$') {
            $EvidenceKind = 'receipt'
        } else {
            $Current = Get-GcloudObjectState `
                -Gcloud $Gcloud `
                -Uri ([string]$Entry.uri)
            if (
                $Current.exists -and
                [string]$Current.generation -ne $CapturedGeneration
            ) {
                try {
                    Assert-GcloudFileMatches `
                        -Gcloud $Gcloud `
                        -LocalPath $SourcePath `
                        -Uri "$([string]$Entry.uri)#$([string]$Current.generation)"
                    $AppliedGeneration = [string]$Current.generation
                    $EvidenceKind = 'current_generation'
                } catch {
                    if ($ResolvedSuccessors.Count -eq 0) { throw }
                    $Successor = Get-VerifiedSuccessorPointerEvidence `
                        -Entry $Entry `
                        -SuccessorReceiptPaths $ResolvedSuccessors `
                        -ReceiptRoot $ReceiptRoot `
                        -Bucket $Bucket
                    $AppliedGeneration = [string]$Successor.generation
                    $EvidenceKind = 'successor_receipt'
                }
            } elseif (
                (-not $Current.exists -and $CapturedGeneration -eq '0') -or
                ($Current.exists -and
                    [string]$Current.generation -eq $CapturedGeneration)
            ) {
                $WhatIfPreference = $RequestedWhatIf
                $MutationApproved = $PSCmdlet.ShouldProcess(
                    [string]$Entry.uri,
                    'resume conditional pointer mutation from pending journal'
                )
                $WhatIfPreference = $false
                if (-not $MutationApproved) {
                    $Results.Add([pscustomobject]@{
                        uri = [string]$Entry.uri
                        before_generation = $CapturedGeneration
                        applied_generation = $null
                        evidence = 'mutation_required'
                    }) | Out-Null
                    continue
                }
                $Update = Invoke-GcloudConditionalCopy `
                    -Gcloud $Gcloud `
                    -Source $SourcePath `
                    -Destination ([string]$Entry.uri) `
                    -ExpectedGeneration $CapturedGeneration `
                    -ExpectedSourceSha256 $SourceHash
                $AppliedGeneration = [string]$Update.after_generation
                $EvidenceKind = 'conditional_resume'
            } else {
                throw 'Pending pointer journal remote state is inconsistent'
            }
        }
        if ($AppliedGeneration -notmatch '^\d+$' -or $AppliedGeneration -eq '0') {
            throw 'Pending pointer applied generation is invalid'
        }
        $Pointer | Add-Member `
            -NotePropertyName applied_generation `
            -NotePropertyValue $AppliedGeneration `
            -Force
        $Results.Add([pscustomobject]@{
            uri = [string]$Entry.uri
            before_generation = $CapturedGeneration
            applied_generation = $AppliedGeneration
            source_sha256 = $SourceHash
            evidence = $EvidenceKind
        }) | Out-Null
    }

    $Unresolved = @($Results | Where-Object { -not $_.applied_generation })
    $WhatIfPreference = $RequestedWhatIf
    $FinalizeApproved = $Unresolved.Count -eq 0 -and $PSCmdlet.ShouldProcess(
            $ResolvedReceipt,
            'finalize receipt and remove verified pending pointer journal'
        )
    $WhatIfPreference = $false
    if ($FinalizeApproved) {
        $Receipt | Add-Member `
            -NotePropertyName applied_at `
            -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o')) `
            -Force
        $Receipt | Add-Member `
            -NotePropertyName reconciled_at `
            -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o')) `
            -Force
        $TemporaryReceipt = "$ResolvedReceipt.tmp"
        if ([IO.File]::Exists($TemporaryReceipt)) {
            throw 'Observation LKG receipt temporary file exists'
        }
        [IO.File]::WriteAllText(
            $TemporaryReceipt,
            ($Receipt | ConvertTo-Json -Depth 8),
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item `
            -LiteralPath $TemporaryReceipt `
            -Destination $ResolvedReceipt `
            -Force
        Remove-GcloudPendingPointerJournal -Path $JournalPath
    }

    [ordered]@{
        schema_version = 1
        kind = 'absorb-observation-lkg-reconciliation'
        receipt = $ResolvedReceipt
        pending_journal = $JournalPath
        results = $Results.ToArray()
    } | ConvertTo-Json -Depth 8 -Compress
} finally {
    if ($null -ne $PointerLock) {
        $PointerLock.Dispose()
    }
    $WhatIfPreference = $RequestedWhatIf
}
