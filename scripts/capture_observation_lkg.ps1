[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\AbsorbData',
    [string]$Bucket = 'line-stock-bot-498908-quant-snapshots'
)

$ErrorActionPreference = 'Stop'
if ($DataRoot -notin @('D:\AbsorbData', 'D:\StockPapiData')) {
    throw 'Data root is not allowlisted'
}
if ($Bucket -ne 'line-stock-bot-498908-quant-snapshots') {
    throw 'Bucket is not allowlisted'
}
. (Join-Path $PSScriptRoot 'observation_release_common.ps1')

$Gcloud = (Get-Command gcloud -ErrorAction Stop).Source
$ReleaseRoot = Join-Path $DataRoot 'release\observation-lkg'
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
$ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).Path
if (
    ((Get-Item -LiteralPath $ReleaseRoot).Attributes -band
        [IO.FileAttributes]::ReparsePoint) -ne 0
) {
    throw 'Observation LKG root must not be a reparse point'
}
$CaptureId = (
    [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ-') +
    [Guid]::NewGuid().ToString('N').Substring(0, 8)
)
$CaptureRoot = Join-Path $ReleaseRoot $CaptureId
New-Item -ItemType Directory -Path $CaptureRoot -Force | Out-Null
$CaptureRoot = Assert-PathWithinRoot -Path $CaptureRoot -Root $ReleaseRoot

$Definitions = @(
    @{
        name = 'quant-latest'
        uri = "gs://$Bucket/quant/v1/latest-TW.json"
    },
    @{
        name = 'quant-insights-latest'
        uri = "gs://$Bucket/quant/v1/latest-insights.json"
    },
    @{
        name = 'dashboard-latest'
        uri = "gs://$Bucket/dashboard/v1/latest-TW.json"
    },
    @{
        name = 'predictions-latest'
        uri = "gs://$Bucket/predictions/v1/latest-TW.json"
    },
    @{
        name = 'reports-v1-index'
        uri = "gs://$Bucket/reports/v1/index-TW.json"
    },
    @{
        name = 'reports-v1-latest'
        uri = "gs://$Bucket/reports/v1/latest-TW.json"
    },
    @{
        name = 'reports-v2-index'
        uri = "gs://$Bucket/reports/v2/index-TW.json"
    },
    @{
        name = 'reports-v2-post-close'
        uri = "gs://$Bucket/reports/v2/latest-TW-post_close.json"
    },
    @{
        name = 'reports-v2-pre-market'
        uri = "gs://$Bucket/reports/v2/latest-TW-pre_market.json"
    },
    @{
        name = 'reports-v2-weekly-model'
        uri = "gs://$Bucket/reports/v2/latest-TW-weekly_model.json"
    },
    @{
        name = 'quant-latest-us'
        uri = "gs://$Bucket/quant/v1/latest-US.json"
    },
    @{
        name = 'dashboard-latest-us'
        uri = "gs://$Bucket/dashboard/v1/latest-US.json"
    },
    @{
        name = 'predictions-latest-us'
        uri = "gs://$Bucket/predictions/v1/latest-US.json"
    },
    @{
        name = 'reports-v2-index-us'
        uri = "gs://$Bucket/reports/v2/index-US.json"
    },
    @{
        name = 'reports-v2-post-close-us'
        uri = "gs://$Bucket/reports/v2/latest-US-post_close.json"
    },
    @{
        name = 'reports-v2-pre-market-us'
        uri = "gs://$Bucket/reports/v2/latest-US-pre_market.json"
    }
)

$Pointers = New-Object System.Collections.Generic.List[object]
foreach ($Definition in $Definitions) {
    $State = Get-GcloudObjectState -Gcloud $Gcloud -Uri $Definition.uri
    if (-not $State.exists) {
        $Pointers.Add([ordered]@{
            name = $Definition.name
            uri = $Definition.uri
            exists = $false
            generation = $null
            previous_file = $null
            previous_sha256 = $null
            previous_size = 0
            applied_generation = $null
        }) | Out-Null
        continue
    }

    $FileName = "$($Definition.name).json"
    $Destination = Join-Path $CaptureRoot $FileName
    $VersionedUri = "$($Definition.uri)#$($State.generation)"
    Invoke-GcloudCaptured -Gcloud $Gcloud -Arguments @(
        'storage', 'cp', '--quiet', $VersionedUri, $Destination
    ) | Out-Null
    $Destination = Assert-PathWithinRoot `
        -Path $Destination `
        -Root $CaptureRoot
    $File = Get-Item -LiteralPath $Destination
    if ($File.Length -le 0 -or $File.Length -gt 1MB) {
        throw "Observation LKG pointer size is invalid: $($Definition.name)"
    }
    try {
        Get-Content -LiteralPath $Destination -Raw -Encoding utf8 |
            ConvertFrom-Json | Out-Null
    } catch {
        throw "Observation LKG pointer JSON is invalid: $($Definition.name)"
    }
    $Pointers.Add([ordered]@{
        name = $Definition.name
        uri = $Definition.uri
        exists = $true
        generation = [string]$State.generation
        previous_file = $FileName
        previous_sha256 = (
            Get-FileHash -LiteralPath $Destination -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        previous_size = [long]$File.Length
        applied_generation = $null
    }) | Out-Null
}

$Receipt = [ordered]@{
    schema_version = 1
    kind = 'absorb-observation-lkg'
    bucket = $Bucket
    capture_id = $CaptureId
    captured_at = [DateTimeOffset]::UtcNow.ToString('o')
    pointers = $Pointers.ToArray()
}
$ReceiptPath = Join-Path $CaptureRoot 'receipt.json'
[IO.File]::WriteAllText(
    $ReceiptPath,
    ($Receipt | ConvertTo-Json -Depth 8),
    [Text.UTF8Encoding]::new($false)
)

[ordered]@{
    receipt = $ReceiptPath
    capture_id = $CaptureId
    pointers = $Pointers.Count
} | ConvertTo-Json -Compress
