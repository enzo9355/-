[CmdletBinding()]
param()

function Test-PreMarketCompletion {
    <#
    .SYNOPSIS
    Returns $true only when the TW pre-market pipeline has completed
    end-to-end for the given applicable trading date, using verified
    completion evidence.

    Completion evidence contract:
      1. The local report v2 pointer publish\reports\v2\latest-TW-pre_market.json
         exists and its applicable_trading_date equals the target date.
      2. The task wrapper receipt logs\tasks\current-TW-PreMarket.json records
         success=$true for a run started on the same local date. The wrapper
         writes this receipt only after the full child pipeline (pre-market
         build, GCS upload with remote verification, and LINE notify) exits
         with code 0.

    Both conditions are required: a local pointer alone only proves local
    publication, not remote publication or delivery. A corrupt or stale file
    fails the check (fail-open into a re-run, which is safe and idempotent).

    This helper must not mutate the caller's PowerShell language semantics:
    it sets no StrictMode or error preference, and every optional JSON
    property is read through PSObject.Properties so the helper behaves
    identically no matter how the caller's session is configured.

    The caller must enforce the DataRoot allowlist before invoking this helper.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$DataRoot,
        [Parameter(Mandatory = $true)][string]$TargetDate
    )

    if ($TargetDate -notmatch '^\d{4}-\d{2}-\d{2}$') { return $false }
    if ([string]::IsNullOrWhiteSpace($DataRoot)) { return $false }

    $ReportV2Root = Join-Path $DataRoot 'publish\reports\v2'
    $LatestPreMarketPath = Join-Path $ReportV2Root 'latest-TW-pre_market.json'
    if (-not (Test-Path -LiteralPath $LatestPreMarketPath -PathType Leaf)) {
        return $false
    }
    try {
        $ExistingLatest = Get-Content -LiteralPath $LatestPreMarketPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch { return $false }
    $ApplicableDateProperty = $ExistingLatest.PSObject.Properties['applicable_trading_date']
    if (
        $null -eq $ApplicableDateProperty -or
        [string]$ApplicableDateProperty.Value -ne $TargetDate
    ) { return $false }

    $TaskStatusPath = Join-Path $DataRoot 'logs\tasks\current-TW-PreMarket.json'
    if (-not (Test-Path -LiteralPath $TaskStatusPath -PathType Leaf)) {
        return $false
    }
    try {
        $TaskStatus = Get-Content -LiteralPath $TaskStatusPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch { return $false }
    $SuccessProperty = $TaskStatus.PSObject.Properties['success']
    if ($null -eq $SuccessProperty -or [bool]$SuccessProperty.Value -ne $true) {
        return $false
    }
    try {
        $StartedAtProperty = $TaskStatus.PSObject.Properties['started_at']
        if ($null -eq $StartedAtProperty) { return $false }
        $StartedAt = [DateTimeOffset]::Parse([string]$StartedAtProperty.Value)
    }
    catch { return $false }
    if ($StartedAt.ToString('yyyy-MM-dd') -ne $TargetDate) { return $false }

    return $true
}
