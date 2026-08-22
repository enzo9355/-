function Protect-NativeProcessText {
    param([AllowEmptyString()][string]$Text)

    $MaxTextLength = 16384
    $WasTruncated = $Text.Length -gt $MaxTextLength
    $Safe = if ($WasTruncated) {
        $Text.Substring(0, $MaxTextLength)
    } else {
        $Text
    }
    $QuotedValue = (
        """[^""\r\n]{0,$MaxTextLength}""|" +
        "'[^'\r\n]{0,$MaxTextLength}'"
    )
    $Value = (
        "(?:$QuotedValue|" +
        "[^\s,;}\]\r\n&#]{1,$MaxTextLength})"
    )
    $LineValue = "[^\r\n]{1,$MaxTextLength}"
    $PrefixedKey = (
        "[A-Za-z][A-Za-z0-9_]{0,127}_" +
        "(?:token|password|secret|api_key|access_token|client_secret)"
    )
    $AuthorizationPrefix = (
        "(?i)((?<![A-Za-z0-9_])[""']?" +
        "(?:authorization|" +
        "[A-Za-z][A-Za-z0-9_]{0,127}_authorization)[""']?" +
        "\s*[:=]\s*)"
    )
    $CookiePrefix = (
        "(?i)((?<![A-Za-z0-9_])[""']?" +
        "(?:cookie|[A-Za-z][A-Za-z0-9_]{0,127}_cookie)[""']?" +
        "\s*[:=]\s*)"
    )
    $KeyPrefix = (
        "(?i)((?<![A-Za-z0-9_])[""']?" +
        "(?:token|password|secret|api_key|" +
        "$PrefixedKey)[""']?" +
        "\s*[:=]\s*)"
    )
    $CliPrefix = (
        '(?i)(--(?:token|password|secret)\s+)'
    )
    $CliCompositePrefix = '(?i)(--(?:authorization|cookie)\s+)'
    $Safe = [regex]::Replace(
        $Safe,
        $AuthorizationPrefix + $LineValue,
        '$1[REDACTED]'
    )
    $Safe = [regex]::Replace(
        $Safe,
        $CliCompositePrefix + $LineValue,
        '$1[REDACTED]'
    )
    $Safe = [regex]::Replace(
        $Safe,
        $CookiePrefix + $LineValue,
        '$1[REDACTED]'
    )
    $Safe = [regex]::Replace($Safe, $KeyPrefix + $Value, '$1[REDACTED]')
    $Safe = [regex]::Replace($Safe, $CliPrefix + $Value, '$1[REDACTED]')
    $Safe = [regex]::Replace(
        $Safe,
        "(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{1,$MaxTextLength}",
        '$1[REDACTED]'
    )
    if ($WasTruncated) {
        return $Safe + '[TRUNCATED]'
    }
    return $Safe
}

function Invoke-NativeProcessCaptured {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1024, 4194304)][int]$MaxOutputChars = 1048576,
        [switch]$AllowFailure
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ExitCode = 1
    $Builder = New-Object System.Text.StringBuilder
    $WasTruncated = $false
    try {
        $ErrorActionPreference = 'Continue'
        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $Line = Protect-NativeProcessText ([string]$_)
            if ($Builder.Length -lt $MaxOutputChars) {
                $Chunk = if ($Builder.Length) {
                    [Environment]::NewLine + $Line
                } else {
                    $Line
                }
                $Remaining = $MaxOutputChars - $Builder.Length
                if ($Chunk.Length -gt $Remaining) {
                    [void]$Builder.Append($Chunk.Substring(0, $Remaining))
                    $WasTruncated = $true
                } else {
                    [void]$Builder.Append($Chunk)
                }
            } else {
                $WasTruncated = $true
            }
        }
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    $Text = $Builder.ToString()
    if ($WasTruncated) {
        $Text += '[TRUNCATED]'
    }
    if ($ExitCode -ne 0 -and -not $AllowFailure) {
        throw "Native process failed with exit code ${ExitCode}: $Text"
    }
    return [pscustomobject]@{
        exit_code = $ExitCode
        text = $Text
    }
}

function Invoke-NativeProcessStreaming {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)][string]$LogPath,
        [ValidateRange(1, 1000)][int]$TailLineCount = 100,
        [switch]$AllowFailure
    )

    try {
        Get-Command -Name $FilePath -ErrorAction Stop | Out-Null
    } catch {
        throw 'Native process start failed: executable was not found'
    }

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ExitCode = 1
    $LogWriteFailed = $false
    $Tail = New-Object 'System.Collections.Generic.Queue[string]'
    try {
        $ErrorActionPreference = 'Continue'
        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $Line = Protect-NativeProcessText ([string]$_)
            try {
                Add-Content `
                    -LiteralPath $LogPath `
                    -Value $Line `
                    -Encoding utf8 `
                    -ErrorAction Stop
            } catch {
                $LogWriteFailed = $true
                return
            }
            $Tail.Enqueue($Line)
            while ($Tail.Count -gt $TailLineCount) {
                [void]$Tail.Dequeue()
            }
        }
        $ExitCode = $LASTEXITCODE
        if ($LogWriteFailed) {
            throw 'Native process log write failed'
        }
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    $Text = [string]::Join([Environment]::NewLine, $Tail.ToArray())
    if ($ExitCode -ne 0 -and -not $AllowFailure) {
        throw "Native process failed with exit code ${ExitCode}: $Text"
    }
    return [pscustomobject]@{
        exit_code = $ExitCode
        text = $Text
    }
}
