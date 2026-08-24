function Invoke-AbsorbUsPipelineNativeCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PythonExe,
        [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$Arguments,
        [Parameter(Mandatory)][string]$FailureLabel
    )

    & $PythonExe @Arguments
    $NativeExitCode = [int]$LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        Write-Error "$FailureLabel failed with exit code $NativeExitCode" `
            -ErrorAction Continue
        exit $NativeExitCode
    }
}
