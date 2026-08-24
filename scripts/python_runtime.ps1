function Resolve-AbsorbPythonExecutable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepoRoot
    )

    if (
        [string]::IsNullOrWhiteSpace($RepoRoot) -or
        -not (Test-Path -LiteralPath $RepoRoot -PathType Container)
    ) {
        throw 'Repository root is unavailable'
    }
    $ResolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot -ErrorAction Stop).Path

    $OverrideIsPresent = $null -ne $env:ABSORB_PYTHON_EXE
    if ($OverrideIsPresent) {
        $Override = [string]$env:ABSORB_PYTHON_EXE
        if (
            [string]::IsNullOrWhiteSpace($Override) -or
            -not [IO.Path]::IsPathRooted($Override) -or
            -not (Test-Path -LiteralPath $Override -PathType Leaf)
        ) {
            throw 'ABSORB_PYTHON_EXE must be an existing absolute file path'
        }
        return (Resolve-Path -LiteralPath $Override -ErrorAction Stop).Path
    }

    $VenvPython = Join-Path $ResolvedRepoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        return (Resolve-Path -LiteralPath $VenvPython -ErrorAction Stop).Path
    }

    $SystemPython = Get-Command `
        -Name python `
        -CommandType Application `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $SystemPython) {
        throw 'Python executable was not found'
    }
    $SystemPath = [string]$SystemPython.Source
    if (
        [string]::IsNullOrWhiteSpace($SystemPath) -or
        -not [IO.Path]::IsPathRooted($SystemPath) -or
        -not (Test-Path -LiteralPath $SystemPath -PathType Leaf)
    ) {
        throw 'Python executable was not found'
    }
    return (Resolve-Path -LiteralPath $SystemPath -ErrorAction Stop).Path
}

function Assert-AbsorbPythonRuntime {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PythonExe,
        [Parameter(Mandatory)][string]$RepoRoot,
        [string[]]$RequiredImports = @('stock_papi')
    )

    if (
        [string]::IsNullOrWhiteSpace($PythonExe) -or
        -not [IO.Path]::IsPathRooted($PythonExe) -or
        -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)
    ) {
        throw 'Selected Python runtime is unavailable'
    }
    if (
        [string]::IsNullOrWhiteSpace($RepoRoot) -or
        -not (Test-Path -LiteralPath $RepoRoot -PathType Container)
    ) {
        throw 'Repository root is unavailable'
    }

    if ($RequiredImports.Count -lt 1 -or $RequiredImports | Where-Object {
        [string]::IsNullOrWhiteSpace($_) -or $_ -notmatch '^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$'
    }) {
        throw 'Required Python imports are invalid'
    }

    $ResolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot -ErrorAction Stop).Path
    $DepsRoot = Join-Path $ResolvedRepoRoot '.deps'
    $RuntimePythonPath = [string]::Join(
        [IO.Path]::PathSeparator,
        @($ResolvedRepoRoot, $DepsRoot)
    )
    $PreviousPythonPath = $env:PYTHONPATH
    $ExitCode = 1
    try {
        $env:PYTHONPATH = $RuntimePythonPath
        try {
            & $PythonExe -c 'import importlib, sys; [importlib.import_module(name) for name in sys.argv[1:]]' @RequiredImports 2>&1 | Out-Null
            $ExitCode = $LASTEXITCODE
        } catch {
            $ExitCode = 1
        }
    } finally {
        if ($null -eq $PreviousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $PreviousPythonPath
        }
    }

    if ($ExitCode -ne 0) {
        throw 'Selected Python runtime cannot import stock_papi or required ABSORB modules; install project requirements or set ABSORB_PYTHON_EXE'
    }
}
