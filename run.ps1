param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ExperimentArgs)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $taskPython = $bundledPython
} else {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
& $taskPython reproduce.py @ExperimentArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
