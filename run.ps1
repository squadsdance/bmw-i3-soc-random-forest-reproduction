# Forward extra arguments, such as --trees 25, to the Python experiment.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ExperimentArgs)
# Stop on PowerShell errors and run from the project folder so paths work.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Prefer the bundled Python on this computer; otherwise use Python from PATH.
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $taskPython = $bundledPython
} else {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
# & launches the chosen program; @ExperimentArgs forwards the argument list.
& $taskPython main.py @ExperimentArgs
# Preserve Python's failure code so callers can tell if the experiment failed.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
