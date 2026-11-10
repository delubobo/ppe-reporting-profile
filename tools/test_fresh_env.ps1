# Fresh-environment test (Windows PowerShell 5.1 or later).
#
#   powershell -ExecutionPolicy Bypass -File tools\test_fresh_env.ps1 [-Python <python executable>]
#
# Copies the package to a new folder under the temporary directory (without .git and archive),
# creates a new virtual environment there, installs requirements.txt, runs the unit tests and then
# tools\reproduce_table.py. The test passes only if both exit with code 0. The copy is kept.
param([string]$Python = "python")

$pkg = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$work = Join-Path $env:TEMP "ppe_reporting_profile_test_$stamp"

Write-Output "package: $pkg"
Write-Output "fresh copy: $work"
robocopy $pkg $work /E /XD .git archive .venv __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Output "copy failed (robocopy code $LASTEXITCODE)"; exit 1 }
Set-Location $work

Write-Output "--- interpreter"
& $Python --version
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Output "venv creation failed"; exit 1 }
$py = Join-Path $work ".venv\Scripts\python.exe"
& $py --version

Write-Output "--- pip install -r requirements.txt"
& $py -m pip install --disable-pip-version-check --no-input -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Output "pip install failed"; exit 1 }
Write-Output "--- installed packages"
& $py -m pip freeze --disable-pip-version-check

Write-Output "--- python -m pytest tests -q"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $py -m pytest tests -q -p no:cacheprovider
$pytestCode = $LASTEXITCODE

Write-Output "--- python tools\reproduce_table.py"
$start = Get-Date
& $py tools\reproduce_table.py
$reproduceCode = $LASTEXITCODE
Write-Output ("reproduce_table.py took {0:N0} s" -f ((Get-Date) - $start).TotalSeconds)

Write-Output "--- summary"
Write-Output "pytest exit code: $pytestCode"
Write-Output "reproduce_table.py exit code: $reproduceCode"
if ($pytestCode -eq 0 -and $reproduceCode -eq 0) {
    Write-Output "FRESH ENVIRONMENT TEST PASSED"
    exit 0
}
Write-Output "FRESH ENVIRONMENT TEST FAILED"
exit 1
