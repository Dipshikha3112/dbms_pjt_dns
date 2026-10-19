Set-Location $PSScriptRoot
if (Test-Path '.\.venv\Scripts\Activate.ps1') { . '.\.venv\Scripts\Activate.ps1' }
if (-not $env:PHASE15_API_KEY) { throw 'Set PHASE15_API_KEY in this PowerShell session first.' }
$env:PHASE14_API_KEY = $env:PHASE15_API_KEY
python phase15/main.py
