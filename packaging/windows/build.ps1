# Builds dist\ohbot-chat\ohbot-chat.exe (one-folder): powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

uv sync --group build --group desktop
if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run --group build --group desktop pyinstaller --noconfirm --clean packaging\ohbotchat.spec
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Compress-Archive -Path dist\ohbot-chat -DestinationPath dist\ohbot-chat.zip -Force

# ASCII only: Windows PowerShell 5.1 reads a BOM-less script as ANSI
Write-Host "Done: dist\ohbot-chat\ohbot-chat.exe and dist\ohbot-chat.zip"
