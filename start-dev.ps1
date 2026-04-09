param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3001,
    [string]$BackendHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$FrontendDir = Join-Path $ProjectRoot "frontend"

if (-not (Test-Path $VenvActivate)) {
    Write-Error "Missing virtual environment activation script at $VenvActivate. Create it first with: py -m venv .venv"
}

if (-not (Test-Path $FrontendDir)) {
    Write-Error "Missing frontend directory at $FrontendDir"
}

$BackendCommand = @(
    "Set-Location '$ProjectRoot'"
    ". '$VenvActivate'"
    "Write-Host 'Starting backend on http://${BackendHost}:${BackendPort}' -ForegroundColor Cyan"
    "python -m uvicorn api:app --host $BackendHost --port $BackendPort"
) -join "; "

$FrontendCommand = @(
    "Set-Location '$FrontendDir'"
    "Write-Host 'Starting frontend on http://127.0.0.1:$FrontendPort' -ForegroundColor Green"
    "npm run dev -- --port $FrontendPort"
) -join "; "

Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $BackendCommand) | Out-Null
Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $FrontendCommand) | Out-Null

Write-Host "Launched backend and frontend in separate PowerShell windows." -ForegroundColor Yellow
Write-Host "Backend:  http://${BackendHost}:${BackendPort}/health"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
