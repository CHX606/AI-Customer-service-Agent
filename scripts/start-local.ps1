<#
.SYNOPSIS
一键启动本地全部服务：Docker Desktop → OpenSearch → 后端 → 前端。

.DESCRIPTION
已经在运行的服务会自动跳过，可以重复执行。后端和前端各开一个新窗口，
关闭对应窗口（或在窗口里按 Ctrl+C）即可停止。

.EXAMPLE
.\scripts\start-local.ps1
.\scripts\start-local.ps1 -NoFrontend
#>
param(
    [switch]$NoFrontend,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }

function Write-Step([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "    $Text" -ForegroundColor Green }

function Wait-Until([scriptblock]$Check, [int]$TimeoutSeconds, [string]$What) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Check) { return }
        Start-Sleep -Seconds 3
    }
    throw "$What 在 $TimeoutSeconds 秒内没有就绪"
}

function Test-Port([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { return $client.ConnectAsync('127.0.0.1', $Port).Wait(1000) -and $client.Connected }
    catch { return $false }
    finally { $client.Dispose() }
}

function Test-Docker {
    docker info *> $null
    return $LASTEXITCODE -eq 0
}

function Start-ServiceWindow([string]$Title, [string]$WorkingDirectory, [string]$Command) {
    Start-Process $shell -WorkingDirectory $WorkingDirectory -ArgumentList @(
        '-NoExit', '-Command', "`$host.UI.RawUI.WindowTitle = '$Title'; $Command"
    )
}

foreach ($file in '.env', 'infra\opensearch\.env.local', '.venv\Scripts\python.exe') {
    if (-not (Test-Path $file)) { throw "缺少 $file，请先按 README 的「本地启动」完成配置" }
}

Write-Step '检查 Docker'
if (-not (Test-Docker)) {
    $dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path $dockerDesktop)) { throw '没有找到 Docker Desktop，请先安装' }
    Start-Process $dockerDesktop
    Write-Host '    已打开 Docker Desktop，等待引擎就绪（可能需要一两分钟）…'
    Wait-Until { Test-Docker } 240 'Docker 引擎'
}
Write-Ok 'Docker 已就绪'

Write-Step '启动 OpenSearch'
docker compose --env-file infra\opensearch\.env.local -f infra\opensearch\compose.yml up -d
if ($LASTEXITCODE -ne 0) { throw 'OpenSearch 容器启动失败' }
Wait-Until { Test-Port 9200 } 180 'OpenSearch（9200 端口）'
Write-Ok 'OpenSearch 已就绪'

Write-Step "启动后端（端口 $BackendPort）"
if (Test-Port $BackendPort) {
    Write-Host "    端口 $BackendPort 已被占用，后端可能已在运行，跳过" -ForegroundColor Yellow
} else {
    Start-ServiceWindow '后端' $root ".\.venv\Scripts\python.exe -m uvicorn back.api:app --host 127.0.0.1 --port $BackendPort"
    Write-Ok '已在新窗口启动'
}

if (-not $NoFrontend) {
    Write-Step "启动前端（端口 $FrontendPort）"
    if (Test-Port $FrontendPort) {
        Write-Host "    端口 $FrontendPort 已被占用，前端可能已在运行，跳过" -ForegroundColor Yellow
    } else {
        $frontend = Join-Path $root 'front'
        $command = if (Test-Path (Join-Path $frontend 'node_modules')) { 'npm run dev' } else { 'npm ci; npm run dev' }
        Start-ServiceWindow '前端' $frontend $command
        Write-Ok '已在新窗口启动'
    }
}

Write-Step '等待后端完成启动（首次加载模型可能需要几分钟）'
$health = $null
Wait-Until {
    try { $script:health = Invoke-RestMethod "http://127.0.0.1:$BackendPort/health" -TimeoutSec 3; return $true }
    catch { return $false }
} 600 '后端'

Write-Host ''
if ($health.status -eq 'ok') {
    Write-Host '全部启动完成' -ForegroundColor Green
} else {
    Write-Host "后端已启动，但有组件不可用（status=$($health.status)），请查看「后端」窗口的日志" -ForegroundColor Yellow
}
Write-Host "  后端：http://127.0.0.1:$BackendPort"
if (-not $NoFrontend) { Write-Host "  前端：http://127.0.0.1:$FrontendPort" }
