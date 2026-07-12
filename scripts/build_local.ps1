param(
    [string]$Compiler = "g++",
    [string]$Configuration = "release"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $Root "bin"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

$Optimization = if ($Configuration -eq "debug") { "-O0" } else { "-O3" }
$DebugFlag = if ($Configuration -eq "debug") { "-g" } else { "-DNDEBUG" }

& $Compiler (Join-Path $Root "main.cc") `
    -std=c++17 $Optimization $DebugFlag -fopenmp -pthread `
    -o (Join-Path $Bin "ann_workflow.exe")

if ($LASTEXITCODE -ne 0) {
    throw "local build failed"
}

Write-Host "Built $Bin\ann_workflow.exe"
