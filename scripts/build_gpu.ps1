param(
    [string]$Nvcc = "nvcc",
    [string]$HostCompiler = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BaselineSource = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter "main_gpu.cu" |
    Select-Object -First 1
if (-not $BaselineSource) {
    throw "cannot locate the GPU source directory"
}
$Source = Split-Path -Parent $BaselineSource.FullName
$Bin = Join-Path $Root "bin\gpu"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

$CcbinArguments = @()
if ($HostCompiler) {
    if (-not (Test-Path -LiteralPath $HostCompiler)) {
        throw "MSVC host compiler not found: $HostCompiler"
    }
    $HostCompilerDirectory = if ((Get-Item -LiteralPath $HostCompiler).PSIsContainer) {
        $HostCompiler
    } else {
        Split-Path -Parent $HostCompiler
    }
    $CcbinArguments = @("-ccbin", $HostCompilerDirectory)
} elseif (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "nvcc requires MSVC cl.exe. Run this script from a VS Developer PowerShell or pass -HostCompiler <path-to-cl.exe>."
}

$Targets = @(
    @{ Source = "main_gpu.cu"; Output = "flat_kernel.exe"; Libraries = @() },
    @{ Source = "main_gpu_cublas.cu"; Output = "flat_cublas.exe"; Libraries = @("-lcublas") },
    @{ Source = "main_gpu_ivf.cu"; Output = "ivf_kernel.exe"; Libraries = @() },
    @{ Source = "main_gpu_ivf_cublas.cu"; Output = "ivf_cublas.exe"; Libraries = @("-lcublas") },
    @{ Source = "main_gpu_ivf_grouped.cu"; Output = "ivf_grouped.exe"; Libraries = @() },
    @{ Source = "hetero_ivfpq_sync.cu"; Output = "hetero_ivfpq_sync.exe"; Libraries = @() },
    @{ Source = "hetero_ivfpq_pipeline.cu"; Output = "hetero_ivfpq_pipeline.exe"; Libraries = @() }
)

Push-Location $Source
try {
    foreach ($Target in $Targets) {
        # Keep paths passed to nvcc ASCII-only. Older Windows host toolchains
        # may otherwise decode the Chinese workspace path with the OEM page.
        $Arguments = @(
            $Target.Source,
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "/arch:AVX2,/openmp",
            "-o",
            (Join-Path "..\bin\gpu" $Target.Output)
        ) + $CcbinArguments + $Target.Libraries
        & $Nvcc @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "GPU build failed: $($Target.Source)"
        }
    }
} finally {
    Pop-Location
}

Write-Host "Built GPU executables in $Bin"
