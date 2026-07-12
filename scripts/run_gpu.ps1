param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$ResultsDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $Root "bin\gpu"
if (-not $ResultsDir) {
    $ResultsDir = Join-Path $Root "results\gpu"
}
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$Targets = @(
    "flat_kernel.exe",
    "flat_cublas.exe",
    "ivf_kernel.exe",
    "ivf_cublas.exe",
    "ivf_grouped.exe",
    "hetero_ivfpq_sync.exe",
    "hetero_ivfpq_pipeline.exe"
)

foreach ($Target in $Targets) {
    $Executable = Join-Path $Bin $Target
    if (-not (Test-Path -LiteralPath $Executable)) {
        throw "GPU executable is missing; run build_gpu.ps1 first: $Executable"
    }
    $Log = Join-Path $ResultsDir ($Target + ".log")
    # The CUDA programs intentionally write progress messages to stderr.  Use
    # native redirection so Windows PowerShell does not turn those lines into
    # error records, then keep both streams in one reproducible run log.
    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $Executable
    $QuotedDataDir = '"' + $DataDir.Replace('"', '\"') + '"'
    if ($Target -like "hetero_ivfpq_*") {
        $StartInfo.Arguments = "--data-dir $QuotedDataDir"
    } else {
        $StartInfo.Arguments = $QuotedDataDir
    }
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo
    if (-not $Process.Start()) {
        throw "cannot start GPU executable: $Target"
    }
    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()
    $Process.WaitForExit()
    $StdoutText = $StdoutTask.Result
    $StderrText = $StderrTask.Result
    $ExitCode = $Process.ExitCode
    [System.IO.File]::WriteAllText($Log, $StderrText + $StdoutText)
    Write-Host $StdoutText
    if ($ExitCode -ne 0) {
        throw "GPU run failed: $Target"
    }
}

Write-Host "GPU runs completed; logs are in $ResultsDir"
