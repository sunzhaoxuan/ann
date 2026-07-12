param(
    [string]$Python = "python",
    [string]$Compiler = "g++"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Data = Join-Path $Root "data\smoke"
$Exe = Join-Path $Root "bin\ann_workflow.exe"

& $Python (Join-Path $Root "tools\generate_smoke_data.py") --output $Data
if ($LASTEXITCODE -ne 0) { throw "smoke data generation failed" }

& (Join-Path $PSScriptRoot "build_local.ps1") -Compiler $Compiler

$Common = @("--data-dir", $Data, "--queries", "16", "--k", "10")
$Runs = @(
    @("--method", "flat-scalar"),
    @("--method", "flat"),
    @("--method", "flat", "--parallel", "query", "--threads", "2"),
    @("--method", "flat", "--parallel", "list", "--threads", "2"),
    @("--method", "sq", "--rerank-p", "64"),
    @("--method", "sq-int8", "--rerank-p", "64"),
    @("--method", "pq", "--m", "4", "--ks", "16", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "pq", "--parallel", "list", "--threads", "2", "--m", "4", "--ks", "16", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivf", "--nlist", "8", "--nprobe", "4", "--train-size", "128", "--kmeans-iters", "2"),
    @("--method", "ivf", "--parallel", "list", "--threads", "2", "--nlist", "8", "--nprobe", "4", "--train-size", "128", "--kmeans-iters", "2"),
    @("--method", "ivfpq-global", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-global-opq", "--opq-iters", "2", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-global-opq", "--opq-iters", "2", "--parallel", "query", "--threads", "2", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-local", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-local", "--parallel", "list", "--threads", "2", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-local-opq", "--opq-iters", "2", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "ivfpq-local-opq", "--opq-iters", "2", "--parallel", "list", "--threads", "2", "--nlist", "8", "--nprobe", "4", "--m", "4", "--ks", "8", "--train-size", "128", "--kmeans-iters", "2", "--rerank-p", "64"),
    @("--method", "hnsw", "--hnsw-m", "8", "--ef-construction", "40", "--ef-search", "32"),
    @("--method", "hnsw", "--hnsw-layout", "rcm", "--hnsw-m", "8", "--ef-construction", "40", "--ef-search", "32"),
    @("--method", "hnsw", "--hnsw-layout", "gorder", "--gorder-window", "5", "--hnsw-m", "8", "--ef-construction", "40", "--ef-search", "32"),
    @("--method", "hnsw", "--hnsw-layout", "porder", "--gorder-window", "5", "--porder-profile-queries", "4", "--warmup-queries", "2", "--hnsw-m", "8", "--ef-construction", "40", "--ef-search", "32")
)

foreach ($Run in $Runs) {
    & $Exe @Common @Run
    if ($LASTEXITCODE -ne 0) { throw "smoke test failed: $($Run -join ' ')" }
}

Write-Host "All local workflow smoke tests passed."
