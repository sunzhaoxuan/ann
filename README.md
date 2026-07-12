# ANN 并行程序设计实验：统一工作流

本目录把本学期分散在体系结构、SIMD、多线程、MPI 和 GPU 阶段的 ANN（近似最近邻搜索）代码串成了一条可重复的实验链。服务器固定编译入口仍是 `main.cc`；统一算法实现放在 header-only 的 `workflow_main.h`，原来的历史入口保存在 `legacy_main.h`。

正式运行环境分为三类：CPU/SIMD/多线程实验在 ARM64 课程服务器通过原版 `test.sh` 提交；MPI 实验手动使用 `mpic++` 编译并通过 `qsub_mpi.sh` 提交；GPU 实验在 CUDA 环境单独运行。

## 1. 实验阶段与代码对应关系

| 阶段 | 报告关注点 | 统一入口中的方法 | 主要实现 |
| --- | --- | --- | --- |
| CPU/体系结构 | 标量扫描、缓存与基线 | `flat-scalar` | `flat_scan.h` |
| SIMD | 向量化、SQ、PQ | `flat`、`sq`、`sq-int8`、`pq` | `flat_scan_simd.h`、`sq_scan_simd*.h`、`pq_simd_scan.h` |
| 多线程 | 查询级与倒排表级并行 | `--parallel query`、`--parallel list` | OpenMP/Pthread 扫描实现 |
| 索引算法 | IVF、IVF-PQ、HNSW | `ivf`、`ivfpq-global`、`ivfpq-local`、`hnsw` | `ivf*_scan_simd.h`、`hnswlib/` |
| MPI | 数据分片、通信与混合并行 | `mpi-ivfpq`、`mpi-ivf-hnsw`、`mpi-random-hnsw`、`mpi-kmeans-hnsw`、`mpi-hoh` | `*_mpi.h`、`mpi_ivfpq_*.h` |
| GPU | Flat/IVF 的 CUDA 与 cuBLAS 版本 | 五个独立 CUDA 可执行文件 | `GPU编程/*.cu` |

这条链对应现有各阶段实验报告的演进关系：先建立可验证的精确基线，再进行 SIMD 量化近似，随后比较查询级/表级多线程，之后扩展到多进程分片，最后比较 CUDA kernel 与 cuBLAS 实现。所有阶段都使用相同的数据格式和 Recall@k 指标。

## 2. 数据约定

默认数据目录中应包含：

- `DEEP100K.base.100k.fbin`：底库向量；
- `DEEP100K.query.fbin`：查询向量；
- `DEEP100K.gt.query.100k.top100.bin`：真实近邻编号。

向量文件采用 `[uint32 数量][uint32 维度][连续数据]`。底库和查询的数据类型为 `float`，ground truth 为 `uint32`。也可以分别用 `--base-file`、`--query-file` 和 `--ground-truth-file` 覆盖文件路径。

## 3. ARM64 服务器正式工作流

本节适用于 CPU、SIMD 和多线程实验。MPI 是例外，使用方法见第 6 节。`test.sh` 是课程提供的加密脚本，禁止修改。它固定执行：

- `g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11`；
- 提交仓库原有的 `qsub.sh`；
- 返回 `test.e` 和 `test.o`，并记录提交结果。

因此不能向 `test.sh` 追加算法参数。先编辑 `files/workflow.conf`，每行填写一个统一入口参数，例如：

```text
--data-dir
/anndata
--method
flat
--queries
2000
--k
10
--output
files/workflow_results.csv
```

随后严格按课程格式提交：

```bash
# 第一个参数为实验序号，第二个参数为申请节点数
# 例如 SIMD 实验只能申请一个节点
bash test.sh 1 1
```

`main.cc` 在无命令行参数时自动读取 `/home/$USER/files/workflow.conf`；本地调试时仍可直接向编译后的程序传参。额外结果必须写入 `files/`，不要删除或修改既有 `test.o/test.e`。ARM64 编译器会自动选择真实 NEON 实现，x86 兼容层不会进入服务器正式路径。

并行模式含义：

- `none`：单查询串行执行，适合测量单请求延迟；
- `query`：查询之间并行，输出的平均时间是“批次墙钟时间 / 查询数”，表示吞吐摊销时间；
- `list`：单个查询内部并行扫描数据块或倒排表，适合观察单请求并行加速。

本地可使用 `--help` 查看全部参数；服务器上把这些参数逐行写进 `workflow.conf`。`--output` 会追加统一格式的 CSV 行，便于不同阶段直接作图比较。

## 4. 如何选择具体实验和参数

### 4.1 方法与并行模式

| `--method` | 实验含义 | 可用 `--parallel` | 主要参数 |
| --- | --- | --- | --- |
| `flat-scalar` | 标量精确扫描基线 | `none`、`query` | `threads` |
| `flat` | ARM NEON 精确扫描 | `none`、`query`、`list` | `threads` |
| `sq` | float/uint8 标量量化近似扫描 | `none`、`query` | `rerank-p` |
| `sq-int8` | int8 SQ 扫描 | `none`、`query` | `rerank-p` |
| `pq` | Product Quantization | `none`、`query`、`list` | `m`、`ks`、`rerank-p`、`local-p` |
| `ivf` | IVF 倒排索引 | `none`、`query`、`list` | `nlist`、`nprobe`、`local-p` |
| `ivfpq-global` | IVF + 全局 PQ | `none`、`query` | IVF/PQ 参数、`rerank-p` |
| `ivfpq-global-opq` | 原始空间 IVF + 原始向量上训练的全局 OPQ/PQ | `none`、`query` | IVF/PQ 参数、`opq-iters`、`rerank-p` |
| `ivfpq-local` | IVF + 每个 list 的局部 PQ | `none`、`query`、`list` | IVF/PQ 参数、`rerank-p`、`local-p` |
| `ivfpq-local-opq` | IVF + 全局 OPQ 旋转 + 每个 list 的局部 PQ | `none`、`query`、`list` | IVF/PQ 参数、`opq-iters`、`rerank-p`、`local-p` |
| `hnsw` | HNSW 图索引 | `none`、`query` | `hnsw-m`、`ef-construction`、`ef-search` |

`query` 是多个查询之间并行，适合测试吞吐量；`list` 是单个查询内部并行，适合测试单请求加速。表中未列出 `list` 的方法即使填写该值也不会获得对应的表级并行优化。

### 4.2 通用参数

| 参数 | 含义 | DEEP100K 建议起点 |
| --- | --- | --- |
| `--data-dir` | 三个数据文件所在目录 | `/anndata` |
| `--queries` | 实际测试的查询数 | 调试 `10`，正式 `2000` |
| `--k` | 返回结果数和 Recall@k 的 k | `10`，且不能超过 ground truth 宽度 100 |
| `--parallel` | `none`、`query` 或 `list` | 单线程基线用 `none` |
| `--threads` | OpenMP 线程数 | 依次比较 `1, 2, 4, 8` |
| `--nlist` | IVF 聚类/list 数量 | `100` |
| `--nprobe` | 每次查询访问的 list 数 | 依次比较 `1, 4, 8, 16, 32`，且不大于 `nlist` |
| `--m` | PQ 子空间数 | DEEP100K 维度为 96，推荐 `12` 或 `16`，必须整除 96 |
| `--ks` | 每个 PQ 子空间的聚类中心数 | `256`，代码编码上限也是 256 |
| `--train-size` | KMeans/PQ 训练向量数 | `10000`，不超过底库大小 |
| `--kmeans-iters` | KMeans 迭代次数 | `6` |
| `--opq-iters` | OPQ/PQ 与正交 Procrustes 的交替次数 | 调试 `1` 或 `2`，正式从 `2` 开始比较 |
| `--rerank-p` | 用原始 float 精排的候选数 | 比较 `10, 50, 100, 200`，且不小于 k |
| `--local-p` | 每线程或每分片保留的局部候选数 | 正确性对比时设为不小于 `rerank-p` |
| `--hnsw-m` | HNSW 每点最大连接规模 | `16` |
| `--ef-construction` | HNSW 建图搜索宽度 | `150` 或 `200` |
| `--ef-search` | HNSW 查询搜索宽度 | 比较 `16, 32, 64, 128`，且不小于 k |
| `--hnsw-layout` | HNSW 建图后的内存布局 | `original`、`rcm`、`gorder` 或 `porder`；重排保持图结构和 Recall 不变 |
| `--gorder-window` | Gorder 滑动窗口大小 | 默认 `5`，与论文实验设置一致 |
| `--porder-profile-queries` | Porder画像查询数 | 默认 `200`；从查询集开头取，正式计时不包含这些查询 |
| `--warmup-queries` | Porder重排后的预热查询数 | 默认 `100`；正式计时不包含这些查询 |
| `--evaluation-query-offset` | 非Porder实验跳过的查询数 | Porder对照实验设为 `profile + warmup` |
| `--output` | 追加结果的 CSV 文件 | `files/workflow_results.csv` |

### 4.3 可直接使用的配置

以下每段都是一份完整配置。把所需的一段复制到 `files/workflow.conf`，保存后再执行 `bash test.sh 实验序号 节点数`。

#### 精确标量基线

```text
--data-dir
/anndata
--method
flat-scalar
--parallel
none
--queries
2000
--k
10
--output
files/workflow_results.csv
```

#### ARM NEON Flat

```text
--data-dir
/anndata
--method
flat
--parallel
none
--queries
2000
--k
10
--output
files/workflow_results.csv
```

先运行 `flat-scalar`，再只把 `--method` 改为 `flat`，即可得到 SIMD 加速比；两者都应接近 Recall@10 = 1。

#### SQ 或 PQ

```text
--data-dir
/anndata
--method
pq
--parallel
none
--queries
2000
--k
10
--m
16
--ks
256
--train-size
10000
--kmeans-iters
6
--rerank-p
200
--output
files/workflow_results.csv
```

测试 SQ 时把方法改为 `sq` 或 `sq-int8`，可以删除 `m/ks/train-size/kmeans-iters`，保留 `rerank-p`。比较 PQ 精度时固定其他参数，只改变 `rerank-p`。

#### OpenMP 查询级并行

```text
--data-dir
/anndata
--method
flat
--parallel
query
--threads
8
--queries
2000
--k
10
--output
files/workflow_results.csv
```

分别提交线程数 `1, 2, 4, 8`。`query` 模式输出的是批次墙钟时间除以查询数，表示吞吐摊销时间，不等同于单请求尾延迟。

#### IVF 表级并行

```text
--data-dir
/anndata
--method
ivf
--parallel
list
--threads
8
--queries
2000
--k
10
--nlist
100
--nprobe
16
--train-size
10000
--kmeans-iters
6
--local-p
200
--output
files/workflow_results.csv
```

研究 latency-recall trade-off 时固定 `nlist=100`，依次改变 `nprobe`；研究线程扩展性时固定 `nprobe`，只改变 `threads`。

#### IVF-PQ 局部码本与表级并行

```text
--data-dir
/anndata
--method
ivfpq-local
--parallel
list
--threads
8
--queries
2000
--k
10
--nlist
100
--nprobe
16
--m
16
--ks
256
--train-size
10000
--kmeans-iters
6
--rerank-p
200
--local-p
200
--output
files/workflow_results.csv
```

要比较全局/局部 PQ，只把方法改为 `ivfpq-global`，并把并行模式改为 `none` 或 `query`。当 `local-p` 小于 `rerank-p` 时，表级并行可能更快但会提前丢失候选，因此比较实现正确性时先令两者相等。

要启用 OPQ，把方法改为 `ivfpq-local-opq`，并增加例如 `--opq-iters 2`。该实现保持 IVF 粗聚类和最终 float 精排在原始空间，只在 IVF residual 上训练一套全局正交旋转；base 和 query 在旋转空间进行局部 PQ 训练、编码与 LUT 扫描。比较 OPQ 收益时，除方法和 `opq-iters` 外应保持其他参数完全一致。

全局码本版本使用 `ivfpq-global-opq`。它保持 IVF 路由在原始空间，在原始训练向量上学习全局 OPQ，并用同一旋转后的 base/query 训练全局 PQ、编码和构造 LUT；它不会把现有 `ivfpq-global` 改成 residual PQ。应与 `ivfpq-global` 使用完全相同的 IVF/PQ 和精排参数进行对照。

#### HNSW

```text
--data-dir
/anndata
--method
hnsw
--parallel
none
--queries
2000
--k
10
--hnsw-m
16
--ef-construction
150
--ef-search
64
--hnsw-layout
porder
--gorder-window
5
--porder-profile-queries
200
--warmup-queries
100
--output
files/workflow_results.csv
```

建图参数固定后依次改变 `ef-search`，即可得到 HNSW 的 latency-recall 曲线。将
`hnsw-layout` 分别设为 `original`、`rcm`、`gorder` 和 `porder` 可对比四种布局；
重排耗时计入索引构建而不计入查询延时。Gorder/Porder 默认使用窗口大小 `5`。
Porder 按“画像、重排、预热、正式评测”执行；`--queries` 是读取的总查询数，正式评测
数量等于 `queries - porder-profile-queries - warmup-queries`。对照实验必须使用相同的
正式评测查询区间：例如 Porder 使用画像 `200`、预热 `100` 时，original、RCM 和
Gorder 应设置 `--evaluation-query-offset 300`。正式对比时索引构建时间和查询时间要
分开报告。

## 5. MPI 集群工作流

MPI 实验不使用 `test.sh`。当前程序已兼容“手动 `mpic++` 编译 + `qsub_mpi.sh` 提交”的流程，但编译时必须定义 `ANN_ENABLE_MPI`，否则程序会按普通单机版本构建。

### 5.1 编译和提交

在服务器的 `ann` 目录执行：

```bash
mpic++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11 -DANN_ENABLE_MPI
qsub qsub_mpi.sh
```

不要直接执行 `main` 或 `mpiexec`。`qsub_mpi.sh` 会把 `main` 和 `files/` 分发到 PBS 分配的节点，再调用 `/usr/local/bin/mpiexec`。结果仍由 rank 0 写入 `files/`，作业结束后同步回主节点的 `ann/files/`。

### 6.2 nodes、ppn、NP 和线程数

在 `qsub_mpi.sh` 顶部同时配置：

```sh
#PBS -l nodes=1:ppn=8
NP=8
```

- `nodes <= 4`；
- `ppn <= 8`；
- `NP <= nodes * ppn`；
- 纯 MPI：`NP = nodes * ppn`，并在 `workflow.conf` 中设置 `--threads 1`；
- MPI + OpenMP：`threads = nodes * ppn / NP`，该值必须与 `workflow.conf` 中的 `--threads` 相同。

脚本会从 `$PBS_NODEFILE` 检查节点数、ppn、NP 和线程数，不满足约束时在启动 MPI 前报错。

纯 MPI、8 个进程：

```sh
#PBS -l nodes=1:ppn=8
NP=8
# workflow.conf: --threads 1
```

8 个 MPI 进程、每进程 2 个 OpenMP 线程：

```sh
#PBS -l nodes=2:ppn=8
NP=8
# workflow.conf: --threads 2
```

### 6.3 MPI 方法

| `--method` | 划分/路由方式 | 主要参数 |
| --- | --- | --- |
| `mpi-ivfpq` | IVF list 按 rank 分片，局部 PQ 扫描后归并 | `nlist`、`nprobe`、`m`、`ks`、`rerank-p`、`local-p`、`split` |
| `mpi-ivf-hnsw` | IVF 路由，每个分片使用 HNSW | IVF 参数、HNSW 参数、`local-p`、`split` |
| `mpi-random-hnsw` | 底库随机分片，每个 rank 建 HNSW | HNSW 参数、`local-p`、`seed` |
| `mpi-kmeans-hnsw` | KMeans 分片，每个 rank 建 HNSW | HNSW 参数、训练参数、`local-p` |
| `mpi-hoh` | HNSW-on-HNSW 两层路由 | HNSW 参数、`route-p`、`local-p` |

`--split cyclic` 通常能让 list 数量分布更均衡；`block` 用于对比分块分片。MPI + OpenMP 的表级并行目前主要用于 `mpi-ivfpq`，配置 `--parallel list` 和大于 1 的 `--threads`。

### 6.4 MPI-IVFPQ 配置示例

纯 MPI 版本的 `files/workflow.conf`：

```text
--data-dir
/anndata
--method
mpi-ivfpq
--parallel
none
--threads
1
--queries
2000
--k
10
--nlist
100
--nprobe
16
--m
16
--ks
256
--train-size
10000
--kmeans-iters
6
--rerank-p
200
--local-p
50
--split
cyclic
--output
files/mpi_results.csv
```

混合并行时只需改为：

```text
--parallel
list
--threads
2
--local-p
200
```

同时把 `qsub_mpi.sh` 设置为例如 `nodes=2, ppn=8, NP=8`。为避免局部阶段过早丢失候选，正确性对比时建议 `local-p >= rerank-p`。

## 7. GPU 工作流

Windows 下需要 CUDA Toolkit 和 MSVC host compiler。建议从 Visual Studio Developer PowerShell 执行：

```powershell
.\scripts\build_gpu.ps1 -Nvcc nvcc
.\scripts\run_gpu.ps1 -DataDir 'D:\data\ann' -ResultsDir '.\results\gpu'
```

如果 `cl.exe` 不在 PATH 中，可显式指定：

```powershell
.\scripts\build_gpu.ps1 -Nvcc 'D:\CUDA\bin\nvcc.exe' -HostCompiler 'C:\path\to\cl.exe'
```

构建会生成 `flat_kernel`、`flat_cublas`、`ivf_kernel`、`ivf_cublas` 和 `ivf_grouped` 五个版本，批量运行日志保存在结果目录中。CUDA 源码现在从第一个命令行参数读取数据目录，未提供时仍使用原来的默认目录。
