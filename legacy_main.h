#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "flat_scan_simd.h"
#include "sq_scan_simd.h"
#include "sq_scan_simd_int8.h"
#include "pq_scan_simd.h"
#include "ivf_scan_simd.h"
#include "ivf_pq_scan_simd.h"
#include "ivf_pq_local_scan_simd.h"
#include "mpi_ivfpq_local_search.h"
#include "mpi_ivfpq_query_parallel.h"
#include "ivf_hnsw_mpi.h"
#include "random_hnsw_mpi.h"
#include "kmeans_hnsw_mpi.h"
#include "hnsw_on_hnsw_mpi.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

void build_hnsw_index(
    float* base,
    size_t base_number,
    size_t vecdim,
    const std::string& index_path
) {
    const int M = 16;
    const int efConstruction = 150;

    hnswlib::InnerProductSpace ipspace(vecdim);

    hnswlib::HierarchicalNSW<float> hnsw_index(
        &ipspace,
        base_number,
        M,
        efConstruction
    );

    // 第一个点单独插入，避免并行插入时空索引初始化出问题
    hnsw_index.addPoint(base, 0);

    #pragma omp parallel for
    for (int i = 1; i < static_cast<int>(base_number); ++i) {
        hnsw_index.addPoint(
            base + 1ll * i * vecdim,
            static_cast<hnswlib::labeltype>(i)
        );
    }

    hnsw_index.saveIndex(index_path);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    IVFPQMPISplitMode split_mode = IVFPQMPISplitMode::Cyclic;
    IVFHNSWMPISplitMode hnsw_split_mode = IVFHNSWMPISplitMode::Cyclic;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        std::cout << "MPI world_size = "
                  << world_size
                  << std::endl;
        std::cout << "omp_get_max_threads = "
            << omp_get_max_threads()
            << std::endl;
    }

    bool test_query_parallel = false;
    int64_t total_diff = 0;

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    //build_index(base, base_number, vecdim);
    //build_hnsw_index(base,base_number,vecdim,"files/hnsw.index");

    //SQIndexSIMD sq_index(base, base_number, vecdim);
    //SQIndexSIMDInt8 sq_index(base, base_number, vecdim);

    // PQIndexSIMD pq_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     16,      // M: 子空间数量 4,8,12,16 特殊优化(循环展开)
    //     256,      // Ks: 每个子空间的聚类中心数量
    //     20000,   // train_size: 训练样本数量
    //     8,        // kmeans_iters: KMeans 迭代轮数
    //     PQInitMode::KMeansPP  // init_mode: 初始化方式 KMeansPP or Uniform
    // );

    // IVFIndexSIMD ivf_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     100,      // nlist
    //     10000,    // train_size
    //     6,        // kmeans_iters
    //     IVFInitMode::KMeansPP
    // );

    // IVFPQIndexSIMD ivfpq_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     100,      // nlist
    //     16,       // M
    //     256,      // Ks
    //     10000,    // train_size
    //     6         // kmeans_iters
    // );

    //先ivf后pq
    // IVFPQLocalIndexSIMD ivfpq_local_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     100,      // nlist
    //     16,       // M
    //     256,      // Ks
    //     10000,    // train_size
    //     6         // kmeans_iters
    // );

    // ivfpq_local_index.keep_only_mpi_local_lists(
    //     rank,
    //     world_size,
    //     split_mode
    // );

    // MPISearchProfileTotal total_mpi_prof;

    // IVF_HNSW_MPI_Index ivf_hnsw_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     100,      // nlist
    //     10000,    // train_size
    //     6,        // kmeans_iters
    //     16,       // hnsw_M
    //     150,      // hnsw_ef_construction
    //     200,       // hnsw_ef_search
    //     rank,
    //     world_size,
    //     hnsw_split_mode
    // );

    // RandomHNSWMPIIndex random_hnsw_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     rank,
    //     world_size,
    //     16,      // hnsw_M
    //     200,     // hnsw_ef_construction
    //     32,      // hnsw_ef_search
    //     2026     // random split seed
    // );

    // KMeansHNSWMPIIndex kmeans_hnsw_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     rank,
    //     world_size,
    //     16,       // hnsw_M
    //     200,      // hnsw_ef_construction
    //     32,       // hnsw_ef_search
    //     10000,    // train_size
    //     6         // kmeans_iters
    // );

    HNSWOnHNSWMPIIndex hoh_index(
        base,
        base_number,
        vecdim,
        rank,
        world_size,
        16,       // lower_hnsw_M
        200,      // lower_ef_construction
        32,       // lower_ef_search
        8,        // upper_hnsw_M
        100,      // upper_ef_construction
        16,       // upper_ef_search
        10000,    // train_size
        6         // kmeans_iters
    );

    MPI_Barrier(MPI_COMM_WORLD);

    // hnswlib::InnerProductSpace ipspace(vecdim);

    // hnswlib::HierarchicalNSW<float> hnsw_index(
    //     &ipspace,
    //     "files/hnsw.index"
    // );

    // // efSearch 控制查询质量和延时
    // hnsw_index.setEf(64);

    // PQSearchProfile total_prof;
    // total_prof.clear();

    //查询测试代码
    // for(int i = 0; i < test_number; ++i) {
    //     const unsigned long Converter = 1000 * 1000;

    //     const size_t rerank_p = 200;
    //     const size_t local_p = 50;
    //     const int num_threads = 8;

    //     const size_t nprobe = 8;

    //     struct timeval val;
    //     int ret = gettimeofday(&val, NULL);

    //     // 该文件已有代码中你只能修改该函数的调用方式
    //     // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。

    //     //hnsw
    //     //auto res = hnsw_index.searchKnn(test_query + i * vecdim,k);

    //     //hnsw多入口点并行
    //     //auto res = hnsw_index.searchKnnMultiEntryParallel(test_query + i * vecdim,k,2,64);

    //     //auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

    //     //auto res = flat_search_simd(base, test_query + i*vecdim, base_number, vecdim, k);

    //     //flat_simd优化版
    //     //auto res = flat_search_simd_fasttopk(base, test_query + i * vecdim, base_number, vecdim, k);

    //     //sq_simd
    //     //auto res = sq_index.search(test_query + i * vecdim, k, rerank_p);

    //     //pq_simd
    //     //auto res = pq_index.search(test_query + i * vecdim , k , rerank_p);

    //     //pq_simd OpenMP
    //     //auto res = pq_index.search_omp_scan(test_query + i * vecdim,k,rerank_p,local_p, num_threads);

    //     //pq_simd pthread
    //     //auto res = pq_index.search_pthread_scan(test_query + i * vecdim,k,rerank_p,local_p,num_threads);

    //     //ivf_simd
    //     //auto res = ivf_index.search(test_query + i * vecdim,k,nprobe);

    //     //ivf openmd list级并行
    //     // auto res = ivf_index.search_omp_list_parallel(test_query + i * vecdim,k,
    //     //     16,    // nprobe
    //     //     10,    // local_p
    //     //     8      // num_threads
    //     // );

    //     //ivf pthread list级并行
    //     // auto res = ivf_index.search_pthread_list_parallel(test_query + i * vecdim,k,
    //     //     16,    // nprobe
    //     //     10,    // local_p
    //     //     8      // num_threads
    //     // );

    //     //ivf openmp centroid级并行
    //     // auto res = ivf_index.search_omp_centroid_parallel(test_query + i * vecdim,k,
    //     //     32,    // nprobe
    //     //     4      // num_threads
    //     // );

    //     //IVF-PQ-SIMD baseline
    //     //先pq，再ivf
    //     //auto res = ivfpq_index.search(test_query + i * vecdim,k,nprobe,rerank_p);

    //     //IVF-PQ-SIMD baseline
    //     //先IVF，再pq
    //     //auto res = ivfpq_local_index.search(test_query + i * vecdim,k,nprobe,rerank_p);

    //     //ivfpq list级并行 OpenMP
    //     //auto res = ivfpq_local_index.search_omp_list_parallel(test_query + i * vecdim,k,nprobe,rerank_p,local_p,num_threads);

    //     //ivfpq list级并行 pthread
    //     //auto res = ivfpq_local_index.search_pthread_list_parallel(test_query + i * vecdim,k,nprobe,rerank_p,local_p,num_threads);

    //     //pq_simd profiling 版本，增加每个阶段的耗时统计
    //     // PQSearchProfile prof;
    //     // auto res = pq_index.search(
    //     //     test_query + i * vecdim,
    //     //     k,
    //     //     rerank_p,
    //     //     &prof
    //     // );
    //     // total_prof.add(prof);

    //     //Flat-SIMD pthread base 划分并行
    //     // auto res = flat_search_simd_pthread_base_parallel(base,test_query + i * vecdim,base_number,vecdim,k,
    //     //     10,  // top p
    //     //     6   // num_threads
    //     // );

    //     // OpenMP base 划分
    //     // auto res = flat_search_simd_omp_base_parallel(base,test_query + i * vecdim,base_number,vecdim,k,
    //     //     10,  // top p
    //     //     6   // num_threads
    //     // );


    //     struct timeval newVal;
    //     ret = gettimeofday(&newVal, NULL);
    //     int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    //     std::set<uint32_t> gtset;
    //     for(int j = 0; j < k; ++j){
    //         int t = test_gt[j + i*test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;
    //     while (res.size()) {
    //         int x = res.top().second;
    //         if(gtset.find(x) != gtset.end()){
    //             ++acc;
    //         }
    //         res.pop();
    //     }
    //     float recall = (float)acc/k;

    //     results[i] = {recall, diff};
    // }

    //查询测试代码：MPI
    // for (int i = 0; i < test_number; ++i) {
    //     const unsigned long Converter = 1000 * 1000;

    //     const size_t rerank_p = 200;
    //     const size_t local_p = 100;
    //     const size_t nprobe = 8;

    //     struct timeval val;
    //     struct timeval newVal;

    //     if (rank == 0) {
    //         gettimeofday(&val, NULL);
    //     }

    //     // ========================================================
    //     // IVF-PQ-SIMD + MPI
    //     // 先 IVF，再簇内 PQ；
    //     // 每个 MPI 进程只搜索自己负责的 inverted lists；
    //     // rank 0 收集 local Top-p 后 merge 成 global Top-k。
    //     // ========================================================
    //     auto res = mpi_ivfpq_local_search_one(
    //         ivfpq_local_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         vecdim,
    //         k,
    //         nprobe,
    //         rerank_p,
    //         local_p,
    //         rank,
    //         world_size,
    //         split_mode,
    //         MPI_COMM_WORLD
    //     );

    //     if (rank == 0) {
    //         gettimeofday(&newVal, NULL);

    //         int64_t diff =
    //             (newVal.tv_sec * Converter + newVal.tv_usec)
    //             - (val.tv_sec * Converter + val.tv_usec);

    //         std::set<uint32_t> gtset;

    //         for (int j = 0; j < k; ++j) {
    //             int t = test_gt[j + i * test_gt_d];
    //             gtset.insert(static_cast<uint32_t>(t));
    //         }

    //         size_t acc = 0;

    //         while (!res.empty()) {
    //             uint32_t x = res.top().second;

    //             if (gtset.find(x) != gtset.end()) {
    //                 ++acc;
    //             }

    //             res.pop();
    //         }

    //         float recall = static_cast<float>(acc) / static_cast<float>(k);

    //         results[i] = {recall, diff};
    //     }
    // }

    //mpi_profiling
    // for (int i = 0; i < test_number; ++i) {
    //     const size_t rerank_p = 200;
    //     const size_t local_p = 20;
    //     const size_t nprobe = 16;

    //     const int omp_threads = omp_get_max_threads();

    //     MPISearchProfileOne one_prof;

    //     // auto res = mpi_ivfpq_local_search_one_profile(
    //     //     ivfpq_local_index,
    //     //     test_query + static_cast<size_t>(i) * vecdim,
    //     //     vecdim,
    //     //     k,
    //     //     nprobe,
    //     //     rerank_p,
    //     //     local_p,
    //     //     rank,
    //     //     world_size,
    //     //     split_mode,
    //     //     MPI_COMM_WORLD,
    //     //     &one_prof
    //     // );

    //     // auto res = mpi_ivfpq_local_search_one_omp_list_profile(
    //     //     ivfpq_local_index,
    //     //     test_query + static_cast<size_t>(i) * vecdim,
    //     //     vecdim,
    //     //     k,
    //     //     nprobe,
    //     //     rerank_p,
    //     //     local_p,
    //     //     rank,
    //     //     world_size,
    //     //     split_mode,
    //     //     omp_threads,
    //     //     MPI_COMM_WORLD,
    //     //     &one_prof
    //     // );

    //     // auto res = mpi_ivfpq_local_search_one_no_bcast_profile(
    //     //     ivfpq_local_index,
    //     //     test_query + static_cast<size_t>(i) * vecdim,
    //     //     vecdim,
    //     //     k,
    //     //     nprobe,
    //     //     rerank_p,
    //     //     local_p,
    //     //     rank,
    //     //     world_size,
    //     //     IVFPQMPISplitMode::Cyclic,
    //     //     MPI_COMM_WORLD,
    //     //     &one_prof
    //     // );

    //     auto res = mpi_ivfpq_local_search_one_omp_list_no_bcast_profile(
    //         ivfpq_local_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         vecdim,
    //         k,
    //         nprobe,
    //         rerank_p,
    //         local_p,
    //         rank,
    //         world_size,
    //         split_mode,
    //         omp_threads,
    //         MPI_COMM_WORLD,
    //         &one_prof
    //     );

    //     if (rank == 0) {
    //         std::set<uint32_t> gtset;

    //         for (int j = 0; j < k; ++j) {
    //             int t = test_gt[j + i * test_gt_d];
    //             gtset.insert(static_cast<uint32_t>(t));
    //         }

    //         size_t acc = 0;

    //         while (!res.empty()) {
    //             uint32_t x = res.top().second;

    //             if (gtset.find(x) != gtset.end()) {
    //                 ++acc;
    //             }

    //             res.pop();
    //         }

    //         float recall =
    //             static_cast<float>(acc) / static_cast<float>(k);

    //         results[i] = {
    //             recall,
    //             static_cast<int64_t>(one_prof.total_us)
    //         };

    //         total_mpi_prof.add(one_prof);
    //     }
    // }

    //查询测试代码：批量 query 级并行
    // test_query_parallel = true;
    // std::vector<std::priority_queue<std::pair<float, uint32_t> > > all_res;
    // all_res.resize(test_number);

    // const unsigned long Converter = 1000 * 1000;

    // struct timeval val;
    // gettimeofday(&val, NULL);

    // // pthread
    // // flat_search_simd_pthread_query_parallel(
    // //     base,
    // //     test_query,
    // //     base_number,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     8,          // 线程数
    // //     all_res
    // // );

    // // openmp
    // // flat_search_simd_omp_query_parallel(
    // //     base,
    // //     test_query,
    // //     base_number,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     1,
    // //     all_res
    // // );

    // //ivf_pthread
    // // ivf_search_simd_pthread_query_parallel(
    // //     ivf_index,
    // //     test_query,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     16,      // nprobe
    // //     1,       // num_threads
    // //     all_res
    // // );

    // //ivf_openmp
    // // ivf_search_simd_omp_query_parallel(
    // //     ivf_index,
    // //     test_query,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     64,      // nprobe
    // //     8,       // num_threads
    // //     all_res
    // // );

    // //ivfpq openmp
    // // ivfpq_local_search_omp_query_parallel(
    // //     ivfpq_local_index,
    // //     test_query,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     64,      // nprobe
    // //     200,     // rerank_p
    // //     8,       // num_threads
    // //     all_res
    // // );

    // //ivfpq pthread
    // // ivfpq_local_search_pthread_query_parallel(
    // //     ivfpq_local_index,
    // //     test_query,
    // //     test_number,
    // //     vecdim,
    // //     k,
    // //     64,      // nprobe
    // //     200,     // rerank_p
    // //     8,       // num_threads
    // //     all_res
    // // );

    // struct timeval newVal;
    // gettimeofday(&newVal, NULL);

    // total_diff =
    //     (newVal.tv_sec * Converter + newVal.tv_usec)
    //     - (val.tv_sec * Converter + val.tv_usec);

    // int64_t avg_query_latency = total_diff / static_cast<int64_t>(test_number);

    // // 逐条 query 计算 recall
    // for (int i = 0; i < test_number; ++i) {
    //     std::set<uint32_t> gtset;

    //     for (int j = 0; j < k; ++j) {
    //         int t = test_gt[j + i * test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;

    //     // 拷贝一份，因为 priority_queue pop 后会被破坏
    //     auto res = all_res[i];

    //     while (!res.empty()) {
    //         int x = res.top().second;

    //         if (gtset.find(x) != gtset.end()) {
    //             ++acc;
    //         }

    //         res.pop();
    //     }

    //     float recall = static_cast<float>(acc) / k;

    //     results[i] = {recall, avg_query_latency};
    // }

    // 查询测试代码：mpi内query级并行
    // const size_t rerank_p = 200;
    // const size_t local_p = 20;
    // const size_t nprobe = 16;

    // const int omp_threads = omp_get_max_threads();

    // // batch_size 越大，MPI_Gather 次数越少；
    // // 但 batch 太大时 rank 0 合并候选的内存和 merge 压力会增加。
    // // 建议先测 32 或 64。
    // const int batch_size = 32;

    // MPIQueryParallelProfile qp_prof;

    // auto qp_results = mpi_ivfpq_query_parallel_search_all(
    //     ivfpq_local_index,
    //     test_query,
    //     static_cast<int>(test_number),
    //     vecdim,
    //     k,
    //     nprobe,
    //     rerank_p,
    //     local_p,
    //     test_gt,
    //     static_cast<int>(test_gt_d),
    //     rank,
    //     world_size,
    //     split_mode,
    //     omp_threads,
    //     batch_size,
    //     &qp_prof,
    //     MPI_COMM_WORLD
    // );

    // if (rank == 0) {
    //     double avg_recall = 0.0;
    //     double avg_latency = 0.0;

    //     for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //         avg_recall += qp_results[i].recall;
    //         avg_latency += qp_results[i].latency_us;
    //     }

    //     avg_recall /= static_cast<double>(test_number);
    //     avg_latency /= static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";

    //     qp_prof.print_average();
    // }

    // const size_t rerank_p = 200;
    // const size_t local_p = 20;
    // const size_t nprobe = 16;

    // double total_recall_sum = 0.0;
    // double total_latency_sum = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     double latency_us = 0.0;

    //     const int omp_threads = omp_get_max_threads();

    //     // auto res = mpi_ivfpq_local_search_one_omp_list_no_bcast_light(
    //     //     ivfpq_local_index,
    //     //     test_query + static_cast<size_t>(i) * vecdim,
    //     //     k,
    //     //     nprobe,
    //     //     rerank_p,
    //     //     local_p,
    //     //     rank,
    //     //     world_size,
    //     //     split_mode,
    //     //     omp_threads,
    //     //     MPI_COMM_WORLD,
    //     //     &latency_us
    //     // );

    //     // auto res = mpi_ivfpq_local_search_one_omp_list_no_bcast_candidate_gather(
    //     //     ivfpq_local_index,
    //     //     test_query + static_cast<size_t>(i) * vecdim,
    //     //     k,
    //     //     nprobe,
    //     //     rerank_p,
    //     //     local_p,
    //     //     rank,
    //     //     world_size,
    //     //     split_mode,
    //     //     omp_threads,
    //     //     MPI_COMM_WORLD,
    //     //     &latency_us
    //     // );

    //     if (rank == 0) {
    //         std::set<uint32_t> gtset;

    //         for (int j = 0; j < static_cast<int>(k); ++j) {
    //             int t = test_gt[j + i * test_gt_d];
    //             gtset.insert(static_cast<uint32_t>(t));
    //         }

    //         size_t acc = 0;

    //         while (!res.empty()) {
    //             uint32_t x = res.top().second;

    //             if (gtset.find(x) != gtset.end()) {
    //                 ++acc;
    //             }

    //             res.pop();
    //         }

    //         float recall =
    //             static_cast<float>(acc) / static_cast<float>(k);

    //         total_recall_sum += static_cast<double>(recall);
    //         total_latency_sum += latency_us;
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall =
    //         total_recall_sum / static_cast<double>(test_number);

    //     double avg_latency =
    //         total_latency_sum / static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";
    // }


    // const size_t nprobe = 16;
    // const size_t local_p = 20;

    // IVFHNSWMPISearchProfileTotal total_hnsw_prof;

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     IVFHNSWMPISearchProfileOne one_prof;

    //     auto res = mpi_ivf_hnsw_search_one_no_bcast_profile(
    //         ivf_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         nprobe,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &one_prof
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_ivfhnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         results[i] = {
    //             recall,
    //             static_cast<int64_t>(one_prof.total_us)
    //         };

    //         total_hnsw_prof.add(one_prof);
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall = 0.0;
    //     double avg_latency = 0.0;

    //     for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //         avg_recall += results[i].recall;
    //         avg_latency += results[i].latency;
    //     }

    //     avg_recall /= static_cast<double>(test_number);
    //     avg_latency /= static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";

    //     total_hnsw_prof.print_average();
    // }


    // const size_t local_p = 10;

    // RandomHNSWProfileTotal total_random_hnsw_prof;

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     RandomHNSWProfileOne one_prof;

    //     auto res = mpi_random_hnsw_search_one_no_bcast_profile(
    //         random_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &one_prof
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_random_hnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         results[i].recall = recall;
    //         results[i].latency = static_cast<int64_t>(one_prof.total_us);

    //         total_random_hnsw_prof.add(one_prof);
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall = 0.0;
    //     double avg_latency = 0.0;

    //     for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //         avg_recall += results[i].recall;
    //         avg_latency += results[i].latency;
    //     }

    //     avg_recall /= static_cast<double>(test_number);
    //     avg_latency /= static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";

    //     total_random_hnsw_prof.print_average();
    // }

    // const size_t local_p = 10;

    // double total_recall_sum = 0.0;
    // double total_latency_sum = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     double latency_us = 0.0;

    //     auto res = mpi_kmeans_hnsw_search_one_no_bcast_light(
    //         kmeans_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &latency_us
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_kmeans_hnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         total_recall_sum += static_cast<double>(recall);
    //         total_latency_sum += latency_us;
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall =
    //         total_recall_sum / static_cast<double>(test_number);

    //     double avg_latency =
    //         total_latency_sum / static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";
    // }


    // const size_t local_p = 10;

    // KMeansHNSWProfileTotal total_kmeans_hnsw_prof;

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     KMeansHNSWProfileOne one_prof;

    //     auto res = mpi_kmeans_hnsw_search_one_no_bcast_profile(
    //         kmeans_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &one_prof
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_kmeans_hnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         results[i].recall = recall;
    //         results[i].latency = static_cast<int64_t>(one_prof.total_us);

    //         total_kmeans_hnsw_prof.add(one_prof);
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall = 0.0;
    //     double avg_latency = 0.0;

    //     for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //         avg_recall += results[i].recall;
    //         avg_latency += results[i].latency;
    //     }

    //     avg_recall /= static_cast<double>(test_number);
    //     avg_latency /= static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";

    //     total_kmeans_hnsw_prof.print_average();
    // }

    // const size_t local_p = 10;

    // // route_p 表示上层 HNSW 选择多少个分区参与搜索。
    // // route_p 越小 latency 越低，但 recall 可能下降。
    // // route_p = world_size 时退化为所有分区都搜索。
    // const size_t route_p = 4;

    // HNSWOnHNSWProfileTotal total_hoh_prof;

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     HNSWOnHNSWProfileOne one_prof;

    //     auto res = mpi_hnsw_on_hnsw_search_one_no_bcast_profile(
    //         hoh_index,
    //         base,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         vecdim,
    //         k,
    //         local_p,
    //         route_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &one_prof
    //     );

    //     if (rank == 0) {
    //         float recall = hoh_compute_recall_at_k(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         results[i].recall = recall;
    //         results[i].latency = static_cast<int64_t>(one_prof.total_us);

    //         total_hoh_prof.add(one_prof);
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall = 0.0;
    //     double avg_latency = 0.0;

    //     for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //         avg_recall += results[i].recall;
    //         avg_latency += results[i].latency;
    //     }

    //     avg_recall /= static_cast<double>(test_number);
    //     avg_latency /= static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";

    //     total_hoh_prof.print_average();
    // }


    // const size_t nprobe = 16;
    // const size_t local_p = 10;

    // double total_recall_sum = 0.0;
    // double total_latency_sum = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     double latency_us = 0.0;

    //     auto res = mpi_ivf_hnsw_search_one_no_bcast_light(
    //         ivf_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         nprobe,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &latency_us
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_ivfhnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         total_recall_sum += static_cast<double>(recall);
    //         total_latency_sum += latency_us;
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall =
    //         total_recall_sum / static_cast<double>(test_number);

    //     double avg_latency =
    //         total_latency_sum / static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";
    // }

    // const size_t local_p = 10;

    // double total_recall_sum = 0.0;
    // double total_latency_sum = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);

    // for (int i = 0; i < static_cast<int>(test_number); ++i) {
    //     double latency_us = 0.0;

    //     auto res = mpi_random_hnsw_search_one_no_bcast_light(
    //         random_hnsw_index,
    //         test_query + static_cast<size_t>(i) * vecdim,
    //         k,
    //         local_p,
    //         rank,
    //         world_size,
    //         MPI_COMM_WORLD,
    //         &latency_us
    //     );

    //     if (rank == 0) {
    //         float recall = compute_recall_at_k_random_hnsw(
    //             res,
    //             test_gt,
    //             static_cast<int>(test_gt_d),
    //             i,
    //             k
    //         );

    //         total_recall_sum += static_cast<double>(recall);
    //         total_latency_sum += latency_us;
    //     }
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // if (rank == 0) {
    //     double avg_recall =
    //         total_recall_sum / static_cast<double>(test_number);

    //     double avg_latency =
    //         total_latency_sum / static_cast<double>(test_number);

    //     std::cout << "average recall: "
    //             << avg_recall
    //             << "\n";

    //     std::cout << "average latency (us): "
    //             << avg_latency
    //             << "\n";
    // }

    const size_t local_p = 10;

    // route_p 表示上层 HNSW 选择多少个分区参与搜索。
    // route_p 越小，搜索分区越少，latency 越低，但 recall 可能下降。
    // route_p = world_size 时，退化为所有分区都搜索。
    const size_t route_p = 4;

    double total_recall_sum = 0.0;
    double total_latency_sum = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < static_cast<int>(test_number); ++i) {
        double latency_us = 0.0;

        auto res = mpi_hnsw_on_hnsw_search_one_no_bcast_light(
            hoh_index,
            base,
            test_query + static_cast<size_t>(i) * vecdim,
            vecdim,
            k,
            local_p,
            route_p,
            rank,
            world_size,
            MPI_COMM_WORLD,
            &latency_us
        );

        if (rank == 0) {
            float recall = hoh_compute_recall_at_k(
                res,
                test_gt,
                static_cast<int>(test_gt_d),
                i,
                k
            );

            total_recall_sum += static_cast<double>(recall);
            total_latency_sum += latency_us;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_recall =
            total_recall_sum / static_cast<double>(test_number);

        double avg_latency =
            total_latency_sum / static_cast<double>(test_number);

        std::cout << "average recall: "
                << avg_recall
                << "\n";

        std::cout << "average latency (us): "
                << avg_latency
                << "\n";
    }



    // double qn = static_cast<double>(test_number);

    // std::cout << "PQ profile average per query (us):\n";
    // std::cout << "  LUT build   : " << total_prof.lut_build_us / qn << "\n";
    // std::cout << "  LUT quant   : " << total_prof.lut_quant_us / qn << "\n";
    // std::cout << "  PQ scan     : " << total_prof.pq_scan_us / qn << "\n";
    // std::cout << "  rerank      : " << total_prof.rerank_us / qn << "\n";
    // std::cout << "  output      : " << total_prof.output_us / qn << "\n";
    // std::cout << "  total       : " << total_prof.total_us / qn << "\n";

    // MPI_Barrier(MPI_COMM_WORLD);

    // if(rank==0){

    //     float avg_recall = 0, avg_latency = 0;
    //     for(int i = 0; i < test_number; ++i) {
    //         avg_recall += results[i].recall;
    //         avg_latency += results[i].latency;
    //     }

    //     // 浮点误差可能导致一些精确算法平均recall不是1
    //     std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    //     if(test_query_parallel) {
    //         std::cout << "average latency (us): "<<static_cast<float>(total_diff) / test_number<<"\n";
    //     }else{
    //         std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    //         total_mpi_prof.print_average();
    //     }

    // }
    return 0;
}
