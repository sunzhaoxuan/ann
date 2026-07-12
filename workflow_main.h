#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "flat_scan.h"
#include "flat_scan_simd.h"
#include "sq_scan_simd.h"
#include "sq_scan_simd_int8.h"
#include "pq_simd_scan.h"
#include "ivf_scan_simd.h"
#include "ivf_pq_scan_simd.h"
#include "ivf_pq_local_scan_simd.h"
#include "hnswlib/hnswlib/hnswlib.h"

#ifdef ANN_ENABLE_MPI
#include <mpi.h>
#include "mpi_ivfpq_local_search.h"
#include "ivf_hnsw_mpi.h"
#include "random_hnsw_mpi.h"
#include "kmeans_hnsw_mpi.h"
#include "hnsw_on_hnsw_mpi.h"
#endif

namespace {

using SearchQueue = std::priority_queue<std::pair<float, uint32_t> >;

struct Config {
    std::string method = "flat";
    std::string parallel = "none";
    std::string data_dir = "./data";
    std::string base_file = "DEEP100K.base.100k.fbin";
    std::string query_file = "DEEP100K.query.fbin";
    std::string ground_truth_file = "DEEP100K.gt.query.100k.top100.bin";
    std::string output_file;
    std::string split = "cyclic";

    size_t query_count = 2000;
    size_t k = 10;
    int threads = 1;

    size_t nlist = 100;
    size_t nprobe = 16;
    size_t pq_m = 16;
    size_t pq_ks = 256;
    size_t train_size = 10000;
    size_t kmeans_iters = 6;
    size_t opq_iters = 2;
    size_t rerank_p = 200;
    size_t local_p = 50;

    int hnsw_m = 16;
    int ef_construction = 150;
    int ef_search = 64;
    std::string hnsw_layout = "original";
    size_t gorder_window = 5;
    size_t porder_profile_queries = 200;
    size_t warmup_queries = 100;
    size_t evaluation_query_offset = 0;
    size_t route_p = 4;
    uint32_t seed = 2026;

    bool dry_run = false;
};

struct Dataset {
    std::vector<float> base;
    std::vector<float> queries;
    std::vector<int> ground_truth;
    size_t base_count = 0;
    size_t query_count = 0;
    size_t dimension = 0;
    size_t ground_truth_width = 0;
};

struct Metrics {
    double recall = 0.0;
    double build_ms = 0.0;
    double average_latency_us = 0.0;
    size_t queries = 0;
};

static std::string join_path(const std::string& directory, const std::string& file) {
    if (file.empty()) return directory;
    if (file.size() >= 2 && file[1] == ':') return file;
    if (file[0] == '/' || file[0] == '\\') return file;
    if (directory.empty() || directory == ".") return file;
    const char last = directory[directory.size() - 1];
    if (last == '/' || last == '\\') return directory + file;
#ifdef _WIN32
    return directory + "\\" + file;
#else
    return directory + "/" + file;
#endif
}

template <typename T>
static std::vector<T> load_matrix(
    const std::string& path,
    size_t& rows,
    size_t& columns
) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open data file: " + path);
    }

    uint32_t row_header = 0;
    uint32_t column_header = 0;
    input.read(reinterpret_cast<char*>(&row_header), sizeof(row_header));
    input.read(reinterpret_cast<char*>(&column_header), sizeof(column_header));
    if (!input || row_header == 0 || column_header == 0) {
        throw std::runtime_error("invalid matrix header: " + path);
    }

    rows = static_cast<size_t>(row_header);
    columns = static_cast<size_t>(column_header);
    if (rows > std::numeric_limits<size_t>::max() / columns) {
        throw std::runtime_error("matrix size overflow: " + path);
    }

    std::vector<T> values(rows * columns);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))
    );
    if (!input) {
        throw std::runtime_error("truncated matrix payload: " + path);
    }

    return values;
}

static Dataset load_dataset(const Config& config) {
    Dataset data;
    size_t base_dim = 0;
    size_t query_dim = 0;
    size_t gt_rows = 0;

    data.base = load_matrix<float>(
        join_path(config.data_dir, config.base_file),
        data.base_count,
        base_dim
    );
    data.queries = load_matrix<float>(
        join_path(config.data_dir, config.query_file),
        data.query_count,
        query_dim
    );
    data.ground_truth = load_matrix<int>(
        join_path(config.data_dir, config.ground_truth_file),
        gt_rows,
        data.ground_truth_width
    );

    if (base_dim != query_dim) {
        throw std::runtime_error("base/query dimensions do not match");
    }
    if (gt_rows < data.query_count) {
        throw std::runtime_error("ground truth has fewer rows than the query set");
    }
    if (config.k > data.ground_truth_width) {
        throw std::runtime_error("k exceeds the ground-truth width");
    }

    data.dimension = base_dim;
    data.query_count = std::min(data.query_count, config.query_count);
    if (config.hnsw_layout == "porder"
        && config.porder_profile_queries + config.warmup_queries >= data.query_count) {
        throw std::runtime_error(
            "Porder profile + warmup queries must leave at least one evaluation query");
    }
    if (config.evaluation_query_offset >= data.query_count) {
        throw std::runtime_error(
            "evaluation-query-offset must be smaller than the loaded query count");
    }
    return data;
}

static double recall_at_k(
    SearchQueue result,
    const Dataset& data,
    size_t query_id,
    size_t k
) {
    std::set<uint32_t> truth;
    const size_t offset = query_id * data.ground_truth_width;
    for (size_t i = 0; i < k; ++i) {
        truth.insert(static_cast<uint32_t>(data.ground_truth[offset + i]));
    }

    size_t hits = 0;
    while (!result.empty()) {
        if (truth.find(result.top().second) != truth.end()) ++hits;
        result.pop();
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

template <typename SearchFunction>
static Metrics evaluate_queries(
    const Config& config,
    const Dataset& data,
    double build_ms,
    SearchFunction search,
    size_t query_begin = 0
) {
    if (query_begin >= data.query_count) {
        throw std::runtime_error("evaluation query range is empty");
    }
    const size_t evaluation_count = data.query_count - query_begin;
    std::vector<SearchQueue> results(evaluation_count);
    const auto begin = std::chrono::steady_clock::now();

    if (config.parallel == "query" && config.threads > 1) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(config.threads) schedule(static)
        for (long long index = 0; index < static_cast<long long>(evaluation_count); ++index) {
            const size_t qid = query_begin + static_cast<size_t>(index);
            results[static_cast<size_t>(index)] = search(qid);
        }
    } else {
        for (size_t index = 0; index < evaluation_count; ++index) {
            results[index] = search(query_begin + index);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(end - begin).count();

    double recall_sum = 0.0;
    for (size_t index = 0; index < evaluation_count; ++index) {
        recall_sum += recall_at_k(
            results[index], data, query_begin + index, config.k);
    }

    Metrics metrics;
    metrics.recall = recall_sum / static_cast<double>(evaluation_count);
    metrics.build_ms = build_ms;
    metrics.average_latency_us = elapsed_us / static_cast<double>(evaluation_count);
    metrics.queries = evaluation_count;
    return metrics;
}

static SearchQueue flat_base_parallel(
    float* base,
    float* query,
    size_t base_count,
    size_t dimension,
    size_t k,
    int threads
) {
    if (threads <= 1 || base_count < static_cast<size_t>(threads)) {
        return flat_search_simd_fasttopk(base, query, base_count, dimension, k);
    }

    std::vector<SearchQueue> local(static_cast<size_t>(threads));
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const size_t chunk = (base_count + static_cast<size_t>(threads) - 1)
                           / static_cast<size_t>(threads);
        const size_t begin = std::min(base_count, static_cast<size_t>(tid) * chunk);
        const size_t end = std::min(base_count, begin + chunk);
        SearchQueue& queue = local[static_cast<size_t>(tid)];

        for (size_t id = begin; id < end; ++id) {
            const float distance = 1.0f - inner_product_neon16_fma(
                base + id * dimension,
                query,
                dimension
            );
            if (queue.size() < k) {
                queue.push({distance, static_cast<uint32_t>(id)});
            } else if (distance < queue.top().first) {
                queue.pop();
                queue.push({distance, static_cast<uint32_t>(id)});
            }
        }
    }

    SearchQueue result;
    for (SearchQueue& queue : local) {
        while (!queue.empty()) {
            const auto candidate = queue.top();
            queue.pop();
            if (result.size() < k) {
                result.push(candidate);
            } else if (candidate.first < result.top().first) {
                result.pop();
                result.push(candidate);
            }
        }
    }
    return result;
}

static SearchQueue convert_hnsw_result(
    std::priority_queue<std::pair<float, hnswlib::labeltype> > source
) {
    SearchQueue result;
    while (!source.empty()) {
        result.push({source.top().first, static_cast<uint32_t>(source.top().second)});
        source.pop();
    }
    return result;
}

static Metrics run_serial_method(const Config& config, const Dataset& data) {
    float* base = const_cast<float*>(data.base.data());
    const auto query_ptr = [&](size_t qid) {
        return const_cast<float*>(data.queries.data() + qid * data.dimension);
    };

    if (config.method == "flat-scalar") {
        return evaluate_queries(config, data, 0.0, [&](size_t qid) {
            return flat_search(
                base, query_ptr(qid), data.base_count, data.dimension, config.k
            );
        });
    }

    if (config.method == "flat") {
        return evaluate_queries(config, data, 0.0, [&](size_t qid) {
            if (config.parallel == "list") {
                return flat_base_parallel(
                    base, query_ptr(qid), data.base_count, data.dimension,
                    config.k, config.threads
                );
            }
            return flat_search_simd_fasttopk(
                base, query_ptr(qid), data.base_count, data.dimension, config.k
            );
        });
    }

    if (config.method == "sq" || config.method == "sq-int8") {
        const auto build_begin = std::chrono::steady_clock::now();
        if (config.method == "sq") {
            SQIndexSIMD index(base, data.base_count, data.dimension);
            const auto build_end = std::chrono::steady_clock::now();
            const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
            return evaluate_queries(config, data, build_ms, [&](size_t qid) {
                return index.search(query_ptr(qid), config.k, config.rerank_p);
            });
        }

        SQIndexSIMDInt8 index(base, data.base_count, data.dimension);
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            return index.search(query_ptr(qid), config.k, config.rerank_p);
        });
    }

    if (config.method == "pq") {
        const auto build_begin = std::chrono::steady_clock::now();
        PQIndexSIMD index(
            base, data.base_count, data.dimension,
            config.pq_m, config.pq_ks, config.train_size,
            config.kmeans_iters, PQInitMode::KMeansPP
        );
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            if (config.parallel == "list") {
                return index.search_omp_scan(
                    query_ptr(qid), config.k, config.rerank_p,
                    config.local_p, config.threads
                );
            }
            return index.search(query_ptr(qid), config.k, config.rerank_p);
        });
    }

    if (config.method == "ivf") {
        const auto build_begin = std::chrono::steady_clock::now();
        IVFIndexSIMD index(
            base, data.base_count, data.dimension,
            config.nlist, config.train_size, config.kmeans_iters,
            IVFInitMode::KMeansPP
        );
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            if (config.parallel == "list") {
                return index.search_omp_list_parallel(
                    query_ptr(qid), config.k, config.nprobe,
                    config.local_p, config.threads
                );
            }
            return index.search(query_ptr(qid), config.k, config.nprobe);
        });
    }

    if (config.method == "ivfpq-global" || config.method == "ivfpq-global-opq") {
        const auto build_begin = std::chrono::steady_clock::now();
        const size_t opq_iters = config.method == "ivfpq-global-opq"
            ? config.opq_iters
            : 0;
        IVFPQIndexSIMD index(
            base, data.base_count, data.dimension,
            config.nlist, config.pq_m, config.pq_ks,
            config.train_size, config.kmeans_iters,
            IVFPQInitMode::Uniform, opq_iters
        );
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            return index.search(
                query_ptr(qid), config.k, config.nprobe, config.rerank_p
            );
        });
    }

    if (config.method == "ivfpq-local" || config.method == "ivfpq-local-opq") {
        const auto build_begin = std::chrono::steady_clock::now();
        const size_t opq_iters = config.method == "ivfpq-local-opq"
            ? config.opq_iters
            : 0;
        IVFPQLocalIndexSIMD index(
            base, data.base_count, data.dimension,
            config.nlist, config.pq_m, config.pq_ks,
            config.train_size, config.kmeans_iters,
            IVFPQLocalInitMode::Uniform, opq_iters
        );
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            if (config.parallel == "list") {
                return index.search_omp_list_parallel(
                    query_ptr(qid), config.k, config.nprobe,
                    config.rerank_p, config.local_p, config.threads
                );
            }
            return index.search(
                query_ptr(qid), config.k, config.nprobe, config.rerank_p
            );
        });
    }

    if (config.method == "hnsw") {
        const auto build_begin = std::chrono::steady_clock::now();
        hnswlib::InnerProductSpace space(static_cast<int>(data.dimension));
        hnswlib::HierarchicalNSW<float> index(
            &space,
            data.base_count,
            config.hnsw_m,
            config.ef_construction
        );
        index.addPoint(base, 0);
        for (size_t i = 1; i < data.base_count; ++i) {
            index.addPoint(base + i * data.dimension, static_cast<hnswlib::labeltype>(i));
        }
        index.setEf(config.ef_search);
        if (config.hnsw_layout == "rcm") {
            const size_t moved = index.reorderIndexRCM();
            std::cout << "RCM moved " << moved << " of "
                      << data.base_count << " HNSW nodes\n";
        } else if (config.hnsw_layout == "gorder") {
            const size_t moved = index.reorderIndexGorder(config.gorder_window);
            std::cout << "Gorder moved " << moved << " of "
                      << data.base_count << " HNSW nodes\n";
        } else if (config.hnsw_layout == "porder") {
            index.startEdgeProfiling();
            for (size_t qid = 0; qid < config.porder_profile_queries; ++qid) {
                (void) index.searchKnnProfiled(query_ptr(qid), config.k);
            }
            const uint64_t traversals = index.getProfiledEdgeTraversals();
            const size_t moved = index.reorderIndexPorder(config.gorder_window);
            std::cout << "Porder profiled " << traversals
                      << " edge traversals using "
                      << config.porder_profile_queries << " queries\n"
                      << "Porder moved " << moved << " of "
                      << data.base_count << " HNSW nodes\n";

            const size_t warmup_end =
                config.porder_profile_queries + config.warmup_queries;
            for (size_t qid = config.porder_profile_queries;
                 qid < warmup_end; ++qid) {
                (void) index.searchKnn(query_ptr(qid), config.k);
            }
        }
        const auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
        const size_t query_begin = config.hnsw_layout == "porder"
            ? config.porder_profile_queries + config.warmup_queries
            : config.evaluation_query_offset;
        return evaluate_queries(config, data, build_ms, [&](size_t qid) {
            return convert_hnsw_result(index.searchKnn(query_ptr(qid), config.k));
        }, query_begin);
    }

    throw std::runtime_error("method requires an MPI build or is unknown: " + config.method);
}

#ifdef ANN_ENABLE_MPI

template <typename SearchFunction>
static Metrics evaluate_mpi_queries(
    const Config& config,
    const Dataset& data,
    double local_build_ms,
    int rank,
    MPI_Comm communicator,
    SearchFunction search,
    size_t query_begin = 0
) {
    if (query_begin >= data.query_count) {
        throw std::runtime_error("MPI evaluation query range is empty");
    }
    const size_t evaluation_count = data.query_count - query_begin;
    double build_ms = 0.0;
    MPI_Reduce(&local_build_ms, &build_ms, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    double recall_sum = 0.0;
    double latency_sum = 0.0;
    MPI_Barrier(communicator);
    for (size_t qid = query_begin; qid < data.query_count; ++qid) {
        double latency_us = 0.0;
        SearchQueue result = search(qid, &latency_us);
        if (rank == 0) {
            recall_sum += recall_at_k(result, data, qid, config.k);
            latency_sum += latency_us;
        }
    }
    MPI_Barrier(communicator);

    Metrics metrics;
    if (rank == 0) {
        metrics.recall = recall_sum / static_cast<double>(evaluation_count);
        metrics.build_ms = build_ms;
        metrics.average_latency_us = latency_sum / static_cast<double>(evaluation_count);
        metrics.queries = evaluation_count;
    }
    return metrics;
}

static Metrics run_mpi_method(
    const Config& config,
    const Dataset& data,
    int rank,
    int world_size,
    MPI_Comm communicator
) {
    float* base = const_cast<float*>(data.base.data());
    const auto query_ptr = [&](size_t qid) {
        return const_cast<float*>(data.queries.data() + qid * data.dimension);
    };
    const IVFPQMPISplitMode pq_split = config.split == "block"
        ? IVFPQMPISplitMode::Block
        : IVFPQMPISplitMode::Cyclic;
    const IVFHNSWMPISplitMode hnsw_split = config.split == "block"
        ? IVFHNSWMPISplitMode::Block
        : IVFHNSWMPISplitMode::Cyclic;

    if (config.method == "mpi-ivfpq") {
        const auto begin = std::chrono::steady_clock::now();
        IVFPQLocalIndexSIMD index(
            base, data.base_count, data.dimension,
            config.nlist, config.pq_m, config.pq_ks,
            config.train_size, config.kmeans_iters,
            IVFPQLocalInitMode::Uniform
        );
        index.keep_only_mpi_local_lists(rank, world_size, pq_split);
        const auto end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return evaluate_mpi_queries(config, data, build_ms, rank, communicator,
            [&](size_t qid, double* latency) {
                if (config.parallel == "list" && config.threads > 1) {
                    return mpi_ivfpq_local_search_one_omp_list_no_bcast_light(
                        index, query_ptr(qid), config.k, config.nprobe,
                        config.rerank_p, config.local_p, rank, world_size,
                        pq_split, config.threads, communicator, latency
                    );
                }
                return mpi_ivfpq_local_search_one_no_bcast_light(
                    index, query_ptr(qid), config.k, config.nprobe,
                    config.rerank_p, config.local_p, rank, world_size,
                    pq_split, communicator, latency
                );
            }
        );
    }

    if (config.method == "mpi-ivf-hnsw") {
        const auto begin = std::chrono::steady_clock::now();
        IVF_HNSW_MPI_Index index(
            base, data.base_count, data.dimension,
            config.nlist, config.train_size, config.kmeans_iters,
            config.hnsw_m, config.ef_construction, config.ef_search,
            rank, world_size, hnsw_split
        );
        const auto end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return evaluate_mpi_queries(config, data, build_ms, rank, communicator,
            [&](size_t qid, double* latency) {
                return mpi_ivf_hnsw_search_one_no_bcast_light(
                    index, query_ptr(qid), config.k, config.nprobe,
                    config.local_p, rank, world_size, communicator, latency
                );
            }
        );
    }

    if (config.method == "mpi-random-hnsw") {
        const auto begin = std::chrono::steady_clock::now();
        RandomHNSWMPIIndex index(
            base, data.base_count, data.dimension,
            rank, world_size, config.hnsw_m,
            config.ef_construction, config.ef_search, config.seed
        );
        const auto end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return evaluate_mpi_queries(config, data, build_ms, rank, communicator,
            [&](size_t qid, double* latency) {
                return mpi_random_hnsw_search_one_no_bcast_light(
                    index, query_ptr(qid), config.k, config.local_p,
                    rank, world_size, communicator, latency
                );
            }
        );
    }

    if (config.method == "mpi-kmeans-hnsw") {
        const auto begin = std::chrono::steady_clock::now();
        KMeansHNSWMPIIndex index(
            base, data.base_count, data.dimension,
            rank, world_size, config.hnsw_m,
            config.ef_construction, config.ef_search,
            config.train_size, static_cast<int>(config.kmeans_iters)
        );
        const auto end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return evaluate_mpi_queries(config, data, build_ms, rank, communicator,
            [&](size_t qid, double* latency) {
                return mpi_kmeans_hnsw_search_one_no_bcast_light(
                    index, query_ptr(qid), config.k, config.local_p,
                    rank, world_size, communicator, latency
                );
            }
        );
    }

    if (config.method == "mpi-hoh") {
        const auto begin = std::chrono::steady_clock::now();
        HNSWOnHNSWMPIIndex index(
            base, data.base_count, data.dimension,
            rank, world_size,
            config.hnsw_m, config.ef_construction, config.ef_search,
            8, 100, 16,
            config.train_size, static_cast<int>(config.kmeans_iters),
            config.hnsw_layout, config.gorder_window
        );
        if (config.hnsw_layout == "porder") {
            index.start_local_edge_profiling();
            for (size_t qid = 0; qid < config.porder_profile_queries; ++qid) {
                double ignored_latency = 0.0;
                (void) mpi_hnsw_on_hnsw_search_one_no_bcast_light(
                    index, base, query_ptr(qid), data.dimension,
                    config.k, config.local_p, config.route_p,
                    rank, world_size, communicator, &ignored_latency);
            }
            const unsigned long long local_traversals =
                static_cast<unsigned long long>(
                    index.local_profiled_edge_traversals());
            index.finish_local_porder(config.gorder_window);

            unsigned long long total_traversals = 0;
            MPI_Reduce(
                &local_traversals, &total_traversals, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator);
            if (rank == 0) {
                std::cout << "Porder profiled " << total_traversals
                          << " lower-HNSW edge traversals using "
                          << config.porder_profile_queries << " queries\n";
            }
            MPI_Barrier(communicator);

            const size_t warmup_end =
                config.porder_profile_queries + config.warmup_queries;
            for (size_t qid = config.porder_profile_queries;
                 qid < warmup_end; ++qid) {
                double ignored_latency = 0.0;
                (void) mpi_hnsw_on_hnsw_search_one_no_bcast_light(
                    index, base, query_ptr(qid), data.dimension,
                    config.k, config.local_p, config.route_p,
                    rank, world_size, communicator, &ignored_latency);
            }
        }
        if (config.hnsw_layout != "original") {
            const unsigned long long local_moved =
                static_cast<unsigned long long>(index.reordered_moved_count());
            unsigned long long total_moved = 0;
            MPI_Reduce(
                &local_moved, &total_moved, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator);
            if (rank == 0) {
                std::cout << config.hnsw_layout << " moved " << total_moved
                          << " lower-HNSW nodes across all ranks\n";
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        const size_t query_begin = config.hnsw_layout == "porder"
            ? config.porder_profile_queries + config.warmup_queries
            : config.evaluation_query_offset;
        return evaluate_mpi_queries(config, data, build_ms, rank, communicator,
            [&](size_t qid, double* latency) {
                return mpi_hnsw_on_hnsw_search_one_no_bcast_light(
                    index, base, query_ptr(qid), data.dimension,
                    config.k, config.local_p, config.route_p,
                    rank, world_size, communicator, latency
                );
            }, query_begin
        );
    }

    return run_serial_method(config, data);
}

#endif

static size_t parse_size(const std::string& text, const std::string& option) {
    std::istringstream input(text);
    unsigned long long value = 0;
    input >> value;
    if (!input || !input.eof()) throw std::runtime_error("invalid value for " + option + ": " + text);
    return static_cast<size_t>(value);
}

static int parse_int(const std::string& text, const std::string& option) {
    std::istringstream input(text);
    int value = 0;
    input >> value;
    if (!input || !input.eof()) throw std::runtime_error("invalid value for " + option + ": " + text);
    return value;
}

static void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Core options:\n"
        << "  --method NAME       flat-scalar, flat, sq, sq-int8, pq, ivf,\n"
        << "                      ivfpq-global, ivfpq-global-opq,\n"
        << "                      ivfpq-local, ivfpq-local-opq, hnsw,\n"
        << "                      mpi-ivfpq, mpi-ivf-hnsw,\n"
        << "                      mpi-random-hnsw, mpi-kmeans-hnsw, mpi-hoh\n"
        << "  --data-dir PATH     directory containing the three DEEP100K files\n"
        << "  --queries N         number of queries to evaluate (default 2000)\n"
        << "  --k N               result count (default 10)\n"
        << "  --parallel MODE     none, query, or list\n"
        << "  --threads N         OpenMP thread count\n"
        << "  --output FILE       append one CSV result row\n\n"
        << "Index/search options:\n"
        << "  --nlist N --nprobe N --m N --ks N\n"
        << "  --train-size N --kmeans-iters N --opq-iters N\n"
        << "  --rerank-p N --local-p N\n"
        << "  --hnsw-m N --ef-construction N --ef-search N\n"
        << "  --hnsw-layout original|rcm|gorder|porder --gorder-window N\n"
        << "  --porder-profile-queries N --warmup-queries N\n"
        << "  --evaluation-query-offset N\n"
        << "  --route-p N --split cyclic|block --seed N\n\n"
        << "File overrides:\n"
        << "  --base-file FILE --query-file FILE --ground-truth-file FILE\n"
        << "  --dry-run           validate arguments and data only\n";
}

static Config parse_arguments(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--help" || option == "-h") {
            print_help(argv[0]);
            std::exit(0);
        }
        if (option == "--dry-run") {
            config.dry_run = true;
            continue;
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value(argv[++i]);

        if (option == "--method") config.method = value;
        else if (option == "--parallel") config.parallel = value;
        else if (option == "--data-dir") config.data_dir = value;
        else if (option == "--base-file") config.base_file = value;
        else if (option == "--query-file") config.query_file = value;
        else if (option == "--ground-truth-file") config.ground_truth_file = value;
        else if (option == "--output") config.output_file = value;
        else if (option == "--split") config.split = value;
        else if (option == "--queries") config.query_count = parse_size(value, option);
        else if (option == "--k") config.k = parse_size(value, option);
        else if (option == "--threads") config.threads = parse_int(value, option);
        else if (option == "--nlist") config.nlist = parse_size(value, option);
        else if (option == "--nprobe") config.nprobe = parse_size(value, option);
        else if (option == "--m") config.pq_m = parse_size(value, option);
        else if (option == "--ks") config.pq_ks = parse_size(value, option);
        else if (option == "--train-size") config.train_size = parse_size(value, option);
        else if (option == "--kmeans-iters") config.kmeans_iters = parse_size(value, option);
        else if (option == "--opq-iters") config.opq_iters = parse_size(value, option);
        else if (option == "--rerank-p") config.rerank_p = parse_size(value, option);
        else if (option == "--local-p") config.local_p = parse_size(value, option);
        else if (option == "--hnsw-m") config.hnsw_m = parse_int(value, option);
        else if (option == "--ef-construction") config.ef_construction = parse_int(value, option);
        else if (option == "--ef-search") config.ef_search = parse_int(value, option);
        else if (option == "--hnsw-layout") config.hnsw_layout = value;
        else if (option == "--gorder-window") config.gorder_window = parse_size(value, option);
        else if (option == "--porder-profile-queries") config.porder_profile_queries = parse_size(value, option);
        else if (option == "--warmup-queries") config.warmup_queries = parse_size(value, option);
        else if (option == "--evaluation-query-offset") config.evaluation_query_offset = parse_size(value, option);
        else if (option == "--route-p") config.route_p = parse_size(value, option);
        else if (option == "--seed") config.seed = static_cast<uint32_t>(parse_size(value, option));
        else throw std::runtime_error("unknown option: " + option);
    }

    if (config.k == 0 || config.query_count == 0) throw std::runtime_error("k and queries must be positive");
    if (config.threads < 1) throw std::runtime_error("threads must be positive");
    if (config.parallel != "none" && config.parallel != "query" && config.parallel != "list") {
        throw std::runtime_error("parallel must be none, query, or list");
    }
    if (config.split != "cyclic" && config.split != "block") {
        throw std::runtime_error("split must be cyclic or block");
    }
    if (config.hnsw_layout != "original" && config.hnsw_layout != "rcm"
        && config.hnsw_layout != "gorder" && config.hnsw_layout != "porder") {
        throw std::runtime_error(
            "hnsw-layout must be original, rcm, gorder, or porder");
    }
    if (config.gorder_window == 0) {
        throw std::runtime_error("gorder-window must be positive");
    }
    if (config.hnsw_layout == "porder" && config.porder_profile_queries == 0) {
        throw std::runtime_error("porder-profile-queries must be positive");
    }
    if (config.hnsw_layout != "original"
        && config.method != "hnsw" && config.method != "mpi-hoh") {
        throw std::runtime_error(
            "hnsw-layout reordering is supported only for hnsw and mpi-hoh");
    }
    return config;
}

static void print_configuration(const Config& config, const Dataset& data, int rank, int world_size) {
    if (rank != 0) return;
    std::cout << "========== ANN experiment workflow ==========\n"
              << "method: " << config.method << "\n"
              << "parallel: " << config.parallel << "\n"
              << "threads: " << config.threads << "\n"
              << "MPI ranks: " << world_size << "\n"
              << "base/query/dim: " << data.base_count << "/"
              << data.query_count << "/" << data.dimension << "\n"
              << "k/nlist/nprobe: " << config.k << "/"
              << config.nlist << "/" << config.nprobe << "\n"
              << "M/Ks/rerank/local: " << config.pq_m << "/"
              << config.pq_ks << "/" << config.rerank_p << "/"
              << config.local_p << "\n"
              << "OPQ iterations: "
              << ((config.method == "ivfpq-local-opq"
                   || config.method == "ivfpq-global-opq") ? config.opq_iters : 0)
              << "\n"
              << "HNSW layout: " << config.hnsw_layout << "\n";
    if (config.hnsw_layout == "gorder") {
        std::cout << "Gorder window: " << config.gorder_window << "\n";
    } else if (config.hnsw_layout == "porder") {
        std::cout << "Porder window/profile/warmup: "
                  << config.gorder_window << "/"
                  << config.porder_profile_queries << "/"
                  << config.warmup_queries << "\n";
    }
    if (config.evaluation_query_offset != 0) {
        std::cout << "Evaluation query offset: "
                  << config.evaluation_query_offset << "\n";
    }
}

static void print_metrics(const Config& config, const Metrics& metrics) {
    std::cout << std::fixed << std::setprecision(6)
              << "evaluated queries: " << metrics.queries << "\n"
              << "average recall@" << config.k << ": " << metrics.recall << "\n"
              << "index build (ms): " << metrics.build_ms << "\n"
              << "average latency (us): " << metrics.average_latency_us << "\n";
    if (config.parallel == "query") {
        std::cout << "latency meaning: batch wall time / query count (throughput-amortized)\n";
    }
}

static void append_csv(const Config& config, const Dataset& data, const Metrics& metrics) {
    if (config.output_file.empty()) return;
    bool write_header = false;
    {
        std::ifstream existing(config.output_file.c_str(), std::ios::binary | std::ios::ate);
        write_header = !existing || existing.tellg() == 0;
    }
    std::ofstream output(config.output_file.c_str(), std::ios::app);
    if (!output) throw std::runtime_error("cannot open output CSV: " + config.output_file);
    if (write_header) {
        output << "method,parallel,threads,queries,base_count,dimension,k,nlist,nprobe,m,ks,"
                  "opq_iters,rerank_p,local_p,recall,build_ms,average_latency_us\n";
    }
    const bool applies_hnsw_layout =
        config.method == "hnsw" || config.method == "mpi-hoh";
    std::string method_label = config.method;
    if (applies_hnsw_layout && config.hnsw_layout != "original") {
        method_label += "-" + config.hnsw_layout;
        if (config.hnsw_layout == "gorder") {
            method_label += "-w" + std::to_string(config.gorder_window);
        } else if (config.hnsw_layout == "porder") {
            method_label += "-w" + std::to_string(config.gorder_window)
                + "-p" + std::to_string(config.porder_profile_queries);
        }
    }
    output << method_label << ',' << config.parallel << ',' << config.threads << ','
           << metrics.queries << ',' << data.base_count << ',' << data.dimension << ','
           << config.k << ',' << config.nlist << ',' << config.nprobe << ','
           << config.pq_m << ',' << config.pq_ks << ','
           << ((config.method == "ivfpq-local-opq"
                || config.method == "ivfpq-global-opq") ? config.opq_iters : 0) << ','
           << config.rerank_p << ','
           << config.local_p << ',' << std::setprecision(10) << metrics.recall << ','
           << metrics.build_ms << ',' << metrics.average_latency_us << '\n';
}

}  // namespace

inline int ann_workflow_main(int argc, char** argv) {
    int rank = 0;
    int world_size = 1;
    int mpi_thread_support = 0;

#ifdef ANN_ENABLE_MPI
    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &mpi_thread_support
    );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    int exit_code = 0;
    try {
        const Config config = parse_arguments(argc, argv);
#ifdef ANN_ENABLE_MPI
        if (config.threads > 1 && mpi_thread_support < MPI_THREAD_FUNNELED) {
            throw std::runtime_error(
                "MPI implementation does not provide MPI_THREAD_FUNNELED"
            );
        }
#endif
        const Dataset data = load_dataset(config);
        print_configuration(config, data, rank, world_size);

        if (config.dry_run) {
            if (rank == 0) std::cout << "data validation: OK\n";
        } else {
            Metrics metrics;
#ifdef ANN_ENABLE_MPI
            metrics = run_mpi_method(config, data, rank, world_size, MPI_COMM_WORLD);
#else
            if (config.method.compare(0, 4, "mpi-") == 0) {
                throw std::runtime_error("this method requires the MPI build");
            }
            metrics = run_serial_method(config, data);
#endif
            if (rank == 0) {
                print_metrics(config, metrics);
                append_csv(config, data, metrics);
            }
        }
    } catch (const std::exception& error) {
        if (rank == 0) std::cerr << "error: " << error.what() << '\n';
        exit_code = 1;
    }

#ifdef ANN_ENABLE_MPI
    MPI_Finalize();
#endif
    return exit_code;
}
