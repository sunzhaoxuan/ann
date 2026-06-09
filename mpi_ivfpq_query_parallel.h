#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "ivf_pq_local_scan_simd.h"

struct MPIQueryCandidate {
    float dist;
    uint32_t id;
};

struct MPIQueryParallelResult {
    float recall;
    double latency_us;

    MPIQueryParallelResult()
        : recall(0.0f), latency_us(0.0) {}

    MPIQueryParallelResult(float r, double l)
        : recall(r), latency_us(l) {}
};

struct MPIQueryParallelProfile {
    int batch_count;
    int query_count;

    double local_compute_us;
    double local_compute_max_us;
    double gather_us;
    double merge_us;
    double total_us;

    MPIQueryParallelProfile()
        : batch_count(0),
          query_count(0),
          local_compute_us(0.0),
          local_compute_max_us(0.0),
          gather_us(0.0),
          merge_us(0.0),
          total_us(0.0) {}

    void add_batch(
        int bsz,
        double local_compute,
        double local_compute_max,
        double gather,
        double merge,
        double total
    ) {
        batch_count += 1;
        query_count += bsz;

        local_compute_us += local_compute;
        local_compute_max_us += local_compute_max;
        gather_us += gather;
        merge_us += merge;
        total_us += total;
    }

    void print_average() const {
        if (query_count == 0) {
            return;
        }

        double inv = 1.0 / static_cast<double>(query_count);

        std::cout << "\nMPI + OpenMP query-level profile average per query (us):\n";
        std::cout << "  Local compute(rank0) : " << local_compute_us * inv << "\n";
        std::cout << "  Local compute max    : " << local_compute_max_us * inv << "\n";
        std::cout << "  Gather               : " << gather_us * inv << "\n";
        std::cout << "  Merge                : " << merge_us * inv << "\n";
        std::cout << "  Total                : " << total_us * inv << "\n";
        std::cout << "  Batch count          : " << batch_count << "\n";
        std::cout << "  Query count          : " << query_count << "\n";
    }
};

static inline void pack_priority_queue_to_candidates(
    std::priority_queue<std::pair<float, uint32_t> > local_pq,
    size_t local_p,
    MPIQueryCandidate* out
) {
    const float INF = std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < local_p; ++i) {
        out[i].dist = INF;
        out[i].id = UINT32_MAX;
    }

    size_t idx = 0;

    while (!local_pq.empty() && idx < local_p) {
        out[idx].dist = local_pq.top().first;
        out[idx].id = local_pq.top().second;

        local_pq.pop();
        ++idx;
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
merge_one_query_topk_from_gathered_candidates(
    const std::vector<MPIQueryCandidate>& gathered,
    int world_size,
    int batch_size,
    int query_offset_in_batch,
    size_t local_p,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (int r = 0; r < world_size; ++r) {
        size_t base =
            (static_cast<size_t>(r) * static_cast<size_t>(batch_size)
             + static_cast<size_t>(query_offset_in_batch)) * local_p;

        for (size_t j = 0; j < local_p; ++j) {
            const MPIQueryCandidate& cand = gathered[base + j];

            if (cand.id == UINT32_MAX) {
                continue;
            }

            if (global_topk.size() < k) {
                global_topk.push({cand.dist, cand.id});
            } else if (cand.dist < global_topk.top().first) {
                global_topk.pop();
                global_topk.push({cand.dist, cand.id});
            }
        }
    }

    return global_topk;
}

static inline float compute_recall_at_k_from_result(
    std::priority_queue<std::pair<float, uint32_t> > res,
    const int* test_gt,
    int test_gt_d,
    int qid,
    size_t k
) {
    std::set<uint32_t> gtset;

    for (size_t j = 0; j < k; ++j) {
        int t = test_gt[j + qid * test_gt_d];
        gtset.insert(static_cast<uint32_t>(t));
    }

    size_t acc = 0;

    while (!res.empty()) {
        uint32_t id = res.top().second;

        if (gtset.find(id) != gtset.end()) {
            ++acc;
        }

        res.pop();
    }

    return static_cast<float>(acc) / static_cast<float>(k);
}

static inline std::vector<MPIQueryParallelResult>
mpi_ivfpq_query_parallel_search_all(
    const IVFPQLocalIndexSIMD& index,
    float* test_query,
    int test_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    const int* test_gt,
    int test_gt_d,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    int omp_threads,
    int batch_size,
    MPIQueryParallelProfile* out_profile,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    std::vector<MPIQueryParallelResult> results;

    if (rank == 0) {
        results.resize(test_number);
    }

    if (omp_threads <= 0) {
        omp_threads = omp_get_max_threads();
    }

    if (batch_size <= 0) {
        batch_size = 32;
    }

    local_p = std::max(local_p, k);

    if (rank == 0 && out_profile != nullptr) {
        *out_profile = MPIQueryParallelProfile();
    }

    MPI_Barrier(comm);

    for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
        int cur_batch_size =
            std::min(batch_size, test_number - batch_start);

        double batch_total_begin = MPI_Wtime();

        std::vector<MPIQueryCandidate> local_batch_candidates(
            static_cast<size_t>(cur_batch_size) * local_p
        );

        // ====================================================
        // 1. 进程内 Query 级并行
        //    每个 rank 内部用 OpenMP 并行处理 batch 内的多条 query。
        //    注意：这里没有 MPI 调用。
        // ====================================================
        double local_begin = MPI_Wtime();

        omp_set_dynamic(0);

        #pragma omp parallel for schedule(dynamic) num_threads(omp_threads)
        for (int bi = 0; bi < cur_batch_size; ++bi) {
            int qid = batch_start + bi;

            float* query =
                test_query + static_cast<size_t>(qid) * vecdim;

            auto local_pq = index.search_mpi_local(
                query,
                k,
                nprobe,
                rerank_p,
                local_p,
                rank,
                world_size,
                split_mode,
                nullptr
            );

            MPIQueryCandidate* out =
                local_batch_candidates.data()
                + static_cast<size_t>(bi) * local_p;

            pack_priority_queue_to_candidates(
                local_pq,
                local_p,
                out
            );
        }

        double local_end = MPI_Wtime();

        double local_compute_us =
            (local_end - local_begin) * 1000000.0;

        double local_compute_max_us = 0.0;

        MPI_Reduce(
            &local_compute_us,
            &local_compute_max_us,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            comm
        );

        // ====================================================
        // 2. 一次性 Gather 整个 batch 的 local Top-p
        //    使用 MPI_BYTE 传结构体，避免单独 Gather dist 和 id。
        // ====================================================
        std::vector<MPIQueryCandidate> gathered_candidates;

        if (rank == 0) {
            gathered_candidates.resize(
                static_cast<size_t>(world_size)
                * static_cast<size_t>(cur_batch_size)
                * local_p
            );
        }

        int send_bytes =
            static_cast<int>(
                static_cast<size_t>(cur_batch_size)
                * local_p
                * sizeof(MPIQueryCandidate)
            );

        double gather_begin = MPI_Wtime();

        MPI_Gather(
            local_batch_candidates.data(),
            send_bytes,
            MPI_BYTE,
            rank == 0 ? gathered_candidates.data() : nullptr,
            send_bytes,
            MPI_BYTE,
            0,
            comm
        );

        double gather_end = MPI_Wtime();

        double gather_us =
            (gather_end - gather_begin) * 1000000.0;

        // ====================================================
        // 3. rank 0 对 batch 内每条 query 做 global Top-k merge
        // ====================================================
        double merge_us = 0.0;

        if (rank == 0) {
            double merge_begin = MPI_Wtime();

            for (int bi = 0; bi < cur_batch_size; ++bi) {
                int qid = batch_start + bi;

                auto global_topk =
                    merge_one_query_topk_from_gathered_candidates(
                        gathered_candidates,
                        world_size,
                        cur_batch_size,
                        bi,
                        local_p,
                        k
                    );

                float recall = compute_recall_at_k_from_result(
                    global_topk,
                    test_gt,
                    test_gt_d,
                    qid,
                    k
                );

                results[qid].recall = recall;
            }

            double merge_end = MPI_Wtime();
            merge_us = (merge_end - merge_begin) * 1000000.0;
        }

        double batch_total_end = MPI_Wtime();

        double batch_total_us =
            (batch_total_end - batch_total_begin) * 1000000.0;

        // ====================================================
        // 4. rank 0 记录 batch 平均摊销 latency
        //    注意：这里的 latency 是 batch throughput 意义下的
        //    amortized latency，不是单条 query 串行延时。
        // ====================================================
        if (rank == 0) {
            double per_query_latency =
                batch_total_us / static_cast<double>(cur_batch_size);

            for (int bi = 0; bi < cur_batch_size; ++bi) {
                int qid = batch_start + bi;
                results[qid].latency_us = per_query_latency;
            }

            if (out_profile != nullptr) {
                out_profile->add_batch(
                    cur_batch_size,
                    local_compute_us,
                    local_compute_max_us,
                    gather_us,
                    merge_us,
                    batch_total_us
                );
            }
        }
    }

    MPI_Barrier(comm);

    return results;
}