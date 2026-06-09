#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "ivf_pq_local_scan_simd.h"


struct MPISearchProfileOne {
    double bcast_us;

    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;

    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    unsigned long long owned_lists_sum;
    unsigned long long owned_lists_max;

    unsigned long long scanned_codes_sum;
    unsigned long long scanned_codes_max;

    unsigned long long returned_candidates_sum;
    unsigned long long returned_candidates_max;

    MPISearchProfileOne()
        : bcast_us(0.0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          owned_lists_sum(0),
          owned_lists_max(0),
          scanned_codes_sum(0),
          scanned_codes_max(0),
          returned_candidates_sum(0),
          returned_candidates_max(0) {}
};

struct MPISearchProfileTotal {
    int cnt;

    double bcast_us;
    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;
    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    double owned_lists_sum;
    double owned_lists_max;
    double scanned_codes_sum;
    double scanned_codes_max;
    double returned_candidates_sum;
    double returned_candidates_max;

    MPISearchProfileTotal()
        : cnt(0),
          bcast_us(0.0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          owned_lists_sum(0.0),
          owned_lists_max(0.0),
          scanned_codes_sum(0.0),
          scanned_codes_max(0.0),
          returned_candidates_sum(0.0),
          returned_candidates_max(0.0) {}

    void add(const MPISearchProfileOne& p) {
        cnt++;

        bcast_us += p.bcast_us;
        local_search_us_min += p.local_search_us_min;
        local_search_us_avg += p.local_search_us_avg;
        local_search_us_max += p.local_search_us_max;
        pack_us_max += p.pack_us_max;
        gather_us_max += p.gather_us_max;
        merge_us += p.merge_us;
        total_us += p.total_us;

        owned_lists_sum += static_cast<double>(p.owned_lists_sum);
        owned_lists_max += static_cast<double>(p.owned_lists_max);

        scanned_codes_sum += static_cast<double>(p.scanned_codes_sum);
        scanned_codes_max += static_cast<double>(p.scanned_codes_max);

        returned_candidates_sum += static_cast<double>(p.returned_candidates_sum);
        returned_candidates_max += static_cast<double>(p.returned_candidates_max);
    }

    void print_average() const {
        if (cnt == 0) {
            return;
        }

        double inv = 1.0 / static_cast<double>(cnt);

        std::cout << "\nMPI profile average per query (us):\n";
        std::cout << "  Bcast          : " << bcast_us * inv << "\n";
        std::cout << "  Local search min: " << local_search_us_min * inv << "\n";
        std::cout << "  Local search avg: " << local_search_us_avg * inv << "\n";
        std::cout << "  Local search max: " << local_search_us_max * inv << "\n";
        std::cout << "  Pack max       : " << pack_us_max * inv << "\n";
        std::cout << "  Gather max     : " << gather_us_max * inv << "\n";
        std::cout << "  Merge          : " << merge_us * inv << "\n";
        std::cout << "  Total          : " << total_us * inv << "\n";

        std::cout << "\nMPI workload average per query:\n";
        std::cout << "  owned lists sum: " << owned_lists_sum * inv << "\n";
        std::cout << "  owned lists max: " << owned_lists_max * inv << "\n";
        std::cout << "  scanned codes sum: " << scanned_codes_sum * inv << "\n";
        std::cout << "  scanned codes max: " << scanned_codes_max * inv << "\n";
        std::cout << "  returned candidates sum: " << returned_candidates_sum * inv << "\n";
        std::cout << "  returned candidates max: " << returned_candidates_max * inv << "\n";

        std::cout << "\nMPI load balance indicators:\n";
        std::cout << "  local search max/avg: "
                  << (local_search_us_max / local_search_us_avg)
                  << "\n";
        std::cout << "  scanned codes max/sum: "
                  << (scanned_codes_max / scanned_codes_sum)
                  << "\n";
    }
};

static inline void pack_local_top_p(
    std::priority_queue<std::pair<float, uint32_t> > local_pq,
    size_t local_p,
    std::vector<float>& local_dist,
    std::vector<uint32_t>& local_id
) {
    const float INF = std::numeric_limits<float>::infinity();

    local_dist.assign(local_p, INF);
    local_id.assign(local_p, UINT32_MAX);

    size_t idx = 0;

    while (!local_pq.empty() && idx < local_p) {
        local_dist[idx] = local_pq.top().first;
        local_id[idx] = local_pq.top().second;

        local_pq.pop();
        ++idx;
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
merge_global_top_k(
    const std::vector<float>& all_dist,
    const std::vector<uint32_t>& all_id,
    size_t total_candidates,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < total_candidates; ++i) {
        uint32_t id = all_id[i];

        if (id == UINT32_MAX) {
            continue;
        }

        float dist = all_dist[i];

        if (global_topk.size() < k) {
            global_topk.push({dist, id});
        } else if (dist < global_topk.top().first) {
            global_topk.pop();
            global_topk.push({dist, id});
        }
    }

    return global_topk;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_rank0,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode = IVFPQMPISplitMode::Cyclic,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    std::vector<float> query(vecdim);

    if (rank == 0) {
        std::memcpy(
            query.data(),
            query_on_rank0,
            vecdim * sizeof(float)
        );
    }

    MPI_Bcast(
        query.data(),
        static_cast<int>(vecdim),
        MPI_FLOAT,
        0,
        comm
    );

    auto local_pq = index.search_mpi_local(
        query.data(),
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode
    );

    std::vector<float> local_dist;
    std::vector<uint32_t> local_id;

    pack_local_top_p(
        local_pq,
        local_p,
        local_dist,
        local_id
    );

    std::vector<float> all_dist;
    std::vector<uint32_t> all_id;

    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(world_size) * local_p);
        all_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    MPI_Gather(
        local_dist.data(),
        static_cast<int>(local_p),
        MPI_FLOAT,
        rank == 0 ? all_dist.data() : nullptr,
        static_cast<int>(local_p),
        MPI_FLOAT,
        0,
        comm
    );

    MPI_Gather(
        local_id.data(),
        static_cast<int>(local_p),
        MPI_UINT32_T,
        rank == 0 ? all_id.data() : nullptr,
        static_cast<int>(local_p),
        MPI_UINT32_T,
        0,
        comm
    );

    if (rank == 0) {
        return merge_global_top_k(
            all_dist,
            all_id,
            static_cast<size_t>(world_size) * local_p,
            k
        );
    }

    return std::priority_queue<std::pair<float, uint32_t> >();
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_profile(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_rank0,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    MPI_Comm comm,
    MPISearchProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    std::vector<float> query(vecdim);

    if (rank == 0) {
        std::memcpy(
            query.data(),
            query_on_rank0,
            vecdim * sizeof(float)
        );
    }

    // ========================================================
    // 1. Query broadcast
    // ========================================================
    double t0 = MPI_Wtime();

    MPI_Bcast(
        query.data(),
        static_cast<int>(vecdim),
        MPI_FLOAT,
        0,
        comm
    );

    double t1 = MPI_Wtime();

    double bcast_local_us = (t1 - t0) * 1000000.0;
    double bcast_max_us = 0.0;

    MPI_Reduce(
        &bcast_local_us,
        &bcast_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 2. Local search
    // ========================================================
    IVFPQLocalMPIProfile local_work_profile;

    t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query.data(),
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        &local_work_profile
    );

    t1 = MPI_Wtime();

    double local_search_us = (t1 - t0) * 1000000.0;

    double local_search_min_us = 0.0;
    double local_search_max_us = 0.0;
    double local_search_sum_us = 0.0;

    MPI_Reduce(
        &local_search_us,
        &local_search_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    // workload 统计
    unsigned long long owned_lists_local =
        static_cast<unsigned long long>(local_work_profile.owned_probe_lists);

    unsigned long long scanned_codes_local =
        static_cast<unsigned long long>(local_work_profile.scanned_codes);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_work_profile.returned_candidates);

    unsigned long long owned_lists_sum = 0;
    unsigned long long owned_lists_max = 0;

    unsigned long long scanned_codes_sum = 0;
    unsigned long long scanned_codes_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 3. Pack local Top-p
    // ========================================================
    std::vector<float> local_dist;
    std::vector<uint32_t> local_id;

    t0 = MPI_Wtime();

    pack_local_top_p(
        local_pq,
        local_p,
        local_dist,
        local_id
    );

    t1 = MPI_Wtime();

    double pack_local_us = (t1 - t0) * 1000000.0;
    double pack_max_us = 0.0;

    MPI_Reduce(
        &pack_local_us,
        &pack_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 4. Gather local Top-p
    // ========================================================
    std::vector<float> all_dist;
    std::vector<uint32_t> all_id;

    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(world_size) * local_p);
        all_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_dist.data(),
        static_cast<int>(local_p),
        MPI_FLOAT,
        rank == 0 ? all_dist.data() : NULL,
        static_cast<int>(local_p),
        MPI_FLOAT,
        0,
        comm
    );

    MPI_Gather(
        local_id.data(),
        static_cast<int>(local_p),
        MPI_UINT32_T,
        rank == 0 ? all_id.data() : NULL,
        static_cast<int>(local_p),
        MPI_UINT32_T,
        0,
        comm
    );

    t1 = MPI_Wtime();

    double gather_local_us = (t1 - t0) * 1000000.0;
    double gather_max_us = 0.0;

    MPI_Reduce(
        &gather_local_us,
        &gather_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 5. Merge global Top-k on rank 0
    // ========================================================
    double merge_us = 0.0;

    if (rank == 0) {
        double merge_begin = MPI_Wtime();

        final_result = merge_global_top_k(
            all_dist,
            all_id,
            static_cast<size_t>(world_size) * local_p,
            k
        );

        double merge_end = MPI_Wtime();
        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && out_profile != NULL) {
        out_profile->bcast_us = bcast_max_us;

        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_max = local_search_max_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us = (total_end - total_begin) * 1000000.0;

        out_profile->owned_lists_sum = owned_lists_sum;
        out_profile->owned_lists_max = owned_lists_max;

        out_profile->scanned_codes_sum = scanned_codes_sum;
        out_profile->scanned_codes_max = scanned_codes_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_omp_list_profile(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_rank0,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    int omp_threads,
    MPI_Comm comm,
    MPISearchProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    std::vector<float> query(vecdim);

    if (rank == 0) {
        std::memcpy(
            query.data(),
            query_on_rank0,
            vecdim * sizeof(float)
        );
    }

    // ========================================================
    // 1. Query broadcast
    // ========================================================
    double t0 = MPI_Wtime();

    MPI_Bcast(
        query.data(),
        static_cast<int>(vecdim),
        MPI_FLOAT,
        0,
        comm
    );

    double t1 = MPI_Wtime();

    double bcast_local_us = (t1 - t0) * 1000000.0;
    double bcast_max_us = 0.0;

    MPI_Reduce(
        &bcast_local_us,
        &bcast_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 2. Local search with OpenMP list-level parallelism
    // ========================================================
    IVFPQLocalMPIProfile local_work_profile;

    t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local_omp_list_parallel(
        query.data(),
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        omp_threads,
        &local_work_profile
    );

    t1 = MPI_Wtime();

    double local_search_us = (t1 - t0) * 1000000.0;

    double local_search_min_us = 0.0;
    double local_search_max_us = 0.0;
    double local_search_sum_us = 0.0;

    MPI_Reduce(
        &local_search_us,
        &local_search_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    unsigned long long owned_lists_local =
        static_cast<unsigned long long>(local_work_profile.owned_probe_lists);

    unsigned long long scanned_codes_local =
        static_cast<unsigned long long>(local_work_profile.scanned_codes);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_work_profile.returned_candidates);

    unsigned long long owned_lists_sum = 0;
    unsigned long long owned_lists_max = 0;

    unsigned long long scanned_codes_sum = 0;
    unsigned long long scanned_codes_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 3. Pack local Top-p
    // ========================================================
    std::vector<float> local_dist;
    std::vector<uint32_t> local_id;

    t0 = MPI_Wtime();

    pack_local_top_p(
        local_pq,
        local_p,
        local_dist,
        local_id
    );

    t1 = MPI_Wtime();

    double pack_local_us = (t1 - t0) * 1000000.0;
    double pack_max_us = 0.0;

    MPI_Reduce(
        &pack_local_us,
        &pack_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 4. Gather local Top-p
    // ========================================================
    std::vector<float> all_dist;
    std::vector<uint32_t> all_id;

    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(world_size) * local_p);
        all_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_dist.data(),
        static_cast<int>(local_p),
        MPI_FLOAT,
        rank == 0 ? all_dist.data() : NULL,
        static_cast<int>(local_p),
        MPI_FLOAT,
        0,
        comm
    );

    MPI_Gather(
        local_id.data(),
        static_cast<int>(local_p),
        MPI_UINT32_T,
        rank == 0 ? all_id.data() : NULL,
        static_cast<int>(local_p),
        MPI_UINT32_T,
        0,
        comm
    );

    t1 = MPI_Wtime();

    double gather_local_us = (t1 - t0) * 1000000.0;
    double gather_max_us = 0.0;

    MPI_Reduce(
        &gather_local_us,
        &gather_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 5. Merge global Top-k on rank 0
    // ========================================================
    double merge_us = 0.0;

    if (rank == 0) {
        double merge_begin = MPI_Wtime();

        final_result = merge_global_top_k(
            all_dist,
            all_id,
            static_cast<size_t>(world_size) * local_p,
            k
        );

        double merge_end = MPI_Wtime();
        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && out_profile != NULL) {
        out_profile->bcast_us = bcast_max_us;

        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_max = local_search_max_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us = (total_end - total_begin) * 1000000.0;

        out_profile->owned_lists_sum = owned_lists_sum;
        out_profile->owned_lists_max = owned_lists_max;

        out_profile->scanned_codes_sum = scanned_codes_sum;
        out_profile->scanned_codes_max = scanned_codes_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_no_bcast_profile(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_this_rank,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    MPI_Comm comm,
    MPISearchProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    // ========================================================
    // 1. 不再进行 MPI_Bcast
    // 所有 rank 都已经本地读取了 test_query，
    // 因此这里直接使用当前 rank 本地的 query 指针。
    // ========================================================
    float* query = query_on_this_rank;

    double bcast_max_us = 0.0;

    // ========================================================
    // 2. Local search
    // ========================================================
    IVFPQLocalMPIProfile local_work_profile;

    double t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query,
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        &local_work_profile
    );

    double t1 = MPI_Wtime();

    double local_search_us = (t1 - t0) * 1000000.0;

    double local_search_min_us = 0.0;
    double local_search_max_us = 0.0;
    double local_search_sum_us = 0.0;

    MPI_Reduce(
        &local_search_us,
        &local_search_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    // ========================================================
    // 3. Workload profile
    // ========================================================
    unsigned long long owned_lists_local =
        static_cast<unsigned long long>(local_work_profile.owned_probe_lists);

    unsigned long long scanned_codes_local =
        static_cast<unsigned long long>(local_work_profile.scanned_codes);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_work_profile.returned_candidates);

    unsigned long long owned_lists_sum = 0;
    unsigned long long owned_lists_max = 0;

    unsigned long long scanned_codes_sum = 0;
    unsigned long long scanned_codes_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 4. Pack local Top-p
    // ========================================================
    std::vector<float> local_dist;
    std::vector<uint32_t> local_id;

    t0 = MPI_Wtime();

    pack_local_top_p(
        local_pq,
        local_p,
        local_dist,
        local_id
    );

    t1 = MPI_Wtime();

    double pack_local_us = (t1 - t0) * 1000000.0;
    double pack_max_us = 0.0;

    MPI_Reduce(
        &pack_local_us,
        &pack_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 5. Gather local Top-p
    // ========================================================
    std::vector<float> all_dist;
    std::vector<uint32_t> all_id;

    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(world_size) * local_p);
        all_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_dist.data(),
        static_cast<int>(local_p),
        MPI_FLOAT,
        rank == 0 ? all_dist.data() : NULL,
        static_cast<int>(local_p),
        MPI_FLOAT,
        0,
        comm
    );

    MPI_Gather(
        local_id.data(),
        static_cast<int>(local_p),
        MPI_UINT32_T,
        rank == 0 ? all_id.data() : NULL,
        static_cast<int>(local_p),
        MPI_UINT32_T,
        0,
        comm
    );

    t1 = MPI_Wtime();

    double gather_local_us = (t1 - t0) * 1000000.0;
    double gather_max_us = 0.0;

    MPI_Reduce(
        &gather_local_us,
        &gather_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 6. Merge global Top-k on rank 0
    // ========================================================
    double merge_us = 0.0;

    if (rank == 0) {
        double merge_begin = MPI_Wtime();

        final_result = merge_global_top_k(
            all_dist,
            all_id,
            static_cast<size_t>(world_size) * local_p,
            k
        );

        double merge_end = MPI_Wtime();
        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    // ========================================================
    // 7. Save profile on rank 0
    // ========================================================
    if (rank == 0 && out_profile != NULL) {
        out_profile->bcast_us = bcast_max_us;

        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_max = local_search_max_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us = (total_end - total_begin) * 1000000.0;

        out_profile->owned_lists_sum = owned_lists_sum;
        out_profile->owned_lists_max = owned_lists_max;

        out_profile->scanned_codes_sum = scanned_codes_sum;
        out_profile->scanned_codes_max = scanned_codes_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_omp_list_no_bcast_profile(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_this_rank,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    int omp_threads,
    MPI_Comm comm,
    MPISearchProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    float* query = query_on_this_rank;

    double bcast_max_us = 0.0;

    // ========================================================
    // 1. Local search with OpenMP list-level parallelism
    // ========================================================
    IVFPQLocalMPIProfile local_work_profile;

    double t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local_omp_list_parallel(
        query,
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        omp_threads,
        &local_work_profile
    );

    double t1 = MPI_Wtime();

    double local_search_us = (t1 - t0) * 1000000.0;

    double local_search_min_us = 0.0;
    double local_search_max_us = 0.0;
    double local_search_sum_us = 0.0;

    MPI_Reduce(
        &local_search_us,
        &local_search_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    // ========================================================
    // 2. Workload profile
    // ========================================================
    unsigned long long owned_lists_local =
        static_cast<unsigned long long>(local_work_profile.owned_probe_lists);

    unsigned long long scanned_codes_local =
        static_cast<unsigned long long>(local_work_profile.scanned_codes);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_work_profile.returned_candidates);

    unsigned long long owned_lists_sum = 0;
    unsigned long long owned_lists_max = 0;

    unsigned long long scanned_codes_sum = 0;
    unsigned long long scanned_codes_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &scanned_codes_local,
        &scanned_codes_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 3. Pack local Top-p
    // ========================================================
    std::vector<float> local_dist;
    std::vector<uint32_t> local_id;

    t0 = MPI_Wtime();

    pack_local_top_p(
        local_pq,
        local_p,
        local_dist,
        local_id
    );

    t1 = MPI_Wtime();

    double pack_local_us = (t1 - t0) * 1000000.0;
    double pack_max_us = 0.0;

    MPI_Reduce(
        &pack_local_us,
        &pack_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 4. Gather local Top-p
    // ========================================================
    std::vector<float> all_dist;
    std::vector<uint32_t> all_id;

    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(world_size) * local_p);
        all_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_dist.data(),
        static_cast<int>(local_p),
        MPI_FLOAT,
        rank == 0 ? all_dist.data() : NULL,
        static_cast<int>(local_p),
        MPI_FLOAT,
        0,
        comm
    );

    MPI_Gather(
        local_id.data(),
        static_cast<int>(local_p),
        MPI_UINT32_T,
        rank == 0 ? all_id.data() : NULL,
        static_cast<int>(local_p),
        MPI_UINT32_T,
        0,
        comm
    );

    t1 = MPI_Wtime();

    double gather_local_us = (t1 - t0) * 1000000.0;
    double gather_max_us = 0.0;

    MPI_Reduce(
        &gather_local_us,
        &gather_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    // ========================================================
    // 5. Merge global Top-k on rank 0
    // ========================================================
    double merge_us = 0.0;

    if (rank == 0) {
        double merge_begin = MPI_Wtime();

        final_result = merge_global_top_k(
            all_dist,
            all_id,
            static_cast<size_t>(world_size) * local_p,
            k
        );

        double merge_end = MPI_Wtime();
        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && out_profile != NULL) {
        out_profile->bcast_us = bcast_max_us;

        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_max = local_search_max_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us = (total_end - total_begin) * 1000000.0;

        out_profile->owned_lists_sum = owned_lists_sum;
        out_profile->owned_lists_max = owned_lists_max;

        out_profile->scanned_codes_sum = scanned_codes_sum;
        out_profile->scanned_codes_max = scanned_codes_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

struct MPILightCandidate {
    float dist;
    uint32_t id;
};

static inline void pack_local_top_p_light(
    std::priority_queue<std::pair<float, uint32_t> > local_pq,
    size_t local_p,
    std::vector<MPILightCandidate>& local_cands
) {
    const float INF = std::numeric_limits<float>::infinity();

    local_cands.resize(local_p);

    for (size_t i = 0; i < local_p; ++i) {
        local_cands[i].dist = INF;
        local_cands[i].id = UINT32_MAX;
    }

    size_t pos = 0;

    while (!local_pq.empty() && pos < local_p) {
        local_cands[pos].dist = local_pq.top().first;
        local_cands[pos].id = local_pq.top().second;
        local_pq.pop();
        ++pos;
    }
}

static inline void update_global_topk_light(
    float dist,
    uint32_t id,
    std::priority_queue<std::pair<float, uint32_t> >& topk,
    size_t k
) {
    if (topk.size() < k) {
        topk.push(std::make_pair(dist, id));
    } else if (dist < topk.top().first) {
        topk.pop();
        topk.push(std::make_pair(dist, id));
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
merge_global_topk_light(
    const std::vector<MPILightCandidate>& all_cands,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < all_cands.size(); ++i) {
        if (all_cands[i].id == UINT32_MAX) {
            continue;
        }

        update_global_topk_light(
            all_cands[i].dist,
            all_cands[i].id,
            global_topk,
            k
        );
    }

    return global_topk;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_no_bcast_light(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_this_rank,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    MPI_Comm comm,
    double* latency_us_out
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    if (latency_us_out != NULL) {
        *latency_us_out = 0.0;
    }

    local_p = std::max(local_p, k);

    double total_begin = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        NULL
    );

    std::vector<MPILightCandidate> local_cands;

    pack_local_top_p_light(
        local_pq,
        local_p,
        local_cands
    );

    std::vector<MPILightCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(MPILightCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(MPILightCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    if (rank == 0) {
        final_result = merge_global_topk_light(
            all_cands,
            k
        );
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && latency_us_out != NULL) {
        *latency_us_out = (total_end - total_begin) * 1000000.0;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_omp_list_no_bcast_light(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_this_rank,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    int omp_threads,
    MPI_Comm comm,
    double* latency_us_out
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    if (latency_us_out != NULL) {
        *latency_us_out = 0.0;
    }

    local_p = std::max(local_p, k);

    double total_begin = MPI_Wtime();

    auto local_pq = index.search_mpi_local_omp_list_parallel(
        query_on_this_rank,
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        omp_threads,
        NULL
    );

    std::vector<MPILightCandidate> local_cands;

    pack_local_top_p_light(
        local_pq,
        local_p,
        local_cands
    );

    std::vector<MPILightCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(MPILightCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(MPILightCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    if (rank == 0) {
        final_result = merge_global_topk_light(
            all_cands,
            k
        );
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && latency_us_out != NULL) {
        *latency_us_out = (total_end - total_begin) * 1000000.0;
    }

    return final_result;
}

struct IVFPQMPICandidate {
    float dist;
    uint32_t id;
};

static inline void pack_ivfpq_local_top_p_candidate(
    std::priority_queue<std::pair<float, uint32_t> > local_pq,
    size_t local_p,
    std::vector<IVFPQMPICandidate>& local_cands
) {
    const float INF = std::numeric_limits<float>::infinity();

    local_cands.resize(local_p);

    for (size_t i = 0; i < local_p; ++i) {
        local_cands[i].dist = INF;
        local_cands[i].id = UINT32_MAX;
    }

    size_t pos = 0;

    while (!local_pq.empty() && pos < local_p) {
        local_cands[pos].dist = local_pq.top().first;
        local_cands[pos].id = local_pq.top().second;
        local_pq.pop();
        ++pos;
    }
}

static inline void update_ivfpq_global_topk_candidate(
    float dist,
    uint32_t id,
    std::priority_queue<std::pair<float, uint32_t> >& topk,
    size_t k
) {
    if (k == 0) {
        return;
    }

    if (topk.size() < k) {
        topk.push(std::make_pair(dist, id));
    } else if (dist < topk.top().first) {
        topk.pop();
        topk.push(std::make_pair(dist, id));
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
merge_ivfpq_global_topk_candidate(
    const std::vector<IVFPQMPICandidate>& all_cands,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < all_cands.size(); ++i) {
        if (all_cands[i].id == UINT32_MAX) {
            continue;
        }

        update_ivfpq_global_topk_candidate(
            all_cands[i].dist,
            all_cands[i].id,
            global_topk,
            k
        );
    }

    return global_topk;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivfpq_local_search_one_omp_list_no_bcast_candidate_gather(
    const IVFPQLocalIndexSIMD& index,
    float* query_on_this_rank,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    size_t local_p,
    int rank,
    int world_size,
    IVFPQMPISplitMode split_mode,
    int omp_threads,
    MPI_Comm comm,
    double* latency_us_out
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    if (latency_us_out != NULL) {
        *latency_us_out = 0.0;
    }

    local_p = std::max(local_p, k);

    double total_begin = MPI_Wtime();

    auto local_pq = index.search_mpi_local_omp_list_parallel(
        query_on_this_rank,
        k,
        nprobe,
        rerank_p,
        local_p,
        rank,
        world_size,
        split_mode,
        omp_threads,
        NULL
    );

    std::vector<IVFPQMPICandidate> local_cands;

    pack_ivfpq_local_top_p_candidate(
        local_pq,
        local_p,
        local_cands
    );

    std::vector<IVFPQMPICandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(IVFPQMPICandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(IVFPQMPICandidate)),
        MPI_BYTE,
        0,
        comm
    );

    if (rank == 0) {
        final_result = merge_ivfpq_global_topk_candidate(
            all_cands,
            k
        );
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && latency_us_out != NULL) {
        *latency_us_out = (total_end - total_begin) * 1000000.0;
    }

    return final_result;
}

