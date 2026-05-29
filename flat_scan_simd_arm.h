#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>
#include <omp.h>

// NEON 内积函数：计算 a · b
static inline float inner_product_neon(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    // 一次处理 4 个 float
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        // sum_vec += va * vb
        float32x4_t prod = vmulq_f32(va, vb);
        sum_vec = vaddq_f32(sum_vec, prod);
    }

    // 将 4 个通道的结果取出并水平求和
    float temp[4];
    vst1q_f32(temp, sum_vec);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    // 处理剩余维度，但这里其实没用，96%4=0
    /*
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    */

    return result;
}

static inline float inner_product_neon8(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    for (; i + 8 <= dim; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
    }

    float32x4_t sum = vaddq_f32(sum0, sum1);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

static inline float inner_product_neon16(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
        sum2 = vaddq_f32(sum2, vmulq_f32(a2, b2));
        sum3 = vaddq_f32(sum3, vmulq_f32(a3, b3));
    }

    // 合并 4 条累加链
    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum = vaddq_f32(sum01, sum23);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

static inline float inner_product_neon32(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    float32x4_t sum5 = vdupq_n_f32(0.0f);
    float32x4_t sum6 = vdupq_n_f32(0.0f);
    float32x4_t sum7 = vdupq_n_f32(0.0f);

    for (; i + 32 <= dim; i += 32) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        float32x4_t a4 = vld1q_f32(a + i + 16);
        float32x4_t b4 = vld1q_f32(b + i + 16);

        float32x4_t a5 = vld1q_f32(a + i + 20);
        float32x4_t b5 = vld1q_f32(b + i + 20);

        float32x4_t a6 = vld1q_f32(a + i + 24);
        float32x4_t b6 = vld1q_f32(b + i + 24);

        float32x4_t a7 = vld1q_f32(a + i + 28);
        float32x4_t b7 = vld1q_f32(b + i + 28);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
        sum2 = vaddq_f32(sum2, vmulq_f32(a2, b2));
        sum3 = vaddq_f32(sum3, vmulq_f32(a3, b3));
        sum4 = vaddq_f32(sum4, vmulq_f32(a4, b4));
        sum5 = vaddq_f32(sum5, vmulq_f32(a5, b5));
        sum6 = vaddq_f32(sum6, vmulq_f32(a6, b6));
        sum7 = vaddq_f32(sum7, vmulq_f32(a7, b7));
    }

    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum45 = vaddq_f32(sum4, sum5);
    float32x4_t sum67 = vaddq_f32(sum6, sum7);

    float32x4_t sum0123 = vaddq_f32(sum01, sum23);
    float32x4_t sum4567 = vaddq_f32(sum45, sum67);
    float32x4_t sum = vaddq_f32(sum0123, sum4567);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

// NEON + FMA + 4路累加器，一次循环处理 16 个 float
static inline float inner_product_neon16_fma(const float* __restrict__ a, const float* __restrict__ b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
        sum2 = vfmaq_f32(sum2, a2, b2);
        sum3 = vfmaq_f32(sum3, a3, b3);
    }

    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum = vaddq_f32(sum01, sum23);

    float result = vaddvq_f32(sum);

    return result;
}

// IP distance:
static inline float ip_distance_neon(const float* base_vec, const float* query, size_t vecdim) {
    return 1.0f - inner_product_neon16_fma(base_vec, query, vecdim);
}

std::priority_queue<std::pair<float, uint32_t> > flat_search_simd(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for (size_t i = 0; i < base_number; ++i) {
        const float* base_vec = base + i * vecdim;

        float dis = ip_distance_neon(base_vec, query, vecdim);

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else {
            if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)});
                q.pop();
            }
        }
    }

    return q;
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_simd_fasttopk(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    const size_t MAX_K = 10;  // 本实验 k=10
    float best_score[MAX_K];
    uint32_t best_id[MAX_K];

    size_t cnt = 0;
    size_t worst_pos = 0;
    float worst_score = 0.0f;

    auto recompute_worst = [&]() {
        worst_pos = 0;
        worst_score = best_score[0];

        for (size_t j = 1; j < cnt; ++j) {
            if (best_score[j] < worst_score) {
                worst_score = best_score[j];
                worst_pos = j;
            }
        }
    };

    for (size_t i = 0; i < base_number; ++i) {
        ///*
        if (i + 16 < base_number) {
            __builtin_prefetch(base + (i + 16) * vecdim, 0, 1);
        }
        //*/
        const float* base_vec = base + i * vecdim;

        float score = inner_product_neon16_fma(base_vec, query, vecdim);

        if (cnt < k) {
            best_score[cnt] = score;
            best_id[cnt] = static_cast<uint32_t>(i);
            ++cnt;

            if (cnt == k) {
                recompute_worst();
            }
        } else {
            if (score > worst_score) {
                best_score[worst_pos] = score;
                best_id[worst_pos] = static_cast<uint32_t>(i);
                recompute_worst();
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;
    for (size_t j = 0; j < cnt; ++j) {
        float dis = 1.0f - best_score[j];
        q.push({dis, best_id[j]});
    }

    return q;
}

//--------------------------------
// Flat-SIMD pthread query 划分并行
//--------------------------------

struct FlatQueryThreadParam {
    int tid;
    int num_threads;

    float* base;
    float* queries;

    size_t base_number;
    size_t query_number;
    size_t vecdim;
    size_t k;

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >* results;
};

static void* flat_query_parallel_worker(void* arg) {
    FlatQueryThreadParam* param =
        static_cast<FlatQueryThreadParam*>(arg);

    int tid = param->tid;
    int num_threads = param->num_threads;

    float* base = param->base;
    float* queries = param->queries;

    size_t base_number = param->base_number;
    size_t query_number = param->query_number;
    size_t vecdim = param->vecdim;
    size_t k = param->k;

    auto& results = *(param->results);

    // 循环划分 query，避免只用连续块时某些线程最后空闲
    for (size_t qid = static_cast<size_t>(tid);
         qid < query_number;
         qid += static_cast<size_t>(num_threads)) {

        float* query = queries + qid * vecdim;

        results[qid] = flat_search_simd_fasttopk(
            base,
            query,
            base_number,
            vecdim,
            k
        );
    }

    return nullptr;
}

static inline void flat_search_simd_pthread_query_parallel(
    float* base,
    float* queries,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t k,
    int num_threads,
    std::vector<std::priority_queue<std::pair<float, uint32_t> > >& results
) {
    if (num_threads <= 1) {
        results.resize(query_number);

        for (size_t qid = 0; qid < query_number; ++qid) {
            float* query = queries + qid * vecdim;

            results[qid] = flat_search_simd_fasttopk(
                base,
                query,
                base_number,
                vecdim,
                k
            );
        }

        return;
    }

    results.resize(query_number);

    std::vector<pthread_t> threads(num_threads);
    std::vector<FlatQueryThreadParam> params(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        params[t].tid = t;
        params[t].num_threads = num_threads;
        params[t].base = base;
        params[t].queries = queries;
        params[t].base_number = base_number;
        params[t].query_number = query_number;
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].results = &results;

        pthread_create(
            &threads[t],
            nullptr,
            flat_query_parallel_worker,
            &params[t]
        );
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

//--------------------------------
// Flat-SIMD pthread base 划分并行
//--------------------------------

struct FlatBaseThreadParam {
    int tid;

    float* base;
    float* query;

    size_t begin;
    size_t end;
    size_t vecdim;

    size_t k;        // 最终返回的 Top-k
    size_t local_p;  // 每个线程维护的局部 Top-p

    float* local_score;
    uint32_t* local_id;
    size_t* local_cnt;
};

static inline void flat_recompute_worst_score(
    const float* best_score,
    size_t cnt,
    size_t& worst_pos,
    float& worst_score
) {
    worst_pos = 0;
    worst_score = best_score[0];

    for (size_t i = 1; i < cnt; ++i) {
        if (best_score[i] < worst_score) {
            worst_score = best_score[i];
            worst_pos = i;
        }
    }
}

// 这里的 capacity 可以是 local_p，也可以是 k
// 用于维护一个固定数组 Top-capacity
static inline void flat_update_top_score(
    float score,
    uint32_t id,
    float* best_score,
    uint32_t* best_id,
    size_t capacity,
    size_t& cnt,
    size_t& worst_pos,
    float& worst_score
) {
    if (cnt < capacity) {
        best_score[cnt] = score;
        best_id[cnt] = id;
        ++cnt;

        if (cnt == capacity) {
            flat_recompute_worst_score(
                best_score,
                cnt,
                worst_pos,
                worst_score
            );
        }
    } else {
        if (score > worst_score) {
            best_score[worst_pos] = score;
            best_id[worst_pos] = id;

            flat_recompute_worst_score(
                best_score,
                cnt,
                worst_pos,
                worst_score
            );
        }
    }
}

// 每个线程扫描一段 base，并维护自己的 local Top-p
static void* flat_base_parallel_worker(void* arg) {
    FlatBaseThreadParam* param =
        static_cast<FlatBaseThreadParam*>(arg);

    float* base = param->base;
    float* query = param->query;

    size_t begin = param->begin;
    size_t end = param->end;
    size_t vecdim = param->vecdim;
    size_t local_p = param->local_p;

    float* best_score = param->local_score;
    uint32_t* best_id = param->local_id;

    size_t cnt = 0;
    size_t worst_pos = 0;
    float worst_score = 0.0f;

    for (size_t i = begin; i < end; ++i) {
        if (i + 16 < end) {
            __builtin_prefetch(
                base + (i + 16) * vecdim,
                0,
                1
            );
        }

        const float* base_vec = base + i * vecdim;

        float score = inner_product_neon16_fma(
            base_vec,
            query,
            vecdim
        );

        // 注意：这里维护的是 local Top-p，而不是 Top-k
        flat_update_top_score(
            score,
            static_cast<uint32_t>(i),
            best_score,
            best_id,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );
    }

    *(param->local_cnt) = cnt;

    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
flat_search_simd_pthread_base_parallel(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t local_p,
    int num_threads
) {
    // local_p 不能小于 k，否则可能无法保证全局 Top-k 的候选完整性
    // if (local_p < k) {
    //     local_p = k;
    // }

    if (num_threads <= 1) {
        return flat_search_simd_fasttopk(
            base,
            query,
            base_number,
            vecdim,
            k
        );
    }

    if (static_cast<size_t>(num_threads) > base_number) {
        num_threads = static_cast<int>(base_number);
    }

    std::vector<pthread_t> threads(num_threads);
    std::vector<FlatBaseThreadParam> params(num_threads);

    // 每个线程维护 local_p 个候选
    std::vector<float> local_scores(
        static_cast<size_t>(num_threads) * local_p
    );

    std::vector<uint32_t> local_ids(
        static_cast<size_t>(num_threads) * local_p
    );

    std::vector<size_t> local_cnt(num_threads, 0);

    // 按连续区间划分 base
    size_t block =
        (base_number + static_cast<size_t>(num_threads) - 1)
        / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        size_t begin = static_cast<size_t>(t) * block;
        size_t end = std::min(begin + block, base_number);

        params[t].tid = t;
        params[t].base = base;
        params[t].query = query;
        params[t].begin = begin;
        params[t].end = end;
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].local_p = local_p;

        params[t].local_score =
            local_scores.data() + static_cast<size_t>(t) * local_p;

        params[t].local_id =
            local_ids.data() + static_cast<size_t>(t) * local_p;

        params[t].local_cnt = &local_cnt[t];

        pthread_create(
            &threads[t],
            nullptr,
            flat_base_parallel_worker,
            &params[t]
        );
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    // reduce：把每个线程的 local Top-p 合并成 global Top-k
    std::vector<float> global_score(k);
    std::vector<uint32_t> global_id(k);

    size_t global_cnt = 0;
    size_t global_worst_pos = 0;
    float global_worst_score = 0.0f;

    for (int t = 0; t < num_threads; ++t) {
        float* score_t =
            local_scores.data() + static_cast<size_t>(t) * local_p;

        uint32_t* id_t =
            local_ids.data() + static_cast<size_t>(t) * local_p;

        for (size_t j = 0; j < local_cnt[t]; ++j) {
            flat_update_top_score(
                score_t[j],
                id_t[j],
                global_score.data(),
                global_id.data(),
                k,
                global_cnt,
                global_worst_pos,
                global_worst_score
            );
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > result;

    for (size_t i = 0; i < global_cnt; ++i) {
        float dis = 1.0f - global_score[i];
        result.push({dis, global_id[i]});
    }

    return result;
}

//--------------------------------
// OpenMP query 级并行版本
//--------------------------------

static inline void flat_search_simd_omp_query_parallel(
    float* base,
    float* queries,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t k,
    int num_threads,
    std::vector<std::priority_queue<std::pair<float, uint32_t> > >& results
) {
    results.resize(query_number);

    if (query_number == 0) {
        return;
    }

    if (num_threads <= 1) {
        for (size_t qid = 0; qid < query_number; ++qid) {
            float* query = queries + qid * vecdim;

            results[qid] = flat_search_simd_fasttopk(
                base,
                query,
                base_number,
                vecdim,
                k
            );
        }

        return;
    }

    if (static_cast<size_t>(num_threads) > query_number) {
        num_threads = static_cast<int>(query_number);
    }

    omp_set_dynamic(0);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (long long qid = 0; qid < static_cast<long long>(query_number); ++qid) {
        float* query = queries + static_cast<size_t>(qid) * vecdim;

        results[static_cast<size_t>(qid)] = flat_search_simd_fasttopk(
            base,
            query,
            base_number,
            vecdim,
            k
        );
    }
}


//-------------------------
// OpenMP base 划分
//-------------------------

static inline void omp_flat_recompute_worst_score(
    const float* best_score,
    size_t cnt,
    size_t& worst_pos,
    float& worst_score
) {
    worst_pos = 0;
    worst_score = best_score[0];

    for (size_t i = 1; i < cnt; ++i) {
        if (best_score[i] < worst_score) {
            worst_score = best_score[i];
            worst_pos = i;
        }
    }
}

static inline void omp_flat_update_top_score(
    float score,
    uint32_t id,
    float* best_score,
    uint32_t* best_id,
    size_t capacity,
    size_t& cnt,
    size_t& worst_pos,
    float& worst_score
) {
    if (capacity == 0) {
        return;
    }

    if (cnt < capacity) {
        best_score[cnt] = score;
        best_id[cnt] = id;
        ++cnt;

        if (cnt == capacity) {
            omp_flat_recompute_worst_score(
                best_score,
                cnt,
                worst_pos,
                worst_score
            );
        }
    } else {
        if (score > worst_score) {
            best_score[worst_pos] = score;
            best_id[worst_pos] = id;

            omp_flat_recompute_worst_score(
                best_score,
                cnt,
                worst_pos,
                worst_score
            );
        }
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
flat_search_simd_omp_base_parallel(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t local_p,
    int num_threads
) {
    if (local_p == 0) {
        std::priority_queue<std::pair<float, uint32_t> > empty_result;
        return empty_result;
    }

    if (num_threads <= 1) {
        return flat_search_simd_fasttopk(
            base,
            query,
            base_number,
            vecdim,
            k
        );
    }

    if (static_cast<size_t>(num_threads) > base_number) {
        num_threads = static_cast<int>(base_number);
    }

    omp_set_dynamic(0);

    std::vector<float> local_scores(
        static_cast<size_t>(num_threads) * local_p
    );

    std::vector<uint32_t> local_ids(
        static_cast<size_t>(num_threads) * local_p
    );

    std::vector<size_t> local_cnt(
        static_cast<size_t>(num_threads),
        0
    );

    size_t block =
        (base_number + static_cast<size_t>(num_threads) - 1)
        / static_cast<size_t>(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        size_t begin = static_cast<size_t>(tid) * block;
        size_t end = std::min(begin + block, base_number);

        float* best_score =
            local_scores.data() + static_cast<size_t>(tid) * local_p;

        uint32_t* best_id =
            local_ids.data() + static_cast<size_t>(tid) * local_p;

        size_t cnt = 0;
        size_t worst_pos = 0;
        float worst_score = 0.0f;

        for (size_t i = begin; i < end; ++i) {
            if (i + 16 < end) {
                __builtin_prefetch(
                    base + (i + 16) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec = base + i * vecdim;

            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            // 每个线程维护自己的 local Top-p
            omp_flat_update_top_score(
                score,
                static_cast<uint32_t>(i),
                best_score,
                best_id,
                local_p,
                cnt,
                worst_pos,
                worst_score
            );
        }

        local_cnt[static_cast<size_t>(tid)] = cnt;
    }

    // reduce：从 num_threads × local_p 个候选中选出全局 Top-k
    std::vector<float> global_score(k);
    std::vector<uint32_t> global_id(k);

    size_t global_cnt = 0;
    size_t global_worst_pos = 0;
    float global_worst_score = 0.0f;

    for (int t = 0; t < num_threads; ++t) {
        float* score_t =
            local_scores.data() + static_cast<size_t>(t) * local_p;

        uint32_t* id_t =
            local_ids.data() + static_cast<size_t>(t) * local_p;

        for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
            omp_flat_update_top_score(
                score_t[j],
                id_t[j],
                global_score.data(),
                global_id.data(),
                k,
                global_cnt,
                global_worst_pos,
                global_worst_score
            );
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > result;

    for (size_t i = 0; i < global_cnt; ++i) {
        float dis = 1.0f - global_score[i];
        result.push({dis, global_id[i]});
    }

    return result;
}