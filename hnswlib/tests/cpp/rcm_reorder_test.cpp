#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../hnswlib/hnswlib.h"

namespace {

using Result = std::vector<std::pair<float, hnswlib::labeltype> >;

Result collect(std::priority_queue<std::pair<float, hnswlib::labeltype> > queue) {
    Result result;
    while (!queue.empty()) {
        result.push_back(queue.top());
        queue.pop();
    }
    return result;
}

uint64_t gscore(hnswlib::HierarchicalNSW<float>& index, size_t window) {
    const size_t count = index.getCurrentElementCount();
    std::vector<std::vector<hnswlib::tableint> > out(count);
    std::vector<std::vector<hnswlib::tableint> > in(count);
    for (hnswlib::tableint node = 0; node < count; ++node) {
        hnswlib::linklistsizeint* links = index.get_linklist0(node);
        const size_t degree = index.getListCount(links);
        hnswlib::tableint* neighbors =
            reinterpret_cast<hnswlib::tableint*>(links + 1);
        out[node].assign(neighbors, neighbors + degree);
        std::sort(out[node].begin(), out[node].end());
        out[node].erase(
            std::unique(out[node].begin(), out[node].end()), out[node].end());
        for (hnswlib::tableint neighbor : out[node]) {
            in[neighbor].push_back(node);
        }
    }

    uint64_t score = 0;
    for (size_t right = 0; right < count; ++right) {
        const size_t begin = right > window ? right - window : 0;
        for (size_t left = begin; left < right; ++left) {
            score += std::binary_search(
                out[left].begin(), out[left].end(),
                static_cast<hnswlib::tableint>(right));
            score += std::binary_search(
                out[right].begin(), out[right].end(),
                static_cast<hnswlib::tableint>(left));

            size_t i = 0;
            size_t j = 0;
            while (i < in[left].size() && j < in[right].size()) {
                if (in[left][i] == in[right][j]) {
                    ++score;
                    ++i;
                    ++j;
                } else if (in[left][i] < in[right][j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }
    }
    return score;
}

}  // namespace

int main() {
    const size_t count = 512;
    const size_t dimension = 16;
    const size_t query_count = 32;
    const size_t k = 10;

    std::mt19937 generator(2026);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> data(count * dimension);
    for (size_t row = 0; row < count; ++row) {
        float norm = 0.0f;
        for (size_t column = 0; column < dimension; ++column) {
            const float value = distribution(generator);
            data[row * dimension + column] = value;
            norm += value * value;
        }
        norm = std::sqrt(norm);
        for (size_t column = 0; column < dimension; ++column) {
            data[row * dimension + column] /= norm;
        }
    }

    hnswlib::InnerProductSpace space(static_cast<int>(dimension));
    hnswlib::HierarchicalNSW<float> index(&space, count, 8, 80, 2026);
    for (size_t row = 0; row < count; ++row) {
        index.addPoint(
            data.data() + row * dimension,
            static_cast<hnswlib::labeltype>(100000 + row));
    }
    index.setEf(64);

    std::vector<Result> before;
    before.reserve(query_count);
    for (size_t query = 0; query < query_count; ++query) {
        before.push_back(collect(index.searchKnn(
            data.data() + query * dimension, k)));
    }

    const size_t moved = index.reorderIndexRCM();
    if (moved == 0) {
        throw std::runtime_error("RCM unexpectedly produced the identity layout");
    }
    index.checkIntegrity();

    for (size_t query = 0; query < query_count; ++query) {
        const Result after = collect(index.searchKnn(
            data.data() + query * dimension, k));
        if (after != before[query]) {
            throw std::runtime_error("search results changed after RCM reordering");
        }
    }

    const char* saved_path = "rcm_reorder_test.index";
    index.saveIndex(saved_path);
    hnswlib::HierarchicalNSW<float> loaded(&space, saved_path);
    loaded.setEf(64);
    for (size_t query = 0; query < query_count; ++query) {
        const Result after_reload = collect(loaded.searchKnn(
            data.data() + query * dimension, k));
        if (after_reload != before[query]) {
            std::remove(saved_path);
            throw std::runtime_error("search results changed after saving and loading RCM index");
        }
    }
    std::remove(saved_path);

    hnswlib::HierarchicalNSW<float> gorder_index(
        &space, count, 8, 80, 2026);
    for (size_t row = 0; row < count; ++row) {
        gorder_index.addPoint(
            data.data() + row * dimension,
            static_cast<hnswlib::labeltype>(100000 + row));
    }
    gorder_index.setEf(64);
    const uint64_t gscore_before = gscore(gorder_index, 5);
    const size_t gorder_moved = gorder_index.reorderIndexGorder(5);
    if (gorder_moved == 0) {
        throw std::runtime_error("Gorder unexpectedly produced the identity layout");
    }
    gorder_index.checkIntegrity();
    const uint64_t gscore_after = gscore(gorder_index, 5);
    if (gscore_after <= gscore_before) {
        throw std::runtime_error("Gorder did not improve its sliding-window objective");
    }
    for (size_t query = 0; query < query_count; ++query) {
        const Result after_gorder = collect(gorder_index.searchKnn(
            data.data() + query * dimension, k));
        if (after_gorder != before[query]) {
            throw std::runtime_error("search results changed after Gorder reordering");
        }
    }

    const char* gorder_saved_path = "gorder_reorder_test.index";
    gorder_index.saveIndex(gorder_saved_path);
    hnswlib::HierarchicalNSW<float> loaded_gorder(&space, gorder_saved_path);
    loaded_gorder.setEf(64);
    for (size_t query = 0; query < query_count; ++query) {
        const Result after_reload = collect(loaded_gorder.searchKnn(
            data.data() + query * dimension, k));
        if (after_reload != before[query]) {
            std::remove(gorder_saved_path);
            throw std::runtime_error(
                "search results changed after saving and loading Gorder index");
        }
    }
    std::remove(gorder_saved_path);

    hnswlib::HierarchicalNSW<float> porder_index(
        &space, count, 8, 80, 2026);
    for (size_t row = 0; row < count; ++row) {
        porder_index.addPoint(
            data.data() + row * dimension,
            static_cast<hnswlib::labeltype>(100000 + row));
    }
    porder_index.setEf(64);
    porder_index.startEdgeProfiling();
    for (size_t query = 0; query < 8; ++query) {
        (void) porder_index.searchKnnProfiled(
            data.data() + query * dimension, k);
    }
    const uint64_t profiled_edges = porder_index.getProfiledEdgeTraversals();
    if (profiled_edges == 0) {
        throw std::runtime_error("Porder did not record any profiled edge traversals");
    }
    const size_t porder_moved = porder_index.reorderIndexPorder(5);
    if (porder_moved == 0) {
        throw std::runtime_error("Porder unexpectedly produced the identity layout");
    }
    porder_index.checkIntegrity();
    for (size_t query = 0; query < query_count; ++query) {
        const Result after_porder = collect(porder_index.searchKnn(
            data.data() + query * dimension, k));
        if (after_porder != before[query]) {
            throw std::runtime_error("search results changed after Porder reordering");
        }
    }

    const char* porder_saved_path = "porder_reorder_test.index";
    porder_index.saveIndex(porder_saved_path);
    hnswlib::HierarchicalNSW<float> loaded_porder(&space, porder_saved_path);
    loaded_porder.setEf(64);
    for (size_t query = 0; query < query_count; ++query) {
        const Result after_reload = collect(loaded_porder.searchKnn(
            data.data() + query * dimension, k));
        if (after_reload != before[query]) {
            std::remove(porder_saved_path);
            throw std::runtime_error(
                "search results changed after saving and loading Porder index");
        }
    }
    std::remove(porder_saved_path);

    std::cout << "RCM, Gorder, and Porder preserved all search results; moved "
              << moved << ", " << gorder_moved << ", and "
              << porder_moved << " of " << count
              << " nodes; Porder profiled " << profiled_edges
              << " edge traversals; Gscore improved from "
              << gscore_before << " to " << gscore_after << "\n";
    return 0;
}
