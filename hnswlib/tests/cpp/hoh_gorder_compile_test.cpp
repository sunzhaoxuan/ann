#include "../../../hnsw_on_hnsw_mpi.h"

void compile_hnsw_on_hnsw_gorder(float* base) {
    HNSWOnHNSWMPIIndex index(
        base,
        1,
        4,
        0,
        1,
        8,
        40,
        32,
        4,
        20,
        8,
        1,
        1,
        "gorder",
        5);
    (void) index.reordered_moved_count();
    index.start_local_edge_profiling();
    (void) index.local_profiled_edge_traversals();
    (void) index.finish_local_porder(5);
}
