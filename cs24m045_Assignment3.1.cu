#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MOD 1000000007ULL

// Path-compressing find on int parents
__device__ int find_root(int* parent, int v) {
    int r = v;
    while (parent[r] != r) r = parent[r];
    while (parent[v] != r) {
        int nxt = parent[v];
        parent[v] = r;
        v = nxt;
    }
    return r;
}

// Union-by-rank with CAS; returns 1 if merged
__device__ int unite_by_rank(int* parent, int* rank, int a, int b) {
    for (int attempt = 0; attempt < 64; ++attempt) {
        int ra = find_root(parent, a);
        int rb = find_root(parent, b);
        if (ra == rb) return 0;

        int root = ra, child = rb;
        bool eq = false;

        int rra = rank[ra], rrb = rank[rb];
        if (rra < rrb) { root = rb; child = ra; }
        else if (rra > rrb) { root = ra; child = rb; }
        else { eq = true; if (ra > rb) { root = rb; child = ra; } }

        int prev = atomicCAS(&parent[child], child, root);
        if (prev == child) { if (eq) atomicAdd(&rank[root], 1); return 1; }
    }
    return 0;
}

// Pack (weight, edgeIndex) so atomicMin picks lightest edge; ties by index
__device__ unsigned long long pack_edge(unsigned long long w, int idx, int E) {
    return w * (unsigned long long)E + (unsigned long long)idx;
}

__global__ void boruvka_kernel(
    int V, int E,
    const int* src,
    const int* dst,
    const long long* wt,
    int* parent,
    int* rank,
    unsigned long long* cheapest,
    int* changed,
    unsigned long long* ans)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nT  = gridDim.x * blockDim.x;

    while (true) {
        for (int i = tid; i < V; i += nT) cheapest[i] = 0xFFFFFFFFFFFFFFFFull;
        grid.sync();

        if (tid == 0) *changed = 0;
        grid.sync();

        for (int e = tid; e < E; e += nT) {
            int u = src[e], v = dst[e];
            int ru = find_root(parent, u);
            int rv = find_root(parent, v);
            if (ru == rv) continue;
            unsigned long long p = pack_edge((unsigned long long)wt[e], e, E);
            atomicMin(&cheapest[ru], p);
            atomicMin(&cheapest[rv], p);
        }
        grid.sync();

        for (int i = tid; i < V; i += nT) {
            unsigned long long c = cheapest[i];
            if (c == 0xFFFFFFFFFFFFFFFFull) continue;
            int ei = (int)(c % (unsigned long long)E);
            int u = src[ei], v = dst[ei];
            if (unite_by_rank(parent, rank, u, v)) {
                atomicAdd(ans, (unsigned long long)(wt[ei] % MOD));
                atomicExch(changed, 1);
            }
        }
        grid.sync();

        if (*changed == 0) break;
        grid.sync();
    }
}
