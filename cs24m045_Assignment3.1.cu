#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MOD 1000000007ULL
#define BLOCK_SIZE 1024

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
        if (prev == child) {
            if (eq) atomicAdd(&rank[root], 1);
            return 1;
        }
    }
    return 0;
}

__device__ unsigned long long pack_edge(unsigned long long w, int idx, int E) {
    return w * (unsigned long long)E + (unsigned long long)idx;
}

__global__ void initializer(int V, int* parent, int* rank, int* changed, unsigned long long* ans) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) { parent[tid] = tid; rank[tid] = 1; }
    if (tid == 0) { *changed = 1; *ans = 0; }
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

static int pick_num_blocks(void* kernel, int blockSize) {
    int perSM = 0, smCount = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    smCount = prop.multiProcessorCount;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kernel, blockSize, 0);
    if (perSM < 1) perSM = 1;
    return perSM * smCount;
}

int main() {

    int V, E;
    if (!(std::cin >> V >> E)) return 0;

    std::vector<int> h_src(E), h_dst(E);
    std::vector<long long> h_wt(E);

    for (int i = 0; i < E; ++i) {
        int u, v, w;
        std::string s;
        std::cin >> u >> v >> w >> s;
        long long mul = 1;
        if (s == "green") mul = 2;
        else if (s == "traffic") mul = 5;
        else if (s == "dept") mul = 3;
        h_src[i] = u;
        h_dst[i] = v;
        h_wt[i]  = (long long)w * mul;
    }

    int *d_src = nullptr, *d_dst = nullptr;
    long long *d_wt = nullptr;
    int *d_parent = nullptr, *d_rank = nullptr, *d_changed = nullptr;
    unsigned long long *d_cheapest = nullptr, *d_ans = nullptr;

    cudaMalloc(&d_src, E * sizeof(int));
    cudaMalloc(&d_dst, E * sizeof(int));
    cudaMalloc(&d_wt,  E * sizeof(long long));
    cudaMalloc(&d_parent, V * sizeof(int));
    cudaMalloc(&d_rank,   V * sizeof(int));
    cudaMalloc(&d_cheapest, V * sizeof(unsigned long long));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_ans, sizeof(unsigned long long));

    cudaMemcpy(d_src, h_src.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt,  h_wt.data(),  E * sizeof(long long), cudaMemcpyHostToDevice);

    int initBlocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializer<<<initBlocks, BLOCK_SIZE>>>(V, d_parent, d_rank, d_changed, d_ans);
    cudaDeviceSynchronize();

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch) {
        std::cerr << "Cooperative launch not supported\n";
        return 1;
    }

    int numBlocks = pick_num_blocks((void*)boruvka_kernel, BLOCK_SIZE);

    void* args[] = {
        &V, &E,
        &d_src, &d_dst, &d_wt,
        &d_parent, &d_rank, &d_cheapest,
        &d_changed, &d_ans
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaLaunchCooperativeKernel((void*)boruvka_kernel, numBlocks, BLOCK_SIZE, args);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    unsigned long long h_ans = 0;
    cudaMemcpy(&h_ans, d_ans, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << (h_ans % MOD) << "\n";
    // std::cerr << "time(s): " << std::chrono::duration<double>(t1 - t0).count() << "\n";

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_wt);
    cudaFree(d_parent);
    cudaFree(d_rank);
    cudaFree(d_cheapest);
    cudaFree(d_changed);
    cudaFree(d_ans);

    return 0;
}
