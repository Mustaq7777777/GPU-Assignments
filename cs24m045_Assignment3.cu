#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MOD 1000000007
#define BLOCK_SIZE 1024

using std::cin;
using std::cout;

__global__ void initializer(int E, int V, long int *parent, int *rank, int *changed, long int *wt, long int *mulwt, unsigned long long *ans)
{

  long int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < E && id < V)
  {
    rank[id] = 1;
    wt[id] *= mulwt[id];
    parent[id] = id;
  }

  else if (id < E && id >= V)
  {
    wt[id] *= mulwt[id];
  }

  else if (id >= E && id < V)
  {
    parent[id] = id;
    rank[id] = 1;
  }

  if (id == 0)
  {
    *changed = 1;
    *ans = 0;
  }
}

__device__ long int find_root(long int *parent, long int i)
{

  for (; i != parent[i]; i = parent[i])
  {
    int temp = parent[parent[i]];
    parent[i] = temp;
  }

  return i;
}

__device__ void findingMinAtomiccaly(int index, const int moduloNumber, long int *wt, unsigned long long *cheapest, long int u, long int v)
{

  unsigned long long edgeCandidate = index + moduloNumber * (unsigned long long)wt[index];
  atomicMin(&cheapest[v], edgeCandidate);
  atomicMin(&cheapest[u], edgeCandidate);
}

__device__ long int pathCompressFind(long int *parent, long int i)
{

  long int root;
  for (root = i; parent[root] != root; root = parent[root])
    ;
  while (parent[i] != root)
  {
    long int next = parent[i];
    parent[i] = root;
    i = next;
  }
  return root;
}

__global__ void boruvka_kernel(int V, int E, long int *src, long int *dst, long int *wt,
                               long int *parent, int *rank,
                               unsigned long long *cheapest, int *changed, unsigned long long *ans)
{

  cg::grid_group grid = cg::this_grid();

  long int id = threadIdx.x + blockIdx.x * blockDim.x;

  long int numberOfThreads = blockDim.x * gridDim.x;

  const int moduloNumber = E;

  while (*changed)
  {

    int ii = id;

    do
    {
      if (ii >= V)
        break;
      cheapest[ii] = ULLONG_MAX;
      ii = ii + numberOfThreads;
    } while (ii < V);

    grid.sync();

    if (id == 0)
      *changed = 0;

    ii = id;

    while (ii < E)
    {

      long int set2 = pathCompressFind(parent, dst[ii]);
      long int set1 = pathCompressFind(parent, src[ii]);

      if (set1 == set2)
      {
        ii = ii + numberOfThreads;
        continue;
      }

      else
      {
        findingMinAtomiccaly(ii, moduloNumber, wt, cheapest, set1, set2);
      }

      ii += numberOfThreads;
    }

    grid.sync();

    for (int i = id; i < V; i = i + numberOfThreads)
    {

      if (cheapest[i] == ULLONG_MAX)
        continue;

      const int max_retries = 100;
      int edgeIdx = cheapest[i] % moduloNumber;

      long int set1 = find_root(parent, src[edgeIdx]);
      long int set2 = find_root(parent, dst[edgeIdx]);

      for (int retries = 0; retries < max_retries; retries++)
      {

        if (set1 > set2)
        {
          set1 = set1 ^ set2;
          set2 = set1 ^ set2;
          set1 = set1 ^ set2;
        }

        if (set1 == set2)
          break;

        if (parent[set1] != set1)
          continue;

        long int prev = atomicCAS((int *)&parent[set1], set1, set2);

        long int u = src[edgeIdx];
        long int v = dst[edgeIdx];

        if (set1 == prev)
        {
          atomicAdd(&rank[set2], 1);
          atomicExch(changed, 1);
          // long int wt_edge = wt[edgeIdx];
          atomicAdd(ans, wt[edgeIdx] % MOD);
          break;
        }

        set1 = find_root(parent, u);
        set2 = find_root(parent, v);
      }
    }

    grid.sync();
  }
}

int main()
{
  int V, E;
  cin >> V >> E;

  long int *h_src = new long int[E];
  long int *h_dst = new long int[E];
  long int *h_wt = new long int[E];
  long int *h_mulwt = new long int[E];
  unsigned long long h_ans = 0;

  for (int i = 0; i < E; ++i)
  {
    int u, v, wt;
    std::string s;
    cin >> u >> v >> wt >> s;
    h_src[i] = u;
    h_dst[i] = v;
    h_wt[i] = wt;
    if (s == "green")
      h_mulwt[i] = 2;
    else if (s == "traffic")
      h_mulwt[i] = 5;
    else if (s == "dept")
      h_mulwt[i] = 3;
    else
      h_mulwt[i] = 1;
  }

  // Device allocations

  long int *d_src, *d_dst, *d_wt, *d_parent;
  long int *d_mulwt;
  int *d_rank, *d_changed;
  unsigned long long *d_cheapest, *d_ans;

  cudaMalloc(&d_src, E * sizeof(long int));
  cudaMalloc(&d_dst, E * sizeof(long int));
  cudaMalloc(&d_wt, E * sizeof(long int));
  cudaMalloc(&d_parent, V * sizeof(long int));
  cudaMalloc(&d_mulwt, E * sizeof(long int));

  cudaMalloc(&d_rank, V * sizeof(int));
  cudaMalloc(&d_cheapest, V * sizeof(long int));
  cudaMalloc(&d_changed, sizeof(int));
  cudaMalloc(&d_ans, sizeof(unsigned long long));

  cudaMemcpy(d_src, h_src, E * sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst, h_dst, E * sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wt, h_wt, E * sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mulwt, h_mulwt, E * sizeof(long int), cudaMemcpyHostToDevice);

  int numblocks1 = (max(E, V) + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int numBlocks = (max(E, V) + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (numBlocks * BLOCK_SIZE >= 32 * 1024)
    numBlocks = 32;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  auto start = std::chrono::high_resolution_clock::now();

  initializer<<<numblocks1, BLOCK_SIZE>>>(E, V, d_parent, d_rank, d_changed, d_wt, d_mulwt, d_ans);
  cudaDeviceSynchronize();

  if (prop.cooperativeLaunch)
  {
    void *args[] = {&V, &E, &d_src, &d_dst, &d_wt, &d_parent, &d_rank,
                    &d_cheapest, &d_changed, &d_ans};
    cudaLaunchCooperativeKernel((void *)boruvka_kernel, numBlocks, BLOCK_SIZE, args);
  }
  else
  {
    std::cerr << "Cooperative launch not supported!\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  cudaMemcpy(&h_ans, d_ans, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  cout << h_ans % MOD << std ::endl;

  // std::ofstream file("cuda.out");
  // if (file.is_open())
  // {
  //   file << h_ans % MOD << "\n";
  //   file.close();
  // }

  // std::ofstream file2("cuda_timing.out");
  // if (file2.is_open())
  // {
  //   file2 << elapsed.count() << "\n";
  //   file2.close();
  // }

  // Cleanup

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_wt);
  cudaFree(d_parent);
  cudaFree(d_rank);
  cudaFree(d_cheapest);
  cudaFree(d_changed);
  cudaFree(d_ans);

  delete[] h_src;
  delete[] h_dst;
  delete[] h_wt;
  delete[] h_mulwt;

  return 0;
}
