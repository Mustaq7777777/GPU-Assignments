#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;


typedef long long ll;

__global__ void dkernel(long int * matrix, long int * filter, long int *result, int h, int w, int c, int r, int s, int k){

    extern __shared__ long int sh[];

    int inserted_H;

    if(blockIdx.y <= (r/2)){
      inserted_H = blockIdx.z + ((r/2) - blockIdx.y) ;
    }
    else{
      inserted_H = blockIdx.z - (blockIdx.y - (r/2));
    }


    if(inserted_H >=0 && inserted_H < h){

      long int * img = sh;
      long int * ker = sh + w;

      int currKernel= blockIdx.x / c;
      int currChannel = blockIdx.x % c;

      img[threadIdx.x] = matrix[currChannel * (h*w) + blockIdx.z * w + threadIdx.x ];

      int offset = currKernel * (r*s*c) + currChannel * (r*s) + blockIdx.y * s ;

      for(int index = threadIdx.x ; index < s ; index += w){
        ker[index] = filter[offset + index];
      }

      __syncthreads();

      long int value=0;
      int start = threadIdx.x - (s/2);

      for(int i=0;i<s;i++){

        int curr = start + i;
        if(curr>=0 && curr < w){
          value += ker[i] * img[curr];
        }
      }

      int id = currKernel * (h*w) + inserted_H * w + threadIdx.x;
      atomicAdd((unsigned long long int *)&result[id], (long long int)value);

    }    

}

__global__ void initialization(long int *result, int k, int h, int w) {

    long long int id= blockIdx.x * (h*w) + blockIdx.y * w + threadIdx.x;

    result[id]=0;
}


int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/



    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    long int *mat;
    long int *filter;
    long int *ans;

    cudaMalloc(&mat, (h * w * c) * sizeof(long int));
    cudaMalloc(&filter, (r * s * c * k) * sizeof(long int));
    cudaMalloc(&ans, (h * w * k) * sizeof(long int));

    cudaMemcpy(mat, h_mat, (h * w * c) * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(filter, h_filter, (r * s * c * k) * sizeof(long int), cudaMemcpyHostToDevice);

    dim3 gridDim(k,h);
    dim3 blockDim(w);
    initialization<<<gridDim, blockDim>>>(ans, k, h, w);

    cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferShared);

    dim3 nblocks1( k*c, r, h);
    dkernel<<<nblocks1, w, (s+w) * sizeof(long int)>>>(mat, filter, ans, h, w, c, r, s, k);
      
    cudaMemcpy(h_ans, ans, (h*w*k) * sizeof(long int), cudaMemcpyDeviceToHost);
    
    
    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
