#include <iostream>
#include "utils.h"
#include <cub/cub.cuh>
#define MEBIBYTE (1<<20)
using namespace std;
__global__ void addkernel (uint32_t * in, uint32_t * out, int datasize)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < datasize; i += blockDim.x * gridDim.x) {
        out[i] = in[i];
    }
}
template<typename T>
__global__ void hadd(T *in, int datasize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0)
        for(int k =1;k<datasize;k++)
        {
            in[0]+=in[k];
        }

}
__global__ void addkernel_linear (uint32_t * in, uint32_t * out, int datasize, int ele_thread)
{

    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*ele_thread;
    int elec=0;


        for (int i = idx; i < datasize&&elec <ele_thread; i++) {
            out[i] = in[i];
            elec++;
        }

}





int main() {
    size_t datasize_MIB = 1024;
    std::vector<uint32_t> col;
    size_t ele_count = MEBIBYTE* datasize_MIB / sizeof (uint32_t);
    col.resize(ele_count);
    col=genRandomInts<uint32_t>(ele_count, 45000);
    uint32_t * d_input = vector_to_gpu(col);



    cout<<"kernel;vectorsize;time;throughput [GiB/s]"<<endl;
    for(size_t vectorsize_MIB=2; vectorsize_MIB<= 1024; vectorsize_MIB=vectorsize_MIB*2)
    {




    size_t ele_count_vector = MEBIBYTE* vectorsize_MIB / sizeof (uint32_t);

    uint32_t * d_output = alloc_gpu<uint32_t>(ele_count_vector);
    uint32_t * d_output_lin = alloc_gpu<uint32_t>(ele_count_vector);
        uint32_t * d_sum = alloc_gpu<uint32_t>(1);
        uint32_t * d_sum_lin = alloc_gpu<uint32_t>(1);
    size_t total_threads = 128*512;
    size_t ele_threads = ele_count_vector/total_threads;
    size_t iterations = datasize_MIB/vectorsize_MIB;

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_output, d_sum, ele_count_vector);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

    float time =0;
    float time_l=0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEvent_t start_l;
    cudaEvent_t stop_l;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventCreate(&start_l));
    CUDA_TRY(cudaEventCreate(&stop_l));

    CUDA_TIME_FORCE_ENABLED( start, stop, 0, &time, {
    for(int i =0; i<iterations;i++)
    {




    addkernel<<<128,512>>>(d_input+i*ele_count_vector, d_output, ele_count_vector);


    }
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_output, d_sum, ele_count_vector);

    });

    CUDA_TIME_FORCE_ENABLED( start_l, stop_l, 0, &time_l, {
        for(int i =0; i<iterations;i++)
        {


            addkernel_linear<<<128,512>>>(d_input+i*ele_count_vector, d_output_lin, ele_count_vector,ele_threads);



        }




        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_output_lin, d_sum_lin, ele_count_vector);
    });

    cout<<"add_strided;" <<vectorsize_MIB <<";"<<time<<";"<< ((datasize_MIB) / (time)) * (float)(1000.0/1024.0)<<endl;
    cout<<"add_linear;" <<vectorsize_MIB <<";"<<time_l<<";"<< ((datasize_MIB) / (time_l)) * (float)(1000.0/1024.0)<<endl;

    }

    return 0;
}
