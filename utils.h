//
// Created by johannes on 08.03.2022.
//
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <stdio.h>
#ifndef MEM_CUDA_UTILS_H
#define MEM_CUDA_UTILS_H

#endif //MEM_CUDA_UTILS_H

using namespace std;
template <typename T> static std::vector<T> genRandomInts(size_t elements, size_t maximum)
{
    std::vector<T> randoms(elements);
    for (size_t i = 0; i < elements; i++)
    {
        randoms[i] = rand() % maximum;


    }


    return randoms;
}

#define CUDA_TRY(expr)                                                                                                                               \
    do {                                                                                                                                             \
        cudaError_t _err_ = (expr);                                                                                                                  \
        if (_err_ != cudaSuccess) {                                                                                                                  \
            report_cuda_error(_err_, #expr, __FILE__, __LINE__, true);                                                                               \
        }                                                                                                                                            \
    } while (0)

#define CUDA_TIME_FORCE_ENABLED(ce_start, ce_stop, stream, time, ...)                                                                                \
    do {                                                                                                                                             \
                                                                                                        \
        CUDA_TRY(cudaEventRecord((ce_start)));                                                                                                       \
        {                                                                                                                                            \
            __VA_ARGS__;                                                                                                                             \
        }                                                                                                                                            \
        CUDA_TRY(cudaEventRecord((ce_stop)));                                                                                                        \
        CUDA_TRY(cudaEventSynchronize((ce_stop)));                                                                                                   \
        CUDA_TRY(cudaEventElapsedTime((time), (ce_start), (ce_stop)));                                                                               \
    } while (0)

static void report_cuda_error(cudaError_t err, const char* cmd, const char* file, int line, bool die)
{
    printf("CUDA Error at %s:%i%s%s\n%s\n", file, line, cmd ? ": " : "", cmd ? cmd : "", cudaGetErrorString(err));
    assert(false);
    if (die) exit(EXIT_FAILURE);
}

void validate_cpu (uint64_t * data, uint64_t * gpu_data, size_t datasize)
{
    uint64_t endresult=0;
    uint64_t * gpu = new uint64_t [1];
    cudaMemcpy(gpu,gpu_data, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for(int i=0;i<datasize;i++)
    {
        endresult+= data[i];
    }
    cout<<endresult<<";"<<gpu[0]<<endl;

    if(endresult== gpu[0])
    {
        cout<<"VALIDATION SUCCESSFULL"<<endl;

    }
    else
    {
        cout<<"VALIDATION FAILED"<<endl;
        cout<<"CPU ; GPU "<<endresult<<" ; "<<gpu[0]<<endl;
    }
}

template <typename T>
static void write_benchmark(size_t clustercount,size_t datasize,string datatype, string dataset, float selectivity, fstream &myfile, float runtime_ms, string kernel)
{
    size_t MEBIBYTE = (1<<20);
    size_t MBSIZE= datasize * sizeof(T) / MEBIBYTE;
    size_t MASKSIZE = datasize  / MEBIBYTE;
    size_t total_size= MBSIZE + MASKSIZE;

    myfile
            <<MBSIZE << ";"   //datasize in MIB
            <<datatype<<";"
            <<dataset<< ";"
            <<selectivity<< ";"
            <<clustercount<<";"
            <<kernel<<";"                                       //kernel name
            // <<thread_dim<<";"
            //  <<block_dim<<";"
            <<runtime_ms<<";"
            <<((total_size) / (runtime_ms)) * (float)(1000.0/1024.0) <<endl;
    //mask size added for throughput



}
template <typename T>
static void write_bench_file (size_t clustercount,
                              string datatype,
                              string filename,
                              std::vector<std::pair<std::string, float>> benchs,
                              std::vector<float> timings,
                              size_t iterations,
                              size_t datasize,
                              string dataset,
                              float selectivity              )
{

    fstream myfile(filename,std::ios_base::app | std::ios_base::trunc);
    myfile.open(filename);

    cout<<" FILE TO WRITE: "<<filename<<endl;

    //only write header if output file is empty
    if(myfile.peek() == std::ifstream::traits_type::eof())
    {
        cout<<"PEEK PERFORMANCE!!"<<endl;




        ofstream myfile_out(filename);

        myfile_out << "datasize[MiB];datatype;dataset;selectivity;cluster_count;kernel;threads;blocks;time in ms;throughput [GiB / s ];" << endl;


        myfile_out.close();

    }

    myfile.close();
    myfile.open(filename);
    myfile.seekg (0, ios::end);



    for (int i = 0; i < benchs.size(); i++) {
        std::cout << "benchmark " << benchs[i].first << " time (ms): " << (double)timings[i] / iterations << std::endl;
        write_benchmark<T>(clustercount,datasize,datatype,dataset,selectivity,myfile,(double)timings[i] / (double) iterations,benchs[i].first);
    }

    myfile.close();
}


template <typename T> void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (uint32_t i = offset; i < offset + length; i++) {
    //    std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout  << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T> void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer + offset, length * sizeof(T), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < length; i++) {

        std::cout  << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template <typename T> T* vector_to_gpu(const std::vector<T>& vec)
{
    T* buff;
    const auto size = vec.size() * sizeof(T);
    CUDA_TRY(cudaMalloc(&buff, size));
    CUDA_TRY(cudaMemcpy(buff, &vec[0], size, cudaMemcpyHostToDevice));
    return buff;
}

template <typename T> std::vector<T> gpu_to_vector(T* buff, size_t length)
{
    std::vector<T> vec;
    vec.resize(length);
    CUDA_TRY(cudaMemcpy(&vec[0], buff, length * sizeof(T), cudaMemcpyDeviceToHost));
    return vec;
}

template <class T> struct dont_deduce_t {
    using type = T;
};

template <typename T> T gpu_to_val(T* d_val)
{
    T val;
    CUDA_TRY(cudaMemcpy(&val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
    return val;
}

template <typename T> void val_to_gpu(T* d_val, typename dont_deduce_t<T>::type val)
{
    CUDA_TRY(cudaMemcpy(d_val, &val, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> T* alloc_gpu(size_t length)
{
    T* buff;
    CUDA_TRY(cudaMalloc(&buff, length * sizeof(T)));
    return buff;
}