/*
一个bock里处理的数据略微多一点好，也不是越多越好
可以覆盖访问内存的巨大延迟
*/

#include<iostream>
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <stdio.h>

// 检查CUDA操作是否成功
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error: %s in file %s, line %d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define THREAD_PER_BLOCK 256

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(float* d_input, float* d_output) {
	float* input_begin = d_input + 2 * NUM_PER_BLOCK * blockIdx.x;
	volatile __shared__ float shared[THREAD_PER_BLOCK];
	for (int i = 0; i < ; i++) {

	}


	//for (int i=blockDim.x/2; i > 32; i /= 2) {		
	//	if (threadIdx.x <i) {
	//		shared[threadIdx.x] += shared[threadIdx.x + i];
	//	}
	//	__syncthreads();
	//}
	//简单来说就是把这个循环一个一个写出来，避免for循环的开销

	if (threadIdx.x < THREAD_PER_BLOCK / 2)//128
	{
		shared[threadIdx.x] += shared[threadIdx.x + blockDim.x / 2];
	}
	__syncthreads();
	if (threadIdx.x < THREAD_PER_BLOCK / 4)//64
	{
		shared[threadIdx.x] += shared[threadIdx.x + blockDim.x / 4];
	}
	__syncthreads();

	//这个的缺点是不好封装，封装太困难，THREAD_PER_BLOCK变了后，代码里面也要变


	if (threadIdx.x < 32) {
		shared[threadIdx.x] += shared[threadIdx.x + 32];
		shared[threadIdx.x] += shared[threadIdx.x + 16];
		shared[threadIdx.x] += shared[threadIdx.x + 8];
		shared[threadIdx.x] += shared[threadIdx.x + 4];
		shared[threadIdx.x] += shared[threadIdx.x + 2];
		shared[threadIdx.x] += shared[threadIdx.x + 1];
	}

	if (threadIdx.x == 0)
		d_output[blockIdx.x] = shared[0];
}

int main() {
	const int N = 32 * 1024 * 1024;
	float* input = (float*)malloc(N * sizeof(float));
	float* d_input;
	CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));

	int block_num = 1024;		//首先是一个shared加载的时候管理两个block，所以这里线程块减半
	int num_per_block = N/block_num;
	float* output = (float*)malloc(block_num * sizeof(float));
	float* d_output;
	cudaMalloc((void**)&d_output, block_num * sizeof(float));		//这里都更改为block_num，和上面一致

	float* result = (float*)malloc(block_num * sizeof(float));

	for (int i = 0; i < N; i++) {
		input[i] = 2.0 * (float)rand() - 1.0;
	}

	int num_per_block = N / block_num; 
	//cpu计算要更改，block的参与数量减半，线程管理加倍
	for (int i = 0; i < block_num; i++) {
		float cur = 0;
		for (int j = 0; j < num_per_block; j++) {
			cur += input[i * num_per_block + j];
		}
		result[i] = cur;
	}

	CHECK(cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));

	dim3 grid(block_num, 1);
	dim3 block(THREAD_PER_BLOCK, 1);

	reduce << <grid, block >> > (d_input, d_output);

	float* out = (float*)malloc(block_num * sizeof(float));
	CHECK(cudaMemcpy(out, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < block_num; i++) {
		if (abs(out[i] - result[i]) > 0.0005) {
			printf("不相等！！！");
			break;
		}
	}
	printf("相等");
	cudaFree(d_input);
	cudaFree(d_output);
	free(input);
	free(output);
	free(result);
	free(out);
	return 0;
}