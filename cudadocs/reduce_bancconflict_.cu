//#include<iostream>
//#include "cuda_runtime.h"
//
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//// 检查CUDA操作是否成功
//#define CHECK(call) \
//    do { \
//        cudaError_t err = call; \
//        if (err != cudaSuccess) { \
//            fprintf(stderr, "Error: %s in file %s, line %d\n", \
//                    cudaGetErrorString(err), __FILE__, __LINE__); \
//            exit(EXIT_FAILURE); \
//        } \
//    } while (0)
//
//#define THREAD_PER_BLOCK 256
//
//__global__ void reduce(float* d_input, float* d_output) {
//	float* input_begin = d_input + blockDim.x * blockIdx.x;
//	int n = threadIdx.x + blockDim.x * blockIdx.x;
//
//	__shared__ float shared[THREAD_PER_BLOCK];
//	shared[threadIdx.x] = d_input[n];
//	__syncthreads();
//
//	for (int i = blockDim.x / 2; i > 0; i /= 2) {
//		if (threadIdx.x < i) {
//			shared[threadIdx.x] += shared[threadIdx.x + i];
//		}
//		__syncthreads();
//	}
//	if (threadIdx.x == 0)
//		d_output[blockIdx.x] = shared[0];
//}
//
//
////cuda的报错就是一坨，说了等于没说，搞笑
//
//int main() {
//	const int N = 32 * 1024 * 1024;
//	float* input = (float*)malloc(N * sizeof(float));
//	float* d_input;
//	CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));
//
//	int block_num = N / THREAD_PER_BLOCK;
//	float* output = (float*)malloc(N / THREAD_PER_BLOCK * sizeof(float));
//	float* d_output;
//	cudaMalloc((void**)&d_output, N / THREAD_PER_BLOCK * sizeof(float));
//
//	float* result = (float*)malloc(N / THREAD_PER_BLOCK * sizeof(float));
//
//	for (int i = 0; i < N; i++) {
//		input[i] = 2.0 * (float)rand() - 1.0;
//	}
//	for (int i = 0; i < block_num; i++) {
//		float cur = 0;
//		for (int j = 0; j < THREAD_PER_BLOCK; j++) {
//			cur += input[i * THREAD_PER_BLOCK + j];
//		}
//		result[i] = cur;
//	}
//
//	CHECK(cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));
//
//	dim3 grid(N / THREAD_PER_BLOCK, 1);
//	dim3 block(THREAD_PER_BLOCK, 1);
//
//	reduce << <grid, block >> > (d_input, d_output);
//
//	// 检查内核错误
//	CHECK(cudaGetLastError());
//	CHECK(cudaDeviceSynchronize());  // 等待内核执行完成
//
//	float* out = (float*)malloc(block_num * sizeof(float));
//	CHECK(cudaMemcpy(out, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost));
//	for (int i = 0; i < block_num; i++) {
//		if (abs(out[i] - result[i]) > 0.0005) {
//			printf("不相等！！！");
//			break;
//		}
//	}
//	printf("相等");
//	cudaFree(d_input);
//	cudaFree(d_output);
//	free(input);
//	free(output);
//	free(result);
//	free(out);
//	return 0;
//}
