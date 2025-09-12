//#include<iostream>
//#include "cuda_runtime.h"
//
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//#define THREAD_PER_BLOCK 256
//
//__global__ void reduce(float*d_input,float*d_output) {
//	float* input_begin = d_input + blockDim.x * blockIdx.x;
//	int n = threadIdx.x + blockDim.x * blockIdx.x;
//	for (int i = 1; i < blockDim.x; i <<= 1) {
//		int index = threadIdx.x * 2 * i;
//		d_input[index] += d_input[index + i];
//		__syncthreads();
//
//	}
//	if (threadIdx.x == 0)
//		d_output[blockIdx.x] = d_input[threadIdx.x];
//}
//
//int main() {
//	const int N = 32 * 1024 * 1024;
//	float* input = (float*)malloc(N * sizeof(float));
//	float* d_input;
//	cudaMalloc((void**)&d_input, N * sizeof(float));
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
//	cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
//
//	dim3 grid(N / THREAD_PER_BLOCK, 1);
//	dim3 block(THREAD_PER_BLOCK, 1);
//
//	reduce << <grid, block >> > (d_input, d_output);
//
//	float *out=(float*)malloc(N/THREAD_PER_BLOCK*sizeof(float));
//	cudaMemcpy(out, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < block_num; i++) {
//		if (abs(out[i]-result[i])>0.0005) {
//			printf("≤ªœ‡µ»£°£°£°");
//			break;
//		}
//	}
//	cudaFree(d_input);
//	cudaFree(d_output);
//	return 0;
//}
