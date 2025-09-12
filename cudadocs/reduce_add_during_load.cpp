//#include<iostream>
//#include "cuda_runtime.h"
//
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//// ���CUDA�����Ƿ�ɹ�
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
//	float* input_begin = d_input +2* blockDim.x * blockIdx.x;
//	__shared__ float shared[THREAD_PER_BLOCK];
//	shared[threadIdx.x] = input_begin[threadIdx.x]+input_begin[threadIdx.x+blockDim.x];
//	__syncthreads();
//
//	for (int i=blockDim.x/2; i > 0; i /= 2) {
//		if (threadIdx.x <i) {
//			shared[threadIdx.x] += shared[threadIdx.x + i];
//		}
//		__syncthreads();
//	}
//	if (threadIdx.x == 0)
//		d_output[blockIdx.x] = shared[0];
//}
//
//int main() {
//	const int N = 32 * 1024 * 1024;
//	float* input = (float*)malloc(N * sizeof(float));
//	float* d_input;
//	CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));
//
//	int block_num = N / THREAD_PER_BLOCK/2;		//������һ��shared���ص�ʱ���������block�����������߳̿����
//	float* output = (float*)malloc(block_num * sizeof(float));
//	float* d_output;
//	cudaMalloc((void**)&d_output, block_num * sizeof(float));		//���ﶼ����Ϊblock_num��������һ��
//
//	float* result = (float*)malloc(block_num * sizeof(float));
//
//	for (int i = 0; i < N; i++) {
//		input[i] = 2.0 * (float)rand() - 1.0;
//	}
//
//	//cpu����Ҫ���ģ�block�Ĳ����������룬�̹߳���ӱ�
//	for (int i = 0; i < block_num; i++) {
//		float cur = 0;
//		for (int j = 0; j < 2*THREAD_PER_BLOCK; j++) {
//			cur += input[i *2* THREAD_PER_BLOCK + j];
//		}
//		result[i] = cur;
//	}
//
//	CHECK(cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));
//
//	dim3 grid(block_num, 1);
//	dim3 block(THREAD_PER_BLOCK, 1);
//
//	reduce << <grid, block >> > (d_input, d_output);
//
//	float* out = (float*)malloc(block_num * sizeof(float));
//	CHECK(cudaMemcpy(out, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost));
//	for (int i = 0; i < block_num; i++) {
//		if (abs(out[i] - result[i]) > 0.0005) {
//			printf("����ȣ�����");
//			break;
//		}
//	}
//	printf("���");
//	cudaFree(d_input);
//	cudaFree(d_output);
//	free(input);
//	free(output);
//	free(result);
//	free(out);
//	return 0;
//}
