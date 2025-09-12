///*
//����˼��
//�ڹ�Լ��ģ�ﵽһ��warp֮ǰ��������Ҫͬ�����������ܹ���ȷ��ȡ
//��������֪��һ��warp�����в�����һ���ģ�Ҳ����simt������������һ������ͬ����
//����֮���ﵽһ��warp�Ĺ�ģ�£�����Ҫͬ������ʱ��
//*/
////Ҫ�ӹؼ���volatile
////volatile �Ǹ������ӵ�һ�� �������ǡ����������þ�һ�仰�����߱����� �����������ֵ���ܻᱻ�㿴�����Ĵ���͵͵�ġ���
//// ���Ա�Ϲ�Ż���ÿ����������ȥshared�ڴ����������Ҳ������д���ڴ档
////������������������Ż�if�����´𰸴���
//
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
//	volatile __shared__ float shared[THREAD_PER_BLOCK];
//	shared[threadIdx.x] = input_begin[threadIdx.x]+input_begin[threadIdx.x+blockDim.x];
//	__syncthreads();
//
//	for (int i=blockDim.x/2; i > 32; i /= 2) {		//�������ֹ������С�ڵ���һ��warp
//		if (threadIdx.x <i) {
//			shared[threadIdx.x] += shared[threadIdx.x + i];
//		}
//		__syncthreads();
//	}
//	//���洦���߳�����С��һ��warp
//	if (threadIdx.x < 32) {
//		shared[threadIdx.x] += shared[threadIdx.x + 32];
//		shared[threadIdx.x] += shared[threadIdx.x + 16];
//		shared[threadIdx.x] += shared[threadIdx.x + 8];
//		shared[threadIdx.x] += shared[threadIdx.x + 4];
//		shared[threadIdx.x] += shared[threadIdx.x + 2];
//		shared[threadIdx.x] += shared[threadIdx.x + 1];
//	}
//
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
//
///*
//�Ĵ������浼�¶�ȡ��ֵ
//��������� shared[threadIdx.x] ��ֵ���浽��ǰ�̵߳ļĴ����У���Ϊ����Ϊ ��û�������̻߳��޸����λ�á�����
//�������߳��޸��� shared[threadIdx.x + 16] ��λ��ʱ����ǰ�߳̿�����Ȼ��ȡ�Ĵ����л���ľ�ֵ�������ǹ����ڴ��е���ֵ��
//���磬����δ����У�
//shared[threadIdx.x] += shared[threadIdx.x + 32];
//shared[threadIdx.x] += shared[threadIdx.x + 16];
//���������ܻ��Ż�Ϊ���üĴ��������һ���ۼӺ�Ľ�������ڶ����ۼ�ʱ��ȡ�� shared[threadIdx.x + 16] ���Ǿ�ֵ��δ�������̸߳��£���
//
//ָ�������ƻ�ִ��˳��
//����������Ϊ������ shared �����Ƕ����ģ����ܻ�����ָ��˳�����磬�Ѻ���� shared[threadIdx.x] += shared[threadIdx.x + 8] ��ǰִ�У����¶�ȡ��δ���µ����ݡ�
//���� warp ��û����ʽͬ�����������Ϊ���� ���� ��Ȼ warp ���߳�ִ����ָͬ�������������������Ż���� ����д����� ��������ϵ��
//
//����������
//���������ܻ���ΪĳЩ�� shared ��д���� ����Ч�ġ����������û�б���ǰ�̶߳�ȡ�����Ӷ��Ż�����Щд���������ʵ���ϣ���Щд���Ǹ������߳�׼�������ݡ�
//
//ע������һ�����ģ�ÿһ���߳�����ֻ���Լ������������ܿ��������̣߳���Ҫ������ӽǴ���
//*/
