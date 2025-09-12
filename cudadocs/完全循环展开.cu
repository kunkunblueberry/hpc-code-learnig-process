//上一个版本进行的循环展开已经拥有很不错的性能了，后面两个办法也只做到比较有限的性能提升


/*
核心思想
在规约规模达到一个warp之前，必须需要同步来让数据能够正确读取
但是我们知道一个warp里所有操作是一样的，也就是simt，这里隐含着一种隐形同步，
换言之当达到一个warp的规模下，不需要同步来耗时了
*/
//要加关键字volatile
//volatile 是给变量加的一个 “特殊标记”，核心作用就一句话：告诉编译器 “这个变量的值可能会被你看不见的代码偷偷改”，
// 所以别瞎优化，每次用它都得去shared内存里读，改了也得立刻写回内存。
//不加这个，编译器会优化if，导致答案错误

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

__global__ void reduce(float* d_input, float* d_output) {
	float* input_begin = d_input +2* blockDim.x * blockIdx.x;
	volatile __shared__ float shared[THREAD_PER_BLOCK];
	shared[threadIdx.x] = input_begin[threadIdx.x]+input_begin[threadIdx.x+blockDim.x];
	__syncthreads();

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

	int block_num = N / THREAD_PER_BLOCK/2;		//首先是一个shared加载的时候管理两个block，所以这里线程块减半
	float* output = (float*)malloc(block_num * sizeof(float));
	float* d_output;
	CHECK(cudaMalloc((void**)&d_output, block_num * sizeof(float)));		//这里都更改为block_num，和上面一致

	float* result = (float*)malloc(block_num * sizeof(float));	//因为这里是结果，所以分配block_num*sizeof，上面输入才是N*sizeof

	for (int i = 0; i < N; i++) {
		input[i] = 2.0 * (float)rand() - 1.0;
	}

	//cpu计算要更改，block的参与数量减半，线程管理加倍
	for (int i = 0; i < block_num; i++) {
		float cur = 0;
		for (int j = 0; j < 2*THREAD_PER_BLOCK; j++) {
			cur += input[i *2* THREAD_PER_BLOCK + j];
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

/*
在 CUDA 中，有两种核心方式可以实现循环展开（Loop Unrolling）—— 一种是编译器自动优化，
另一种是通过#pragma unroll编译指令手动控制。循环展开的本质是 “将循环体中的迭代步骤直接展开为连续代码”，
从而减少循环控制（如条件判断、计数器自增）的开销，同时为编译器提供更多指令级并行（ILP）的优化空间，尤其适合归约、矩阵乘法等高频循环场景

#pragma unroll是 CUDA（基于 C/C++ 扩展）提供的编译指令，用于显式控制循环的展开行为，优先级高于编译器的自动优化。它有三种使用形式，覆盖不同场景

指令形式	作用说明
#pragma unroll			让编译器 “自动决定展开次数”（通常会展开所有迭代，前提是循环次数是编译期常量）
#pragma unroll N		强制展开为N次迭代（N是整数常量，若循环总次数小于N，会忽略多余展开）
#pragma unroll 1		强制不展开循环（禁用编译器的自动展开，用于调试或特殊场景）

编译器自动优化也可以循环展开，但是这个并非手动，过高的优化反而可能降低速率甚至出错
*/
