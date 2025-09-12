///*
//核心思想
//在规约规模达到一个warp之前，必须需要同步来让数据能够正确读取
//但是我们知道一个warp里所有操作是一样的，也就是simt，这里隐含着一种隐形同步，
//换言之当达到一个warp的规模下，不需要同步来耗时了
//*/
////要加关键字volatile
////volatile 是给变量加的一个 “特殊标记”，核心作用就一句话：告诉编译器 “这个变量的值可能会被你看不见的代码偷偷改”，
//// 所以别瞎优化，每次用它都得去shared内存里读，改了也得立刻写回内存。
////不加这个，编译器会优化if，导致答案错误
//
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
//	float* input_begin = d_input +2* blockDim.x * blockIdx.x;
//	volatile __shared__ float shared[THREAD_PER_BLOCK];
//	shared[threadIdx.x] = input_begin[threadIdx.x]+input_begin[threadIdx.x+blockDim.x];
//	__syncthreads();
//
//	for (int i=blockDim.x/2; i > 32; i /= 2) {		//这里的终止条件是小于等于一个warp
//		if (threadIdx.x <i) {
//			shared[threadIdx.x] += shared[threadIdx.x + i];
//		}
//		__syncthreads();
//	}
//	//下面处理线程数量小于一个warp
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
//	int block_num = N / THREAD_PER_BLOCK/2;		//首先是一个shared加载的时候管理两个block，所以这里线程块减半
//	float* output = (float*)malloc(block_num * sizeof(float));
//	float* d_output;
//	cudaMalloc((void**)&d_output, block_num * sizeof(float));		//这里都更改为block_num，和上面一致
//
//	float* result = (float*)malloc(block_num * sizeof(float));
//
//	for (int i = 0; i < N; i++) {
//		input[i] = 2.0 * (float)rand() - 1.0;
//	}
//
//	//cpu计算要更改，block的参与数量减半，线程管理加倍
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
//
///*
//寄存器缓存导致读取旧值
//编译器会把 shared[threadIdx.x] 的值缓存到当前线程的寄存器中（因为它认为 “没有其他线程会修改这个位置”）。
//当其他线程修改了 shared[threadIdx.x + 16] 等位置时，当前线程可能仍然读取寄存器中缓存的旧值，而不是共享内存中的新值。
//例如，在这段代码中：
//shared[threadIdx.x] += shared[threadIdx.x + 32];
//shared[threadIdx.x] += shared[threadIdx.x + 16];
//编译器可能会优化为：用寄存器保存第一次累加后的结果，而第二次累加时读取的 shared[threadIdx.x + 16] 还是旧值（未被其他线程更新）。
//
//指令重排破坏执行顺序
//编译器会认为连续的 shared 操作是独立的，可能会重排指令顺序。例如，把后面的 shared[threadIdx.x] += shared[threadIdx.x + 8] 提前执行，导致读取到未更新的数据。
//这在 warp 内没有显式同步的情况下尤为致命 —— 虽然 warp 内线程执行相同指令流，但编译器的重排会打破 “先写后读” 的依赖关系。
//
//死代码消除
//编译器可能会认为某些对 shared 的写入是 “无效的”（比如后续没有被当前线程读取），从而优化掉这些写入操作。但实际上，这些写入是给其他线程准备的数据。
//
//注意这样一个核心，每一个线程眼中只有自己，我们眼中能看到所有线程，不要被你的视角带入
//*/
