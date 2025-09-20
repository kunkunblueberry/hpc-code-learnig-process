#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 每个小球的结构体（包含速度、位置和质量）
struct Ball {
    float v[3];       // 速度分量 (x, y, z)
    float position[3];// 位置分量 (x, y, z)
    float mass;       // 质量
};

// 全局参数定义
#define THREAD 128       // 每个线程块的线程数
__constant__ const float width = 100.0f;   // 水平边界宽度
__constant__ const float height = 500.0f;   // 垂直边界高度
__constant__  const float gravity = -5.8f;  // 重力加速度
__constant__  const float E = -0.6f;        // 弹性系数（阻尼）
__constant__  const float dt = 1.0f / 60.0f;// 时间步长
__constant__  const float memofG = 0.5f * gravity * dt * dt;  // 重力加速度的位移项


// CUDA核函数：更新小球状态（位置、速度）
template<int TRACE>
__global__ void updateBalls(Ball* balls, int N) {
    // 计算当前线程处理的小球索引
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    // 防止索引越界（当小球总数不是线程数整数倍时）
    if (n >= N) return;
    int count = 0;
    for (int i = 0; i < TRACE; i++) {
        // 保存当前y方向的位置和速度（用于后续计算）
     //   float oldYPos = balls[n].position[1];
        float oldYVel = balls[n].v[1];


        // 更新y方向速度（应用重力加速度）
        balls[n].v[1] += gravity * dt;
        // 更新y方向位置（基于当前速度和重力）
        balls[n].position[1] += oldYVel * dt + memofG;

        // 检测下边界碰撞（y=0处）
        if (balls[n].position[1] <= 0.0f) {
            balls[n].position[1] = 0.0f;  // 防止穿透边界
            balls[n].v[1] = oldYVel * E;  // 速度反向并应用阻尼
            count++;
        }

        // 当速度变化极小时，视为静止（避免微小振动）
        if (fabsf(oldYVel - balls[n].v[1]) <= 0.01f) {
            balls[n].v[1] = 0.0f;
        }

        if (n == 0||n==1) {
            printf("Thread %d: Position = %f, Velocity = %f\n",
                n, balls[n].position[1], balls[n].v[1]);
            if (fabsf(oldYVel - balls[n].v[1]) <= 0.0001f) {
                balls[n].v[1] = 0.0f;
                printf("The ball stops moving - executed %d frames\n,the ball jump %d times\n", i,count);
                break;
            }
        }

    }
}

int main() {
    //时间记录参数
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_total);
    // 小球总数（调试阶段先用1024个，后续可改大）
    const int N = 1024*1024;

    // 1. CPU端初始化小球数据
    Ball* h_balls = new Ball[N];
    for (int i = 0; i < N; i++) {
        // 初始化位置：x在中心附近±0.3，y在顶部，z=0
        h_balls[i].position[0] = width / 2.0f + 0.3f * ((rand() % 100) / 50.0f - 1.0f);
        h_balls[i].position[1] = height;  // 初始在顶部
        h_balls[i].position[2] = 0.0f;

        // 初始化速度：x=0，y方向随机±0.3，z=0
        h_balls[i].v[0] = 0.0f;
        h_balls[i].v[1] = 10.0f * ((rand() % 100) / 50.0f - 1.0f);
        h_balls[i].v[2] = 0.0f;

        // 质量设为1.0（简化计算）
        h_balls[i].mass = 1.0f;
    }

    //  在GPU上分配小球数组内存
    Ball* d_balls;
    cudaMalloc(&d_balls, N * sizeof(Ball));

    //  将CPU端初始化的数据拷贝到GPU
    cudaMemcpy(d_balls, h_balls, N * sizeof(Ball), cudaMemcpyHostToDevice);

    //  配置线程块和网格
    dim3 blockDim(THREAD);                  // 每个块128个线程
    dim3 gridDim((N + THREAD - 1) / THREAD); // 计算所需块数（向上取整）


    cudaEventRecord(start_kernel);
    //  执行核函数（更新小球状态）
    updateBalls<6000> << <gridDim, blockDim >> > (d_balls, N);

    // 等待核函数执行完成，并检查错误
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("核函数执行错误: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 6. 将GPU计算结果拷贝回CPU（用于验证）
    Ball* h_result = new Ball[N];
    cudaMemcpy(h_result, d_balls, N * sizeof(Ball), cudaMemcpyDeviceToHost);

    //  释放内存
    delete[] h_balls;
    delete[] h_result;
    cudaFree(d_balls);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    //时间记录
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    printf("核函数执行时间: %.6f 毫秒\n", kernel_time);

    // 计算并打印整个程序执行时间
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    printf("整个程序执行时间: %.6f 毫秒\n", total_time);

    // 销毁事件（可选，程序结束会自动释放，显式销毁更规范）
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
