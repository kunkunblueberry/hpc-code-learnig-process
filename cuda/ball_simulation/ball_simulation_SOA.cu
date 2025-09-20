#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <curand_kernel.h>
#include <cstdlib>
#include <cmath>  // 添加cmath以使用abs函数

#definde THREAD 128
struct member{
    flaot*v;
    float*position
    float mass;
}

// 设备端常量内存
__constant__ float width;
__constant__ float height;  // 修正拼写错误 heigh->height
__constant__ float gravity;
__constant__ float E;       // 阻尼系数
__constant__ float dt;
__constant__ float memofG;

__global__ void kernel(member* Ball, int N) {
    int tid = threadIdx.x;
    int n = tid + blockIdx.x * blockDim.x;
    if (n >= N) return;

    int index3 = n * 3;

    // 记录当前状态并更新
    float oldv = Ball->v[index3 + 1];
    float current_pos = Ball->position[index3 + 1];  // 使用变量而非未使用的oldposition

    // 更新位置 (使用当前位置计算)
    Ball->position[index3 + 1] = current_pos + oldv * dt + memofG;

    // 地面碰撞检测
    if (Ball->position[index3 + 1] <= 0) {
        Ball->position[index3 + 1] = 0;
        Ball->v[index3 + 1] = oldv * E;
    }

    // 速度过小则停止运动
    if (fabsf(oldv - Ball->v[index3 + 1]) <= 0.01f) {  // 使用CUDA的单精度函数fabsf
        Ball->v[index3 + 1] = 0.0f;
    }

    // 仅让第一个线程打印信息
    if (n == 0) {
        printf("Thread %d: Position = %f, Velocity = %f\n",
            n, Ball->position[index3 + 1], Ball->v[index3 + 1]);
    }
}



int main(){
    const int N=32*1024;

    // 主机端常量定义 (与设备端对应)
    const float host_width = 100.0f;
    const float host_height = 50.0f;
    const float host_gravity = -1.8f;
    const float host_E = -0.6f;
    const float host_dt = 1.f / 60.f;
    const float host_memofG = 0.5f * host_gravity * host_dt * host_dt;

    // 将常量复制到设备端常量内存
    cudaMemcpyToSymbol(width, &host_width, sizeof(float));
    cudaMemcpyToSymbol(height, &host_height, sizeof(float));
    cudaMemcpyToSymbol(gravity, &host_gravity, sizeof(float));
    cudaMemcpyToSymbol(E, &host_E, sizeof(float));
    cudaMemcpyToSymbol(dt, &host_dt, sizeof(float));
    cudaMemcpyToSymbol(memofG, &host_memofG, sizeof(float));


    flaot*h_pos=new float[N*3];
    flaot*h_val=new float[N*3];

    for(int i=0;i<N;i++){
        int idx3=i*3;
        h_pos[idx3]=host_width / 2 + 0.3f * ((rand() % 100) / 50.f - 1.0f);
        h_pos[idx3+1]=host_height;
        h_pos[idx3+2]=0.0f;

        h_vel[idx3] = 0.0f;
        h_vel[idx3 + 1] = 0.3f * ((rand() % 100) / 50.0f - 1.0f);
        h_vel[idx3 + 2] = 0.0f;
    }

    //分配设备内存      这里的逻辑是，先分配好单独的块，然后分配好了再指向结构体，这样更清晰
    member*d_ball;
    cudaMalloc(d_ball,sizeof(member));

    flaot*d_pos;
    float*d_vel;
    cudaMalloc(&d_pos, N * 3 * sizeof(float));
    cudaMalloc(&d_vel, N * 3 * sizeof(float));

    cudaMemcpy(d_pos, h_pos, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, N * 3 * sizeof(float), cudaMemcpyHostToDevice);

    member h_ball;
    h_ball.position = d_pos;
    h_ball.v = d_vel;
    h_ball.mass = 1.0f;  // 质量

    cudaMemcpy(d_ball, &h_ball, sizeof(member), cudaMemcpyHostToDevice);

     // 启动内核
    dim3 block(THREAD);
    im3 grid((N + THREAD - 1) / THREAD);
    kernel << <grid, block >> > (d_ball, N);

    // 错误检查
    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }

    cudaDeviceSynchronize();

    // 释放资源
    delete[] h_pos;
    delete[] h_vel;
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_ball);

    return 0;
}