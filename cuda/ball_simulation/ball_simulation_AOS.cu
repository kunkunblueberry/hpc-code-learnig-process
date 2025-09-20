#include <cuda_runtime.h>
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
//直接大大方方这么表示

// 全局参数定义
#define THREAD 128       // 每个线程块的线程数
const float width = 100.0f;   // 水平边界宽度
const float height = 50.0f;   // 垂直边界高度
const float gravity = -1.8f;  // 重力加速度
const float E = -0.6f;        // 弹性系数（阻尼）
const float dt = 1.0f / 60.0f;// 时间步长
const float memofG = 0.5f * gravity * dt * dt;  // 重力加速度的位移项


__global__ void kernel(Ball*balls,int N){
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 防止索引越界（当小球总数不是线程数整数倍时）
    if (n >= N) return;
    
    // 保存当前y方向的位置和速度（用于后续计算）
    float oldYPos = balls[n].position[1];
    float oldYVel = balls[n].v[1];

    balls[n].position[1] += oldYVel * dt + memofG;
    
    // 检测下边界碰撞（y=0处）
    if (balls[n].position[1] <= 0.0f) {
        balls[n].position[1] = 0.0f;  // 防止穿透边界
        balls[n].v[1] = oldYVel * E;  // 速度反向并应用阻尼
    }
    
    // 当速度变化极小时，视为静止（避免微小振动）
    if (fabsf(oldYVel - balls[n].v[1]) <= 0.01f) {
        balls[n].v[1] = 0.0f;
    }
    
    // 更新y方向速度（应用重力加速度）
    balls[n].v[1] += gravity * dt;
}

int main(){
    const int N=1024;

    Ball*h_ball=new Ball[N];
    for(int i=0;i<N;i++){
        h_ball[i].v[0]=0.0f;
        h_ball[i].v[1]=0.3f*((rand()%100)/50.0f-1.0f);
        h_ball[i].v[2]=0.0f;

        // 初始化位置：x在中心附近±0.3，y在顶部，z=0
        h_balls[i].position[0] = width / 2.0f + 0.3f * ((rand() % 100) / 50.0f - 1.0f);
        h_balls[i].position[1] = height;  // 初始在顶部
        h_balls[i].position[2] = 0.0f;

        h_ball[i].mass=1.0f;
    }

    //分配gpu内存
    Ball*d_ball;
    cudaMalloc(d_ball,N*sizeof(Ball));
    //直接拷贝过去就行了
    cudaMemcpy(d_ball,h_ball,N*sizeof(Ball),cudaMemcpyHostToDevice);

    dim3 blockDim(THREAD);
    dim3 grid((N+THREAD-1)/THREAD);

    updateBalls<<<gridDim, blockDim>>>(d_balls, N);
    
    // 等待核函数执行完成，并检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("核函数执行错误: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 将GPU计算结果拷贝回CPU（用于验证）
    Ball* h_result = new Ball[N];
    cudaMemcpy(h_result, d_balls, N * sizeof(Ball), cudaMemcpyDeviceToHost);
    
    // 打印部分结果（验证是否正确执行）
    printf("前5个小球的状态（y方向）：\n");
    for (int i = 0; i < 5; i++) {
        printf("小球 %d: 位置=%.2f, 速度=%.2f\n",
               i, h_result[i].position[1], h_result[i].v[1]);
    }
    
    // 释放内存
    delete[] h_balls;
    delete[] h_result;
    cudaFree(d_balls);

    retunr 0;
}