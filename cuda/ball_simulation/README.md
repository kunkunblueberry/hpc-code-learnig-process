recoding tow ways to simulate the statement of balls in cuda.

SOA and AOS 
i think the diffcult thing is not the writing kernel.its how allocate memory.it really cost lost of time!!!


// 设备端常量内存
__constant__ float width;
__constant__ float height;  // 修正拼写错误 heigh->height
__constant__ float gravity;
__constant__ float E;       // 阻尼系数
__constant__ float dt;
__constant__ float memofG;

main{
...
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
...
}
//这样才能被使用，但是用nvcc在linux编译就不会


实测在vs 2022里面无法被

