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



    时间记录接口
    测量全部
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    // 计时事件：测量核函数
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

//调用启动接口
    cudaEventRecord(start_total);
    cudaEventSynchronize(stop_kernel); // 确保事件已记录
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total); // 确保事件已记录

//时间打印，很方便咯
float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    printf("核函数执行时间: %.3f 毫秒\n", kernel_time);

    // 计算并打印整个程序执行时间
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    printf("整个程序执行时间: %.3f 毫秒\n", total_time);

    // 销毁事件（可选，程序结束会自动释放，显式销毁更规范）
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
