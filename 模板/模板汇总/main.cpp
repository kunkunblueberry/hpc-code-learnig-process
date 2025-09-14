/*
第一部分：基础任务 —— 固定长度数值向量类（FixedVector）
任务目标
实现一个支持固定维度的数值向量类FixedVector，掌握模板的 “类型参数” 与 “非类型参数” 基础用法，理解编译期类型检查与固定大小数组的 HPC 优势。
知识点回顾
模板参数类型：
类型参数（template <class T>）：指定向量存储的数据类型（如int float double）；
非类型参数（template <size_t Size>）：指定向量的维度（如 3 维、5 维），为编译期常量，支持数组大小固定（无动态内存开销）。
编译期类型检查：static_assert可在编译阶段验证条件（如限制FixedVector仅支持数值类型），避免运行时错误。
初始化列表：std::initializer_list<T>用于便捷初始化向量（如FixedVector<int,3> vec = {1,2,3}）。
*/
#include "vector_lib.h"
#include <cstring>  // 用于解析命令行参数

// 任务1：基础任务测试（对应参数-f）
void run_task_fundamental() {
    std::cout << "\n=== 基础任务测试 ===" << std::endl;
    FixedVector<int, 3> vec_int = {1, 2, 3};
    std::cout << "int vector: ";
    vec_int.print();

    FixedVector<float, 2> vec_float = {3.14f};
    std::cout << "float vector: ";
    vec_float.print();
}

// 任务2：进阶任务测试（对应参数-s）
void run_task_advanced() {
    std::cout << "\n=== 进阶任务测试 ===" << std::endl;
    FixedVector<int, 3> vec1 = {1, 2, 3};
    FixedVector<int, 3> vec2 = {4, 5, 6};
    
    auto dot_int = vec1.dot(vec2);
    std::cout << "Dot(int, int): " << dot_int << std::endl;

    FixedVector<float, 3> vec3 = {1.0f, 2.0f, 3.0f};
    auto dot_mix = vec1.dot(vec3);
    std::cout << "Dot(int, float): " << dot_mix << std::endl;

    auto vec_add = add(vec1, vec2);
    std::cout << "FixedVector add result: ";
    vec_add.print();

    std::vector<double> vec4 = {1.1, 2.2, 3.3};
    std::vector<double> vec5 = {4.4, 5.5};
    auto vec45_add = add(vec4, vec5);
    std::cout << "std::vector add result: [";
    for (size_t i = 0; i < vec45_add.size(); i++) {
        std::cout << vec45_add[i];
        if (i != vec45_add.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// 任务3：高阶任务测试（对应参数-h）
void run_task_highlevel() {
    std::cout << "\n=== 高阶任务测试 ===" << std::endl;
    FixedVector<int, 3> vec1 = {1, 2, 3};
    std::cout << "Debug print: ";
    vec1.print<true>();
    std::cout << "Release print: ";
    vec1.print();

    FixedVector<float, 3> vec_float = {1.0f, 2.0f, 3.0f};
    std::cout << "Before normalize: ";
    vec_float.print();
    vec_float.normalize();
    std::cout << "After normalize: ";
    vec_float.print();

    FixedVector<double, 3> vec_double = {1.0, 2.0, 3.0};
    vec_double.normalize();

    std::cout << "FixedVector dimension (compile-time): " << VectorDim<decltype(vec1)>::value << std::endl;
    std::vector<double> vec4 = {1.1, 2.2};
    std::cout << "std::vector dimension (runtime): " << VectorDim<decltype(vec4)>::value(vec4) << std::endl;
}

// 显示帮助信息
void print_help() {
    std::cout << "用法: ./vector_test [参数]" << std::endl;
    std::cout << "参数说明:" << std::endl;
    std::cout << "  -f    运行基础任务测试（FixedVector初始化与打印）" << std::endl;
    std::cout << "  -s    运行进阶任务测试（点积与加法）" << std::endl;
    std::cout << "  -h    运行高阶任务测试（Debug打印、归一化、VectorDim）" << std::endl;
    std::cout << "  -all  运行所有任务测试" << std::endl;
    std::cout << "  -help 显示此帮助信息" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_help();
        return 1;
    }

    // 根据命令行参数执行对应任务
    if (strcmp(argv[1], "-f") == 0) {
        run_task_fundamental();
    } else if (strcmp(argv[1], "-s") == 0) {
        run_task_advanced();
    } else if (strcmp(argv[1], "-h") == 0) {
        run_task_highlevel();
    } else if (strcmp(argv[1], "-all") == 0) {
        run_task_fundamental();
        run_task_advanced();
        run_task_highlevel();
    } else if (strcmp(argv[1], "-help") == 0) {
        print_help();
    } else {
        std::cerr << "未知参数: " << argv[1] << std::endl;
        print_help();
        return 1;
    }

    return 0;
}