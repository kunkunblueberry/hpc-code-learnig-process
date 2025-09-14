cmake_minimum_required(VERSION 3.12)
project(HPC_Vector_Library)

# 设置C++标准（需C++17支持if constexpr等特性）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项：开启警告和优化
if(MSVC)
    # Visual Studio编译器
    add_compile_options(/W4)  # 开启所有警告
else()
    # GCC/Clang编译器
    add_compile_options(-Wall -Wextra -Wpedantic)  # 严格警告
    add_compile_options(-O3)  # 最高级优化（HPC常用）
endif()

# 生成可执行文件
add_executable(vector_test main.cpp)

# 链接数学库（Linux/macOS需要，Windows自动链接）
if(NOT MSVC)
    target_link_libraries(vector_test PRIVATE m)
endif()
