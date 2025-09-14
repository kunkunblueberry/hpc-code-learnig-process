任务 —— 模板成员函数与重载（向量运算）
任务目标

为FixedVector添加向量点积、向量加法功能，掌握模板成员函数与模板函数重载，理解 HPC 中 “类型适配” 与 “编译期优化” 的实现方式。

知识点回顾
模板成员函数：在类内部定义模板函数（如dot），支持不同类型向量间的运算（如int向量与float向量点积）。
类型推导：decltype可根据表达式推导类型（如decltype(T{} * U{})推导点积结果类型）。
模板函数重载：为不同容器（FixedVector与std::vector）实现同名add函数，编译器自动匹配对应版本。

任务步骤
Step 1：实现模板成员函数 —— 向量点积
在FixedVector类中添加dot模板成员函数，支持与不同类型的FixedVector计算点积（如FixedVector<int,3>与FixedVector<float,3>）。


1. 编译时常量的合法性验证----asster用法解析
确保编译时定义的常量、配置符合预期，避免 “运行时才发现配置错误”。
cpp
constexpr int BUFFER_SIZE = 1024;
static_assert(BUFFER_SIZE >= 512, "缓冲区大小不能小于512字节");

constexpr int VERSION = 2;
static_assert(VERSION > 0, "版本号必须为正数");

2,,vector_lib.h后半部分代码说明
这种 “用结构体一层一层套” 的写法，是 C++ 模板元编程中类型萃取（Type Traits） 的典型实现方式，
核心目的是在编译时 “提取” 类型信息，同时保证接口一致性和编译期错误检查。
之所以这么设计，而不是用普通函数，本质是因为要处理两种完全不同的场景：
FixedVector：维度是编译期常量（模板参数Size）
std::vector：维度是运行时变量（size()返回值）
核心原因 1：统一接口，适配两种维度特性
如果用普通函数，你可能需要写两个完全不同的接口：

// 针对FixedVector（编译期维度）
template <class T, size_t Size>
constexpr size_t getDim(const FixedVector<T, Size>&) {
    return Size;
}

// 针对std::vector（运行时维度）
template <class T>
size_t getDim(const std::vector<T>& vec) {
    return vec.size();
}
但这样会有一个问题：当在模板中需要同时处理两种向量时，无法用统一的方式获取维度。例如：

template <class Vector>
void process(Vector& vec) {
    // 这里需要获取维度，但getDim的调用方式在编译期/运行时不同
    constexpr size_t dim1 = getDim(vec);  // 对std::vector无效（不是常量）
    size_t dim2 = getDim(vec);            // 对FixedVector会丢失constexpr特性
}
而用结构体封装后，接口可以统一：

// 对FixedVector：编译期直接获取
constexpr size_t dim1 = VectorDim<FixedVector<int, 3>>::value;

// 对std::vector：运行时通过参数获取
std::vector<int> vec(5);
size_t dim2 = VectorDim<std::vector<int>>::value(vec);
在模板中使用时，能根据实际类型自动匹配正确的获取方式：

template <class Vector>
void process(Vector& vec) {
    // 编译期维度（FixedVector）会自动用constexpr版本
    // 运行时维度（std::vector）会自动用带参数的版本
    auto dim = VectorDim<Vector>::value(vec); 
}
核心原因 2：编译期错误检查
基类模板中static_assert(false, ...)的作用是：如果用户传入了不支持的类型，编译时直接报错。
例如，当用户尝试对std::list使用VectorDim时：

VectorDim<std::list<int>>::value;  // 编译报错："VectorDim only supports FixedVector and std::vector!"
如果不用结构体特化，而用函数重载，虽然也能实现类似功能，但错误提示会非常模糊（通常是 “找不到匹配的函数重载”），而static_assert可以给出明确的错误原因。
核心原因 3：模板元编程的 “编译期计算” 特性
结构体特化是 C++ 模板元编程的基础技术，它能让编译器在编译阶段就完成类型判断和信息提取，而不是等到运行时。
对于FixedVector，维度Size是模板参数，属于编译期常量。用结构体的static constexpr成员，可以直接在编译期获取这个值，例如：

// 编译期就能确定数组大小，分配栈内存
std::array<int, VectorDim<FixedVector<float, 5>>::value> arr;

这种编译期计算的能力，是普通函数无法实现的（函数调用的结果在编译期无法作为数组大小等常量参数）。
总结：结构体特化的本质是 “类型到信息的映射”
这种写法的核心思想是：将 “类型” 作为输入，通过模板特化 “映射” 到对应的信息（编译期常量或运行时函数）。
对FixedVector<T, Size>类型，映射到编译期常量Size
对std::vector<T>类型，映射到运行时函数vec.size()
对其他类型，映射到编译错误
这种设计既保证了接口一致性（用VectorDim<Type>::value统一访问），又充分利用了编译期计算的优势，
同时提供了清晰的错误提示。这正是 C++ 模板元编程的精妙之处 —— 用类型系统驱动编译期逻辑。