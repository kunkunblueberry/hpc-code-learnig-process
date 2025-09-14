#ifndef VECTOR_LIB_H
#define VECTOR_LIB_H

#include <iostream>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>

// 模板元编程：判断是否为数值类型
template <class T>
struct IsNumeric {
    static constexpr bool value = std::is_arithmetic_v<T>;
};

// 固定长度向量类模板
template <class T, size_t Size>
class FixedVector {
private:
    T data[Size];  // 固定大小数组，无动态内存开销
public:
    // 初始化列表构造函数
    FixedVector(std::initializer_list<T> init) {
        // 编译期检查：仅允许数值类型
        static_assert(IsNumeric<T>::value, "FixedVector only supports numeric types!");
        //简单来说就是这里编译期间就要确定传入的数据是否正确，不正确之间退出
        //：：这个符号也是结构体静态变量的访问方式了，大概需要记住一下
        
        size_t i = 0;
        // 初始化列表赋值
        for (auto& val : init) {
            if (i < Size) data[i++] = val;
        }
        // 未初始化元素设为默认值
        for (; i < Size; i++) {
            data[i] = T{};
        }
    }

    // 基础打印函数（兼容早期任务）
    void print() const {
        std::cout << "[";
        for (size_t i = 0; i < Size; i++) {
            std::cout << data[i];
            if (i != Size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 模板成员函数：向量点积（支持不同类型）
    template <class U>
    auto dot(const FixedVector<U, Size>& other) const {
        using ResultType = decltype(T{} * U{});  // 推导结果类型
        ResultType res = ResultType{};
        for (size_t i = 0; i < Size; i++) {
            res += data[i] * other.data[i];
        }
        return res;
    }

    // 模板成员函数：带Debug模式的打印
    template <bool Debug = false>
    void print() const {
        if constexpr (Debug) {
            std::cout << "FixedVector<" << typeid(T).name() << ", " << Size << "> : [";
        } else {
            std::cout << "[";
        }
        for (size_t i = 0; i < Size; i++) {
            std::cout << data[i];
            if (i != Size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 归一化函数（仅支持float类型）
    void normalize() {
        if constexpr (std::is_same_v<T, float>) {
            auto len = sqrt(dot(*this));
            if (len < 1e-6f) {
                std::cerr << "Warning: Vector length is zero, cannot normalize!" << std::endl;
                return;
            }
            for (size_t i = 0; i < Size; i++) {
                data[i] /= len;
            }
        } else {
            std::cerr << "Error: normalize() only supports FixedVector<float, Size>!" << std::endl;
        }
    }

    // 友元声明：允许add函数访问私有成员data
    template <class U, size_t S>
    friend FixedVector<U, S> add(const FixedVector<U, S>& a, const FixedVector<U, S>& b);
};

// FixedVector加法重载
template <class T, size_t Size>
FixedVector<T, Size> add(const FixedVector<T, Size>& a, const FixedVector<T, Size>& b) {
    T res_data[Size];
    for (size_t i = 0; i < Size; i++) {
        res_data[i] = a.data[i] + b.data[i];
    }
    return FixedVector<T, Size>(res_data);
}

// std::vector加法重载
template <class T>
std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> res;
    size_t min_len = std::min(a.size(), b.size());
    res.reserve(min_len);  // 预分配内存优化
    for (size_t i = 0; i < min_len; i++) {
        res.push_back(a[i] + b[i]);
    }
    return res;
}

// 向量维度获取模板（元编程）
template <class Container>
struct VectorDim {
    static_assert(false, "VectorDim only supports FixedVector and std::vector!");
};

// FixedVector维度特化（编译期常量）
template <class T, size_t Size>
struct VectorDim<FixedVector<T, Size>> {
    static constexpr size_t value = Size;
};

// std::vector维度特化（运行时获取）
template <class T>
struct VectorDim<std::vector<T>> {
    static size_t value(const std::vector<T>& vec) {
        return vec.size();
    }
};

#endif // VECTOR_LIB_H



