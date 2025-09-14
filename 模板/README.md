# FixedVector 向量运算库 README.md 📚

## 一、项目概述 ✨
本项目基于C++模板编程实现`FixedVector`固定维度向量类，核心提供**向量点积**、**向量加法**、**向量归一化**及**条件打印**功能，旨在帮助掌握模板成员函数、模板函数重载、编译期分支等核心技术，理解高性能计算（HPC）中"类型适配"与"编译期优化"逻辑，避免动态内存分配损耗，适配数值计算场景。


## 二、核心知识点回顾 🧠
| 知识点                | 核心作用                                                                 | 应用场景举例                          |
|-----------------------|--------------------------------------------------------------------------|---------------------------------------|
| 模板成员函数          | 类内定义模板函数，支持不同类型向量间运算（如`int`与`float`向量点积）     | `FixedVector<int,3>`与`FixedVector<float,3>`点积 |
| 类型推导（`decltype`）| 自动推导表达式结果类型，避免手动指定类型的兼容性问题                     | 推导`T*U`类型作为点积返回类型         |
| 模板函数重载          | 为不同容器（`FixedVector`/`std::vector`）实现同名函数，编译器自动匹配     | 分别为两种向量实现`add`函数           |
| 编译期分支（`if constexpr`） | 编译阶段确定代码分支，消除运行时开销，实现"零成本抽象"                   | Debug/Release模式差异化打印           |
| **类型萃取（Type Traits）** | **编译期提取类型信息（如向量维度），统一接口并支持编译期错误检查**       | **统一获取`FixedVector`（编译期维度）与`std::vector`（运行时维度）维度** |
| 编译期断言（`static_assert`） | 编译阶段验证常量配置合法性，提前暴露错误（如类型、维度非法）             | 限制`FixedVector`仅支持数值类型       |


## 三、类型萃取（Type Traits）深度解析 🔍

### 3.1 什么是类型萃取？
类型萃取是C++模板元编程的核心技术，通过"结构体一层一层套"的特化方式，在**编译时提取类型信息**，同时保证接口一致性和错误检查。

### 3.2 为什么需要类型萃取？
主要解决两种向量的维度获取差异：
- `FixedVector`：维度是**编译期常量**（模板参数`Size`）
- `std::vector`：维度是**运行时变量**（`size()`返回值）

### 3.3 核心优势 🌟

#### 优势1：统一接口，适配两种维度特性
如果用普通函数，需要两个完全不同的接口：
```cpp
// FixedVector（编译期维度）
template <class T, size_t Size>
constexpr size_t getDim(const FixedVector<T, Size>&) { return Size; }

// std::vector（运行时维度）
template <class T>
size_t getDim(const std::vector<T>& vec) { return vec.size(); }
```

而类型萃取用统一接口解决这个问题：
```cpp
// FixedVector：编译期直接获取
constexpr size_t dim1 = VectorDim<FixedVector<int, 3>>::value;

// std::vector：运行时通过参数获取
std::vector<int> vec(5);
size_t dim2 = VectorDim<std::vector<int>>::value(vec);
```

在模板中使用时自动匹配：
```cpp
template <class Vector>
void process(Vector& vec) {
    // 编译期/运行时自动适配
    auto dim = VectorDim<Vector>::value(vec); 
}
```

#### 优势2：清晰的编译期错误检查
基类模板中`static_assert(false, ...)`确保不支持的类型编译时报错：
```cpp
VectorDim<std::list<int>>::value;  // 编译报错："仅支持FixedVector和std::vector!"
```

相比函数重载的"找不到匹配函数"，错误提示更明确！

#### 优势3：支持编译期计算
结构体特化让编译器在编译阶段完成计算：
```cpp
// 编译期确定数组大小，分配栈内存
std::array<int, VectorDim<FixedVector<float, 5>>::value> arr;
```

这是普通函数无法实现的（函数结果不能作为编译期常量）！

### 3.4 本质：类型到信息的映射 🗺️
- 对`FixedVector<T, Size>` → 映射到编译期常量`Size`
- 对`std::vector<T>` → 映射到运行时函数`vec.size()`
- 对其他类型 → 映射到编译错误

既保证接口一致，又发挥编译期计算优势，这就是模板元编程的精妙之处！


## 四、环境依赖 🛠️
- 编译器：支持C++17及以上标准（GCC 8+、Clang 7+、MSVC 2019+）
- 头文件：`iostream`（输出）、`initializer_list`（初始化列表）、`type_traits`（类型判断）、`cmath`（数学计算）


## 五、类与函数实现指南（考点聚焦） 🎯

### 5.1 核心类：`FixedVector<T, Size>`
模板类定义：`template <class T, size_t Size> class FixedVector`，其中`T`为数值类型，`Size`为编译期固定维度（HPC核心设计，无动态内存分配）。

#### 5.1.1 基础功能（已实现，考点参考）
- **构造函数（初始化列表）**：考点为`static_assert`编译期类型检查（限制数值类型）、初始化列表遍历与未初始化元素置0逻辑。
- **基础打印函数**：考点为数组遍历与格式化输出，为后续模板版打印做铺垫。


#### 5.1.2 考点1：模板成员函数 - 向量点积
**功能**：支持不同类型`FixedVector`计算点积，返回类型自动推导。  
**核心考点**：模板成员函数定义、`decltype`类型推导、跨类型数值运算兼容性。  
**实现伪代码**：
```cpp
template<class U>
auto dot(const FixedVector<U, Size>& other) const {
    using ResultType = decltype(T{} * U{});  // 自动推导结果类型
    ResultType res = ResultType{};          // 初始化结果
    循环累加元素乘积;                        // 遍历计算
    return res;
}
```


#### 5.1.3 考点2：模板版打印函数（Debug模式支持）
**功能**：通过模板参数`Debug`控制打印模式，编译期分支优化。  
**核心考点**：模板默认参数、`if constexpr`编译期分支、`typeid`类型信息获取。  


#### 5.1.4 考点3：向量归一化（`normalize()`）
**功能**：仅支持`float`类型向量归一化，避免除以0错误。  
**核心考点**：`if constexpr`类型判断、`dot`函数复用、浮点数精度处理。  


#### 5.2 考点4：模板函数重载 - FixedVector加法
**功能**：同类型、同维度`FixedVector`逐元素相加，返回新向量。  
**核心考点**：全局模板函数定义、模板函数重载匹配、常量引用参数（避免拷贝开销）。  


## 六、HPC优化核心思想（考点延伸） 💡
本项目HPC优化设计是核心考点延伸，重点理解"编译期优化"：      **(模板的核心就是，能在编译期完成的，绝对不拖到代码运行的时候进行！！！)**
1. **无动态内存分配**：用栈上数组`T data[Size]`存储，避免`std::vector`堆内存开销（考点：HPC内存优化逻辑）。
2. **编译期常量利用**：维度`Size`为模板参数，可用于数组大小、`static_assert`检查（考点：编译期常量的应用场景）。
3. **零成本抽象**：`if constexpr`消除运行时分支，`decltype`保证类型兼容（考点：C++模板的"零开销"设计理念）。


## 七、常见问题与考点关联 ❓
| 问题现象                                  | 关联考点                                  | 解决方案                                  |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| 编译报错"仅支持数值类型"                  | `static_assert`类型检查                   | 确保`T`为`int`/`float`等数值类型          |
| 点积结果类型错误                          | `decltype`类型推导                        | 用`decltype(T{}*U{})`自动推导返回类型     |
| 归一化函数类型报错                        | `if constexpr`类型判断                    | 仅对`FixedVector<float, Size>`调用        |
| `add`函数匹配失败                          | 模板函数重载与参数匹配                    | 确保两个向量类型、维度完全一致            |



