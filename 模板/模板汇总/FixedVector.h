// #include <iostream>
// #include <initializer_list>
// #include <type_traits> // 用于类型判断
// #include <math>

// // 模板类：FixedVector<T, Size>，T为数据类型，Size为向量维度（编译期常量）
// template <class T, size_t Size>
// class FixedVector {
// private:
//     T data[Size]; // 固定大小数组，无动态内存分配（HPC优势）
// public:
//     // 构造函数：用初始化列表初始化向量
//     FixedVector(std::initializer_list<T> init) {
//         // 编译期检查：仅允许数值类型（int、float、double等）
//         static_assert(std::is_arithmetic_v<T>, "FixedVector only supports numeric types!");
        
//         // 将初始化列表的值赋值给data数组
//         size_t i = 0;
//         for (auto& val : init) {
//             if (i < Size) { // 避免初始化列表长度超过向量维度
//                 data[i++] = val;
//             }
//         }
//         // 未被初始化的元素赋值为0（数值类型默认值）
//         for (; i < Size; i++) {
//             data[i] = T{};
//         }
//     }
//             // 公有成员函数：打印向量
//     void print() const {
//         std::cout << "[";
//         for (size_t i = 0; i < Size; i++) {
//             std::cout << data[i];
//             if (i != Size - 1) {
//                 std::cout << ", ";
//             }
//         }
//         std::cout << "]" << std::endl;
//     }

//     //实现向量点积
//     template<class U>
//     auto dot(const FixedVector<U,Size>&other)const{
//         using ResultType = decltype(T{} * U{});
//         //step1 课上讲过这个，解释这句
//         ResultType res = ResultType{}; // 初始化结果为0（适配任意数值类型）

//         for (size_t i = 0; i < Size; i++) {
//         res += data[i] * other.data[i];
//     }
//     return res;
//     }


//     //高阶任务：HPC 优化（编译期分支与类型特化）
//     // （接FixedVector类内dot函数后，补充以下成员函数）
//     // TODO：参考文档“if constexpr编译期分支”知识，实现模板版print函数（支持Debug模式）
//     // 功能：Debug=true时打印类型/维度信息，Debug=false时仅打印值（无运行时开销）
//     // 模板参数：bool Debug（默认false，参考文档“模板默认参数”）
//     template <bool Debug = false>
//     void print() const {
//         if constexpr (Debug) {
//             // 补充代码1：Debug模式打印格式："FixedVector<类型, 维度> : ["（用typeid(T).name()获取类型名）
//             //之间输入T没用，用typeid(T).name
//             std::cout<<"FixedVector<"<<typeid(T).name()<<","<<Size<<"> [";
//         } else {
//             std::cout<<"[";
//             // 补充代码2：Release模式仅打印"["
//         }
//         // 补充代码3：遍历data数组打印元素（同基础版print）
//         for(int i=0;i<Size;i++){
//             cout<<data[i];
//             if(i!=Size-1)   std::cout<<",";
//         }
//         std::cout << "]" << std::endl;
//     }

//     void normalize() {
//         // 补充代码1：用if constexpr判断T是否为float（参考文档“编译期分支”）
//         if constexpr (/* 补充判断条件：T是否为float */typeid(T).name==float) {
//             // 补充代码2：调用dot(*this)计算向量长度平方，再用sqrt求长度（需包含<cmath>）
//             // 补充代码3：避免除以0（长度小于1e-6f时提示警告并返回）
//             // 补充代码4：逐元素除以长度，实现归一化

//             //计算向量点积
//             auto len=sqrt(dot(this->data));
//             if(len<1e-6f){
//                 std::cerr << "Warning: Vector length is zero, cannot normalize!" << std::endl;
//             return;
//             }
//             for(size_t i=0;i<Size;i++){
//                 data[i]/=len;
//             }
//         } else {
//             // 补充代码5：非float类型时，打印错误提示："normalize only supports float type!"
//         std::cerr << "Error: normalize() only supports FixedVector<float, Size>!" << std::endl;
//         }
//     }
// };

//     // TODO：参考文档“模板函数重载”知识，实现FixedVector版本add函数
//     // 功能：两个同类型、同维度FixedVector的逐元素加法
//     // 要求：模板参数T（数据类型）、Size（维度），返回新的FixedVector
//     template <class T, size_t Size>
//     FixedVector<T, Size> add(const FixedVector<T, Size>& a, const FixedVector<T, Size>& b) {
//     // 补充代码1：遍历a.data与b.data，计算res_data[i] = a.data[i] + b.data[i]
//     // 补充代码2：返回用res_data初始化的FixedVector对象
//     FixedVector<T,Size>res_data;
//     for(int i=0;i<size;i++){
//         res_data.data[i]=a.data[i]+b.data[i];
//         }
//         return res_data;
//     }