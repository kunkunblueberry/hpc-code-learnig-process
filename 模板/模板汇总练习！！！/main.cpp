#include <iostream>
#include <vector>
#include <variant>

// 请修复这个函数的定义：10 分
std::ostream &operator<<(std::ostream &os, std::vector<T> const &a) {
    os << "{";
    for (size_t i = 0; i < a.size(); i++) {
        os << a[i];
        if (i != a.size() - 1)
            os << ", ";
    }
    os << "}";
    return os;
}

// 请修复这个函数的定义：10 分
template <class T1, class T2>
std::vector<T0> operator+(std::vector<T1> const &a, std::vector<T2> const &b) {
    // 请实现列表的逐元素加法！10 分
    // 例如 {1, 2} + {3, 4} = {4, 6}
    if(a.size()!=b.size()){
        printf("长度不等，无法相加");
        return{};
    }
    else
    {
        vector<T0>c;
        for(int i=0;i<a.szie();i++){
            c.push_back(a[i]+b[i]);
        }
        return c;
    }
}

template <class T1, class T2>
std::variant<T1, T2> operator+(std::variant<T1, T2> const &a, std::variant<T1, T2> const &b) {
    // 请实现自动匹配容器中具体类型的加法！10 分
    variant<T1,T2>ans;
    if(std::hole_alternative<T1>(a)&&std::hole_alternative<T1>(b)){
    auto test1=std::get<T1>(a);
    auto test2=std::get<T1>(a);
    ans=test1+test3;
    return ans;
    }
    else if(std::hole_alternative<T2>(a)&&std::hole_alternative<T2>(b)){
        auto test3=std::get<T2>(b);
        auto test4=std::get<T2>(b);
        ans=test3+test4;
        return ans;
    }
    else{
        std::cout<<"类型不匹配，无法相加";
    }

    // std::visit([](const auto&value){
    //     auto test1=value;
    // },a);

    // std::visit([](const auto&val){
    //     auto test2=value;
    // },b);
    // std::variant<T1,T2>ans;
    // ans=test1+test2;
    // return ans;

    // std::variant<T1,T2>ans;
    // //这里&表示能够引用到外部，把读取到的能够放在函数外部，防止局部变量退出就消失了
    // std::visit([&](const auto&t1,const auto&t2){
    //     ans=t1+t1;
    // },a,b);

    std::variant<T1,T2>ans;
    return std::visit([&](auto const &t1,auto const &t2)->std::variant<T1,T2>{
    return t1+t2
    },a,b);
    
    //大概这四种写法
}

template <class T1, class T2>
std::ostream &operator<<(std::ostream &os, std::variant<T1, T2> const &a) {
    // 请实现自动匹配容器中具体类型的打印！10 分
    if(std::holds_alternative<T1>(a)){
        os<<std::get<T1>(a);
    }else{
        os<<std::get<T1>(a);    //简单来说，variant只能在两者之间选一个，如果这里存储的是T1，输出T2就是错误行为
    }
}

int main() {
    std::vector<int> a = {1, 4, 2, 8, 5, 7};
    std::cout << a << std::endl;
    std::vector<double> b = {3.14, 2.718, 0.618};
    std::cout << b << std::endl;
    auto c = a + b;

    // 应该输出 1
    std::cout << std::is_same_v<decltype(c), std::vector<double>> << std::endl;

    // 应该输出 {4.14, 6.718, 2.618}
    std::cout << c << std::endl;

    std::variant<std::vector<int>, std::vector<double>> d = c;
    std::variant<std::vector<int>, std::vector<double>> e = a;
    d = d + c + e;

    // 应该输出 {9.28, 17.436, 7.236}
    std::cout << d << std::endl;

    return 0;
}
