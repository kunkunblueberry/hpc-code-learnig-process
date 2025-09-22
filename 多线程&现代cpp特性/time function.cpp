//计时函数
auto t0 = std::chrono::steady_clock::now();

// 耗时操作

auto t1 = std::chrono::steady_clock::now();
auto dt = t1 - t0;
int64_t ms = std::chrono::duration_cast<chrono::milliseconds>(dt).count();
std::cout << "time elapsed:" << ms << "ms" << std::endl;
