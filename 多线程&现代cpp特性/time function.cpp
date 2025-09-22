//计时函数
auto t0 = std::chrono::steady_clock::now();

// 耗时操作

auto t1 = std::chrono::steady_clock::now();
auto dt = t1 - t0;
int64_t ms = std::chrono::duration_cast<chrono::milliseconds>(dt).count();
std::cout << "time elapsed:" << ms << "ms" << std::endl;



std::this_thread::sleep_for
std::this_thread::sleep_for(std::chrono::milliseconds(400));  //让线程休眠 400 毫秒，替代 Unix 专属的usleep
sleep_for在this_thread命名空间下面哦，完整要这么写


std::this_thread::sleep_until
sleep_until(steady_clock::now() + milliseconds(400))
