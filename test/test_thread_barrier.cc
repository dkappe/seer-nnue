#include <iostream>
#include <vector>
#include <thread>

#include <thread_barrier.h>

int main(){
  constexpr size_t num_threads = 1000;
  util::thread_barrier barrier(num_threads);
  std::vector<std::thread> threads{};
  for(size_t i(0); i < num_threads; ++i){ threads.emplace_back([&barrier]{
    std::cout << "-> step 1\n";
    barrier.arrive_and_wait();
    std::cout << "---------> step 2\n";
  }); }
  for(auto& th : threads){ th.join(); }
}
