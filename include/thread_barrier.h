#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cassert>

namespace util{

struct thread_barrier{
  size_t expected_{};
  size_t reached_{0};
  std::mutex wait_mutex_{};
  std::condition_variable cv_{};
  
  void arrive_and_wait(){
    std::unique_lock<std::mutex> wait_lk(wait_mutex_);
    ++reached_;
    assert((reached_ <= expected_));
    if(reached_ == expected_){
      cv_.notify_all();
    }else{
      cv_.wait(wait_lk, [this]{ return expected_ == reached_; });
    }
  }
  
  thread_barrier(const size_t& expected): expected_{expected} {}
};

}
