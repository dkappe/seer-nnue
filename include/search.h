#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <optional>

#include <thread_barrier.h>
#include <nnue_util.h>
#include <nnue_half_kp.h>
#include <zobrist_util.h>
#include <position_history.h>
#include <board.h>
#include <evaluate.h>

namespace mcts{

template<typename T>
inline constexpr T pb_c_base = static_cast<T>(19652);

template<typename T>
inline constexpr T pb_c_init = static_cast<T>(3.2);

template<typename T>
T cpuct(const size_t& nodes){
  return std::log((pb_c_base<T> + nodes + 1) / pb_c_base<T>) + pb_c_init<T>;
}

template<typename T>
struct node{
  static constexpr T negative_one = static_cast<T>(-1);
  static constexpr T zero = static_cast<T>(0);

  std::atomic_bool virtual_loss_{false};
  std::atomic_bool leaf_{true};
  std::atomic_bool active_{false};
  std::atomic<zobrist::hash_type> hash_{};

  std::mutex params_mutex_;
  size_t n_{0};
  T q_{negative_one};
  T p_{zero};

  bool is_virtual_loss() const { return virtual_loss_.load(std::memory_order_relaxed); }
  
  node<T>& set_virtual_loss(){
    virtual_loss_.store(true, std::memory_order_relaxed);
    return *this;
  }
  
  node<T>& clear_virtual_loss(){
    virtual_loss_.store(false, std::memory_order_relaxed);
    return *this;
  }

  bool is_leaf() const { return leaf_.load(std::memory_order_relaxed); }

  node<T>& set_leaf(){
    leaf_.store(true, std::memory_order_relaxed);
    return *this;
  }

  node<T>& clear_leaf(){
    leaf_.store(false, std::memory_order_relaxed);
    return *this;
  }

  bool active() const { return active_.load(std::memory_order_relaxed); }
  zobrist::hash_type hash() const { return hash_.load(std::memory_order_relaxed); }

  node<T>& make_active(){
    active_.store(true, std::memory_order_relaxed);
    return *this;
  }

  node<T>& make_inactive(){
    active_.store(false, std::memory_order_relaxed);
    return *this;
  }
  
  T n() const { return n_; }
  T q() const { return q_; }
  T p() const { return p_; }

  T u(const size_t& n_ancestor) const {
    const T cpuct_ = cpuct<T>(n_ancestor);
    const T q_prime = is_virtual_loss() ? negative_one : -q();
    return q_prime + cpuct_ * p() * std::sqrt(static_cast<T>(n_ancestor)) / static_cast<T>(1 + n()); 
  }

  node<T>& init(const zobrist::hash_type& hash, const T& p){
    set_leaf();
    clear_virtual_loss();
    make_active();
    hash_.store(hash, std::memory_order_relaxed);
    {
      std::scoped_lock<std::mutex> lk(params_mutex_);
      p_ = p;
      q_ = negative_one;
      n_ = 0;
    }
    return *this;
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& ostr, const node<T>& nd){
  return ostr
    << "node<T>(active=" << nd.active() << ", hash=" << nd.hash()
    << ", n=" << nd.n() << ", q=" << nd.q() << ", p=" << nd.p() << ")";
}

template<typename T>
struct bucket{
  using iterator = typename std::vector<node<T>>::iterator;
  iterator begin_;
  iterator end_;
  
  const iterator& begin() const { return begin_; }
  const iterator& end() const { return end_; }
  
  iterator find(const zobrist::hash_type& hash) const {
    return std::find_if(begin(), end(), [hash](const auto& elem){
      return elem.hash() == hash;
    });
  }
  
  iterator maybe_insert(const zobrist::hash_type& hash, const T& prior){
    auto iter = find(hash);
    if(iter == end()){
      iter = std::find_if(begin(), end(), [hash](const auto& elem){
        return !elem.active();
      });
      if(iter != end()){
        iter -> init(hash, prior);
      }
    }
    iter -> make_active();
    return iter;
  }
};

template<typename T>
struct child{
  using iterator = typename std::vector<node<T>>::iterator;
  chess::move move_;
  iterator iter_;
};

template<typename T>
struct roll_out{
  chess::position_history visited_;
  chess::board leaf_;
  chess::evaluation_data<T> result_;
};

template<typename T>
struct puct_tree{
  static constexpr size_t mb = (static_cast<size_t>(1) << static_cast<size_t>(20)) / sizeof(node<T>);
  using iterator = typename std::vector<node<T>>::iterator;
  
  std::atomic<size_t> sel_depth_{0};
  
  size_t bucket_size_;
  std::vector<node<T>> nodes_;
  const nnue::half_kp_weights<T>* src_;
  
  chess::position_history history_;
  chess::board root_{chess::board::start_pos()};
  
  size_t num_buckets() const {
    const size_t num_buckets_ = nodes_.size() / bucket_size_;
    assert((num_buckets_ * bucket_size_ == nodes_.size()));
    return num_buckets_;
  }
  
  bucket<T> get_bucket(const zobrist::hash_type& hash){
    const size_t bucket_idx = hash % num_buckets();
    auto first = nodes_.begin();
    auto last = nodes_.begin();
    std::advance(first, bucket_idx * bucket_size_);
    std::advance(last, bucket_idx * bucket_size_ + bucket_size_);
    return bucket<T>{first, last};
  }

  puct_tree<T>& update_sel_depth(const size_t new_depth){
    if(new_depth > sel_depth_.load(std::memory_order_relaxed)){
      sel_depth_.store(new_depth, std::memory_order_relaxed);
    }
    return *this;
  }

  puct_tree<T>& resize_bucket(const size_t& new_bucket_size){
    const size_t current_size = nodes_.size();
    const size_t new_size = (current_size - (current_size % new_bucket_size));
    nodes_.clear();
    bucket_size_ = new_bucket_size;
    nodes_ = std::vector<node<T>>(new_size);
    return *this;
  }

  puct_tree<T>& resize_tree(const size_t& mem){
    nodes_.clear();
    nodes_ = std::vector<node<T>>(mem * mb - ((mem * mb) % bucket_size_));
    get_bucket(root_.hash()).maybe_insert(root_.hash(), static_cast<T>(1));
    return *this;
  }

  puct_tree<T>& set_weights_src(const nnue::half_kp_weights<T>* src){
    src_ = src;
    return *this;
  }

  const chess::board& root() const { return root_; }
  const chess::position_history& history() const { return history_; }
  
  puct_tree<T>& set_root(const chess::position_history& hist, const chess::board& root){
    sel_depth_.store(0, std::memory_order_relaxed);
    history_ = hist;
    root_ = root;
    std::for_each(nodes_.begin(), nodes_.end(), [](auto& elem){ elem.make_inactive(); });
    get_bucket(root_.hash()).maybe_insert(root_.hash(), static_cast<T>(1));
    return *this;
  }
  
  template<typename F>
  std::optional<child<T>> max_child(const chess::board& bd, F&& criterion){
    typename std::result_of<F(node<T>)>::type criterion_max{};
    std::optional<child<T>> best{std::nullopt};
    for(const auto& mv : bd.generate_moves()){
      const zobrist::hash_type hash = bd.forward(mv).hash();
      const auto bucket = get_bucket(hash);
      const auto iter = bucket.find(hash);
      const T criterion_curr = criterion(*iter);
      if(iter != bucket.end() && (!best.has_value() || (criterion_curr > criterion_max))){
        criterion_max = criterion_curr;
        best = child<T>{mv, iter};
      }
    }
    return best;
  }
  
  roll_out<T> sample(){
    chess::position_history hist = history();
    chess::position_history visited{};
    
    chess::board bd = root();
    nnue::half_kp_eval eval(src_);
    bd.show_init(eval);
    iterator bd_iter = get_bucket(bd.hash()).find(bd.hash());

    while(!hist.is_three_fold(bd.hash()) && !(bd_iter -> is_leaf())){
      hist.push_(bd.hash());
      visited.push_(bd.hash());
      
      const auto next = max_child(bd, [&bd_iter](const node<T>& nd){
        return nd.u(bd_iter -> n());
      });
      
      if(next.has_value()){
        const auto[mv, iter] = *next;
        
        iter -> set_virtual_loss();
        iter -> make_active();
        
        eval = bd.half_kp_updated(mv, eval);
        bd = bd.forward(mv);
        bd_iter = iter;
      }else{ break; }
    }
    
    //add leaf position to visited nodes so that its stats will be updated
    visited.push_(bd.hash());
    
    update_sel_depth(visited.history_.size());
    const auto eval_data = chess::evaluate(hist, eval, bd);
    return roll_out<T>{visited, bd, eval_data};
  }

  puct_tree<T>& back_up(const roll_out<T>& data){
    for(const auto&[mv, prior] : data.result_.policy_){
      const zobrist::hash_type hash = data.leaf_.forward(mv).hash();
      get_bucket(hash).maybe_insert(hash, prior);
    }

    T value = data.result_.value_;
    for(auto hash_iter = data.visited_.history_.rbegin(); hash_iter != data.visited_.history_.rend(); ++hash_iter){
      const zobrist::hash_type hash = *hash_iter;
      const auto bucket = get_bucket(hash);
      auto iter = bucket.find(hash);
      assert((iter != bucket.end()));
      iter -> clear_leaf();
      iter -> clear_virtual_loss();
      {
        std::scoped_lock<std::mutex> lk(iter -> params_mutex_);
        const T w = (iter -> q()) * (iter -> n());
        iter -> q_ = (w + value) / static_cast<T>(1 + iter -> n());
        ++(iter -> n_);
      }
      value = -value;
    }

    return *this;
  }
  
  puct_tree<T>& batched_update(const size_t& batch_size){
    util::thread_barrier barrier(batch_size);
    auto work = [&barrier, this]{
      const roll_out<T> data = sample();
      barrier.arrive_and_wait();
      back_up(data);
    };
    std::vector<std::thread> threads(batch_size);
    std::for_each(threads.begin(), threads.end(), [work](std::thread& th){ th = std::move(std::thread(work)); });
    std::for_each(threads.begin(), threads.end(), [](std::thread& th){ th.join(); });
    return *this;
  }
  
  size_t nodes(){
    const zobrist::hash_type hash = root_.hash();
    return get_bucket(hash).find(hash) -> n();
  }
  
  size_t sel_depth(){
    return sel_depth_.load(std::memory_order_relaxed);
  }

  size_t est_depth(){
    constexpr size_t avg_branching_factor = 8;
    const size_t node_count = nodes();
    return static_cast<size_t>(std::log(1 + node_count) / std::log(avg_branching_factor));
  }
  
  T score(){
    const zobrist::hash_type hash = root_.hash();
    return get_bucket(hash).find(hash) -> q();
  }
  
  std::string pv_string(){
    chess::board bd = root();
    chess::position_history pv_data{};
    std::string pv{};
    while(!pv_data.is_three_fold(bd.hash())){
      pv_data.push_(bd.hash());
      const auto child = max_child(bd, [](const node<T>& nd){ return nd.n(); });
      if(child.has_value()){
        pv += (child -> move_).name(bd.turn()) + " ";
        bd = bd.forward(child -> move_);
      }else{ break; }
    }
    return pv;
  }
  
  chess::move best_move(){
    const auto best = max_child(root(), [](const node<T>& nd){ return nd.n(); });
    if(best.has_value()){
      return best.value().move_;
    }else{
      return chess::move::null();
    }
  }

  puct_tree(size_t mem, size_t bucket_size, const nnue::half_kp_weights<T>* src) : 
    bucket_size_{bucket_size},
    nodes_(mem * mb - ((mem * mb) % bucket_size)),
    src_{src} {}
};

}
