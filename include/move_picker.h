#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <cmath>

#include <move.h>
#include <nnue_util.h>
#include <nnue_half_kp.h>

namespace chess{

template<typename T>
struct move_picker{
  size_t idx_{0};
  
  T log_sum_exp_{};
  nnue::stack_vector<T, move_list::max_branching_factor> logits_{};
  
  move_list list_;
  
  
  
  size_t max_remaining_idx_() const {
    const auto it = std::max_element(std::begin(logits_.data) + idx_, std::begin(logits_.data) + list_.size());
    return std::distance(std::begin(logits_.data), it);
  }
  
  bool has_value() const { return idx_ < list_.size(); }
  
  void step(){
    ++idx_;
    const auto best = max_remaining_idx_();
    std::swap(list_.at(best), list_.at(idx_));
    std::swap(logits_.at(best), logits_.at(idx_));
  }
  
  std::tuple<T, move> val() const {
    return std::tuple(logits_.at(idx_) - log_sum_exp_, list_.at(idx_));
  }
  
  move_picker(const nnue::stack_vector<T, nnue::move_dim>& raw_logits, const move_list& list) : list_{list}{
    constexpr size_t num_squares = 64;
    
    T max_logit{std::numeric_limits<T>::lowest()}; 
    std::transform(list.begin(), list.end(), std::begin(logits_.data), [&max_logit, &raw_logits](const move& mv){
      //necessary to account for seer's unusual rotated board representation
      auto convert_idx = [](const size_t& idx){ return (7 - (idx % 8)) + 8 * (idx / 8); };
      const size_t from_idx = convert_idx(mv.from().index());
      const size_t to_idx = convert_idx(mv.to().index());
      
      const T logit = raw_logits.at(from_idx) + raw_logits.at(num_squares + to_idx);
      max_logit = std::max(logit, max_logit);
      return logit;
    });
    
    log_sum_exp_ = max_logit + std::log(std::accumulate(std::begin(logits_.data), std::begin(logits_.data) + list.size(), static_cast<T>(0), [this, max_logit](T sum, T next){
      return sum + std::exp(next - max_logit);
    }));
    
    const auto best = max_remaining_idx_();
    std::swap(list_.at(best), list_.at(idx_));
    std::swap(logits_.at(best), logits_.at(idx_));
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& ostr, move_picker<T> picker){
  for(size_t i(0); picker.has_value(); picker.step(), ++i){
    const auto[log_prob, mv] = picker.val();
    ostr << i << ". " << mv << " <- " << std::exp(log_prob) << '\n';
  }
  return ostr;
}

} 
