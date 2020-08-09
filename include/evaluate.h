#pragma once


#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include <nnue_half_kp.h>
#include <board.h>
#include <move.h>
#include <move_orderer.h>

namespace chess{

template<typename T>
inline constexpr T epsilon = static_cast<T>(1e-4);

template<typename T>
inline constexpr T big_number = static_cast<T>(256);

template<typename T>
inline constexpr T mate_score = -static_cast<T>(8);

template<typename T>
inline constexpr T draw_score = static_cast<T>(0);

template<typename T>
T q_search(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta, int depth=0){
  constexpr int max_depth = 5;
  const auto list = bd.generate_moves();

  const bool is_check = bd.is_check();
  if(list.size() == 0 && is_check){ return mate_score<T>; }
  if(list.size() == 0) { return draw_score<T>; }
  if(hist.is_three_fold(bd.hash())){ return draw_score<T>; }
  
  const auto loud_list = list.loud_or_checking(bd);
  auto orderer = move_orderer(loud_list);

  const T static_eval = eval.propagate(bd.turn());
  if(depth >= max_depth){ return static_eval; }
  if(loud_list.size() == 0 || (static_eval > beta && !is_check)){ return static_eval; }
  
  T best_score = !is_check ? static_eval : mate_score<T>;
  alpha = std::max(alpha, best_score);
  
  auto _ = hist.scoped_push_(bd.hash());
  for(auto [idx, mv] : orderer){
    assert((mv != move::null()));
    if(best_score > beta){ break; }
    const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
    const board bd_ = bd.forward(mv);
    
    const T score = -q_search(hist, eval_, bd_, -beta, -alpha, depth + 1);
    alpha = std::max(alpha, score);
    best_score = std::max(best_score, score);
  }

  return best_score;
}

template<typename T>
struct evaluation_data{
  bool turn_;
  T value_{};
  std::vector<std::tuple<chess::move, T>> policy_{};
  
  evaluation_data<T>& append_(const move& mv, const T& logit){
    static constexpr T k = static_cast<T>(2.5);
    policy_.push_back(std::tuple(mv, std::exp(k * logit)));
    return *this;
  }
  
  evaluation_data<T>& set_score_(const T& logit){
    constexpr T scale = static_cast<T>(2);
    value_ = std::tanh(scale * logit);
    return *this;
  }
  
  evaluation_data<T>& normalize_(){
    const T Z = std::accumulate(policy_.cbegin(), policy_.cend(), T{}, [](const T& z, const auto& p){
      return z + std::get<T>(p);
    });
    std::for_each(policy_.begin(), policy_.end(), [Z](auto& p){
      std::get<1>(p) /= Z;
    });
    return *this;
  }
  
  evaluation_data(const bool turn) : turn_{turn} {}
};

template<typename T>
evaluation_data<T> evaluate(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd){
  const auto list = bd.generate_moves();

  const bool is_check = bd.is_check();
  if(list.size() == 0 && is_check){ return evaluation_data<T>(bd.turn()).set_score_(mate_score<T>); }
  if(list.size() == 0) { return evaluation_data<T>(bd.turn()).set_score_(draw_score<T>); }
  if(hist.is_three_fold(bd.hash())){ return evaluation_data<T>(bd.turn()).set_score_(draw_score<T>); }

  auto orderer = move_orderer(list);
  
  T best_score = mate_score<T>;
  evaluation_data<T> result(bd.turn());

  auto _ = hist.scoped_push_(bd.hash());
  for(auto [idx, mv] : orderer){
    assert((mv != move::null()));
    const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
    const board bd_ = bd.forward(mv);
    const T score = -q_search(hist, eval_, bd_, -big_number<T>, big_number<T>);
    result.append_(mv, score);
    best_score = std::max(best_score, score);
  }
  
  return result.set_score_(best_score).normalize_();
}

template<typename T>
std::ostream& operator<<(std::ostream& ostr, const evaluation_data<T>& data){
  ostr << "Q: " << data.value_ << '\n';
  auto cpy = data.policy_;
  std::sort(cpy.begin(), cpy.end(), [](const auto& a, const auto& b){
    return std::get<T>(a) > std::get<T>(b);
  });
  std::for_each(cpy.begin(), cpy.end(), [&](const auto& elem){
    const auto[mv, p] = elem;
    ostr << "  " << mv.name(data.turn_) << " => P: " << p << '\n';
  });
  return ostr;
}

}
