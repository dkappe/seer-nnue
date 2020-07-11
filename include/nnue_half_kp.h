#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include <weights_streamer.h>
#include <nnue_util.h>
#include <enum_util.h>

namespace nnue{

constexpr size_t half_kp_dim = 768*64;
constexpr size_t base_dim = 288;
constexpr size_t move_dim = 128;

template<typename T>
struct half_kp_weights{
  big_affine<T, half_kp_dim, base_dim> w{};
  big_affine<T, half_kp_dim, base_dim> b{};

  stack_affine<T, 2*base_dim, 32> val_fc0{};
  stack_affine<T, 32, 32> val_fc1{};
  stack_affine<T, 32, 1> val_fc2{};

  stack_affine<T, 2*base_dim, 144> act_fc0{};
  stack_affine<T, 144, move_dim> act_fc1{};
  
  stack_affine<T, 2*base_dim, 1> val_skip{};
  stack_affine<T, 2*base_dim, move_dim> act_skip{};

  constexpr size_t num_parameters() const {
    return w.num_parameters() +
           b.num_parameters() +
           val_fc0.num_parameters() +
           val_fc1.num_parameters() +
           val_fc2.num_parameters() +
           act_fc0.num_parameters() + 
           act_fc1.num_parameters() +
           val_skip.num_parameters() +
           act_skip.num_parameters();
  }
  
  half_kp_weights<T>& load(weights_streamer<T>& ws){
    w.load_(ws);
    b.load_(ws);
    val_fc0.load_(ws);
    val_fc1.load_(ws);
    val_fc2.load_(ws);
    act_fc0.load_(ws);
    act_fc1.load_(ws);
    val_skip.load_(ws);
    act_skip.load_(ws);
    return *this;
  }
  
  half_kp_weights<T>& load(const std::string& path){
    auto ws = weights_streamer<T>(path);
    return load(ws);
  }
};

template<typename T>
struct feature_transformer{
  const big_affine<T, half_kp_dim, base_dim>* weights_;
  stack_vector<T, base_dim> active_;
  constexpr stack_vector<T, base_dim> active() const { return active_; }

  void clear(){ active_ = stack_vector<T, base_dim>::from(weights_ -> b); }
  void insert(const size_t idx){ weights_ -> insert_idx(idx, active_); }
  void erase(const size_t idx){ weights_ -> erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, half_kp_dim, base_dim>* src) : weights_{src} {
    clear();
  }
};

template<typename T>
struct half_kp_eval : chess::sided<half_kp_eval<T>, feature_transformer<T>>{
  const half_kp_weights<T>* weights_;
  feature_transformer<T> white;
  feature_transformer<T> black;

  constexpr stack_vector<T, 2*base_dim> get_base(const bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    return x0;
  }

  constexpr T get_value(const bool pov) const {
    const auto x0 = get_base(pov);
    const auto x1 = (weights_ -> val_fc0).forward(x0).apply_(relu<T>);
    const auto x2 = (weights_ -> val_fc1).forward(x1).apply_(relu<T>);
    const T val = (weights_ -> val_fc2).forward(x2).item() + (weights_ -> val_skip).forward(x0).item();
    return val;
  }
  
  constexpr stack_vector<T, move_dim> get_action(bool pov) const {
    const auto x0 = get_base(pov);
    const auto x1 = (weights_ -> act_fc0).forward(x0).apply_(relu<T>);
    const auto action = (weights_ -> act_fc1).forward(x1).add_((weights_ -> act_skip).forward(x0).data);
    return action;
  }

  half_kp_eval(const half_kp_weights<T>* src) : weights_{src}, white{&(src -> w)}, black{&(src -> b)} {}
};

}
