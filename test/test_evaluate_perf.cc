#include <iostream>
#include <chrono>

#include <board.h>
#include <position_history.h>
#include <nnue_half_kp.h>
#include <evaluate.h>

void time_(){
  constexpr size_t num_runs = 10000;
  
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << weights.num_parameters() << std::endl;
  nnue::half_kp_eval<float> eval(&weights);
  chess::position_history hist{};
  
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  auto bd = chess::board::parse_fen(fen);
  bd.show_init(eval);
  auto start = std::chrono::high_resolution_clock::now();
  
  float sum{};
  for(size_t i(0); i < num_runs; ++i){
    sum += chess::evaluate(hist, eval, bd).value_;
  }
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =  std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  //prevents compiler optimization
  std::cout << "sum: " << sum << '\n';
  const double evals_per_second = static_cast<double>(num_runs) * 1e6 / static_cast<double>(duration.count());
  std::cout << "evals_per_second: " << evals_per_second << '\n';
}

int main(){
  time_();
}
