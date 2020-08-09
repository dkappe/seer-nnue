#include <iostream>

#include <board.h>
#include <position_history.h>
#include <nnue_half_kp.h>
#include <evaluate.h>

int main(){
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << weights.num_parameters() << std::endl;
  nnue::half_kp_eval<float> eval(&weights);
  
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  auto bd = chess::board::parse_fen(fen);
  bd.show_init(eval);
  
  
  chess::position_history hist{};
  const auto result = chess::evaluate(hist, eval, bd);
  std::cout << bd.fen() << std::endl;
  std::cout << result << std::endl;
}
