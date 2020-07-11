#include <iostream>

#include <nnue_half_kp.h>
#include <board.h>
#include <move_picker.h>

int main(){
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << weights.num_parameters() << std::endl;
  nnue::half_kp_eval<float> eval(&weights);
  
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  auto bd = chess::board::parse_fen(fen);
  bd.show_init(eval);
  
  for(;;){
    const float value = eval.get_value(bd.turn());
    const auto action = eval.get_action(bd.turn());
    std::cout << bd.fen() << '\n';
    std::cout << " -> value: " << value << '\n';
    std::cout << " -> action: " << action << '\n';
    const chess::move_list mv_ls = bd.generate_moves();
    
    auto picker =  chess::move_picker<float>(action, mv_ls);
    std::cout << picker << '\n';
    
    size_t i; std::cin >> i;
    for(size_t idx(0); idx < i; ++idx){ picker.step(); }
    
    const auto mv = std::get<chess::move>(picker.val());
    eval = bd.half_kp_updated(mv, eval);
    bd = bd.forward(mv);
  }
}
