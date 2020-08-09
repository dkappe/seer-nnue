#include <iostream>
#include <thread>
#include <chrono>

#include <position_history.h>
#include <board.h>
#include <nnue_half_kp.h>
#include <search.h>


int main(){
  using real_t = float;
  const auto weights = nnue::half_kp_weights<real_t>{}.load("../train/model/save.bin");

  auto tree = mcts::puct_tree<real_t>(3072, 64, &weights);
  tree.set_root(chess::position_history{}, chess::board::start_pos());

  for(size_t i(0); i < 50000; ++i){ tree.batched_update(2); }
  std::cout << tree.sel_depth() << std::endl;
  std::cout << tree.best_move() << std::endl;
  std::cout << tree.pv_string() << std::endl;
}
