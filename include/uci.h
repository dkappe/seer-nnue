#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <chrono>

#include <board.h>
#include <move.h>
#include <search.h>
#include <option_parser.h>

namespace engine{

struct uci{
  static constexpr size_t default_thread_count = 1;
  static constexpr size_t default_tree_size = 3072;
  static constexpr size_t default_bucket_size = 64;
  static constexpr std::string_view default_weight_path = "/home/connor/Documents/GitHub/seer-nnue/train/model/save.bin";
  
  using real_t = float;

  chess::position_history history_{};
  chess::board position_{chess::board::start_pos()};
  
  size_t num_threads_{default_thread_count};
  nnue::half_kp_weights<real_t> weights_{};
  mcts::puct_tree<real_t> tree_;

  bool go_{false};
  std::chrono::milliseconds budget{0};
  std::chrono::steady_clock::time_point search_start{};
  std::ostream& os = std::cout;

  auto options(){
    auto weight_path = option_callback(string_option("Weights", std::string(default_weight_path)), [this](const std::string& path){
      weights_.load(path);
    });

    auto tree_size = option_callback(spin_option("Tree", default_tree_size, spin_range{1, 65536}), [this](const int size){
      const auto new_size = static_cast<size_t>(size);
      tree_.resize_tree(new_size);
    });

    auto thread_count = option_callback(spin_option("Threads", default_thread_count, spin_range{1, 512}), [this](const int count){
      const auto new_count = static_cast<size_t>(count);
      num_threads_ = new_count;
    });

    return uci_options(weight_path, tree_size, thread_count);
  }

  void uci_new_game(){
    history_.clear();
    position_ = chess::board::start_pos();
    tree_.set_root(history_, position_);
  }

  void set_position(const std::string& line){
    if(line == "position startpos"){ uci_new_game(); return; }
    std::regex spos_w_moves("position startpos moves((?: [a-h][1-8][a-h][1-8]+q?)+)");
    std::regex fen_w_moves("position fen (.*) moves((?: [a-h][1-8][a-h][1-8]+q?)+)");
    std::regex fen("position fen (.*)");
    std::smatch matches{};
    if(std::regex_search(line, matches, spos_w_moves)){
      auto [h_, p_] = chess::board::start_pos().after_uci_moves(matches.str(1));
      history_ = h_; position_ = p_;
    }else if(std::regex_search(line, matches, fen_w_moves)){
      position_ = chess::board::parse_fen(matches.str(1));
      auto [h_, p_] = position_.after_uci_moves(matches.str(2));
      history_ = h_; position_ = p_;
    }else if(std::regex_search(line, matches, fen)){
      history_.clear();
      position_ = chess::board::parse_fen(matches.str(1));
    }
    tree_.set_root(history_, position_);
  }

  void info_string(){
    auto cp_conversion = [](const real_t& x){ 
      return static_cast<int>(static_cast<real_t>(300.0) * std::atanh(x));
    };
    
    const int score = cp_conversion(tree_.score());
    static int last_reported_sel_depth{0};
    //const int est_depth = tree_.est_depth();
    const int sel_depth = tree_.sel_depth();
    
    const size_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - search_start).count();
    const size_t node_count = tree_.nodes();
    const size_t nps = static_cast<size_t>(1000) * node_count / (elapsed_ms+1);
    
    if(last_reported_sel_depth != sel_depth){
      last_reported_sel_depth = sel_depth;
      os << "info depth " << sel_depth << " seldepth " << sel_depth << " multipv 1 score cp " << score;
      os << " nodes " << node_count << " nps " << nps << " tbhits " << 0 << " time " << elapsed_ms << " pv " << tree_.pv_string() << '\n';
    }
  }

  void go(const std::string& line){
    go_ = true;
    std::regex go_w_time("go .*wtime ([0-9]+) .*btime ([0-9]+)");
    std::smatch matches{};
    if(std::regex_search(line, matches, go_w_time)){
      const long long our_time = std::stoll(position_.turn() ? matches.str(1) : matches.str(2));
      //budget 1/7 remaing time
      budget = std::chrono::milliseconds(our_time / 7);
      search_start = std::chrono::steady_clock::now();
    }else{
      //this is very dumb
      budget = std::chrono::milliseconds(1ull << 32ull);
      search_start = std::chrono::steady_clock::now();
    }
  }

  void stop(){
    os << "bestmove " << tree_.best_move().name(position_.turn()) << std::endl;
    go_ = false;
  }

  void ready(){
    os << "readyok\n";
  }

  void id_info(){
    os << "id name seer-puct\n";
    os << "id author C. McMonigle\n";
    os << options();
    os << "uciok\n";
  }

  void uci_loop(const std::string& line){
    std::regex position_rgx("position(.*)");
    std::regex go_rgx("go(.*)");
    if(line == "uci"){
      id_info();
    }else if(line == "isready"){
      ready();
    }else if(line == "ucinewgame"){
      uci_new_game();
    }else if(line == "stop"){
      stop();
    }else if(line == "_internal_board"){
      os << position_ << std::endl;
    }else if(std::regex_match(line, go_rgx)){
      go(line);
    }else if(std::regex_match(line, position_rgx)){
      set_position(line);
    }else if(line == "quit"){
      std::terminate();
    }else if(!go_){
      options().update(line);
    }

    if(go_){
      if((std::chrono::steady_clock::now() - search_start) >= budget){
        stop();
      }else{
        tree_.batched_update(num_threads_);
        info_string();
      }
    }
  }

  uci() : tree_(default_tree_size, default_bucket_size, &weights_) {
    weights_.load(std::string(default_weight_path));
    tree_.set_root(history_, position_);
  }
};

}
