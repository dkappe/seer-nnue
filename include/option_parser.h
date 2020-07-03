#pragma once

#include <iostream>
#include <type_traits>
#include <string>
#include <string_view>
#include <optional>
#include <regex>
#include <tuple>
#include <utility>
#include <functional>

namespace engine{

struct string_option{
  using type = std::string;
  std::string name_;
  std::optional<std::string> default_ = {};

  std::optional<std::string> maybe_read(const std::string& cmd) const {
    std::regex regex("setoption name " + name_ + " value (.*)");
    if(auto matches = std::smatch{}; std::regex_search(cmd, matches, regex)){
      std::string match = matches.str(1);
      if(match == ""){
        return default_;
      }else{
        return match;
      }
    }
    return {};
  }

  string_option(const std::string_view& name) : name_{name} {}
  string_option(const std::string_view& name, const std::string& def) : name_{name}, default_{def} {}
};

struct spin_range{
  int min, max;

  int clamp(int x) const {
    return std::min(std::max(min, x), max);
  }

  spin_range(const int a, const int b) : min{a}, max{b} {}
};

struct spin_option{
  using type = int;
  std::string name_;
  std::optional<int> default_ = {};
  std::optional<spin_range> range_ = {};

  std::optional<int> maybe_read(const std::string& cmd) const {
    std::regex regex("setoption name " + name_ + " value (-?[0-9]+)");
    if(auto matches = std::smatch{}; std::regex_search(cmd, matches, regex)){
      const int raw = std::stoi(matches.str(1));
      if(range_.has_value()){
        return range_.value().clamp(raw);
      }else{
        return raw;
      }
    }
    return {};
  }

  spin_option(const std::string_view& name) : name_{name} {}
  spin_option(const std::string& name, const spin_range& range) : name_{name}, range_{range} {}
  spin_option(const std::string& name, const int def) : name_{name}, default_{def} {}
  spin_option(const std::string& name, const int def, const spin_range& range) : name_{name}, default_{def}, range_{range} {}
};

struct button_option{
  using type = bool;
  std::string name_;

  std::optional<bool> maybe_read(const std::string& cmd) const {
    if(cmd == (std::string("setoption name ") + name_)){
      return true;
    }else{
      return {};
    }
  }

  button_option(const std::string_view& name) : name_{name} {}
};


std::ostream& operator<<(std::ostream& ostr, const string_option& opt){
  ostr << "option name " << opt.name_ << " type string";
  if(opt.default_.has_value()){
    ostr << " default " << opt.default_.value(); 
  }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const spin_option& opt){
  ostr << "option name " << opt.name_ << " type spin";
  if(opt.default_.has_value()){
    ostr << " default " << opt.default_.value(); 
  }
  if(opt.range_.has_value()){
    ostr << " min " << opt.range_.value().min << " max " << opt.range_.value().max;
  }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const button_option& opt){
  ostr << "option name " << opt.name_ << " type button";
  return ostr;
}

template<typename T>
inline constexpr bool is_option_v = std::is_same_v<T, spin_option> || std::is_same_v<T, string_option> || std::is_same_v<T, button_option>;

template<typename T>
struct option_callback{
  static_assert(is_option_v<T>, "T must be of option type");
  
  T option_;
  std::function<void(typename T::type)> callback_;

  void maybe_call(const std::string& cmd){
    std::optional<typename T::type> read = option_.maybe_read(cmd);
    if(read.has_value()){ callback_(read.value()); }
  }

  template<typename F>
  option_callback(const T& option, F&& f) : option_{option}, callback_{f} {}
};

template<typename ... Ts>
struct uci_options{
  std::tuple<option_callback<Ts>...> options_;

  template<typename F, size_t ... I>
  void apply_impl_(F&& f, std::index_sequence<I...>){
    auto helper = [](auto...){};
    auto map = [&f](auto&& x){ f(x); return 0; };
    helper(map(std::get<I>(options_))...);
  }

  template<typename F>
  void apply(F&& f){
    apply_impl_(std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{});
  }

  void update(const std::string& cmd){
    apply([cmd](auto& opt){ opt.maybe_call(cmd); });
  }

  uci_options(const option_callback<Ts>& ... options) : options_{options...} {}
};

template<typename ... Ts>
std::ostream& operator<<(std::ostream& ostr, uci_options<Ts...> options){
  options.apply([&ostr](const auto& opt){ ostr << opt.option_ << '\n'; });
  return ostr;
}

}