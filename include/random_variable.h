#include <cmath>
#include <type_traits>

namespace rv{

template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr T exp_compile_time(const T& x){
  constexpr size_t series_terms = 12;

  auto pow_div_factorial = [](const T& base, const size_t& exponent){
    T result = static_cast<T>(1);
    for(size_t i(1); i <= exponent; ++i){ result *= base / static_cast<T>(i); }
    return result;
  };

  T result = static_cast<T>(0);
  for(size_t i(0); i < series_terms; ++i){
    result += pow_div_factorial(x, i);
  }
  return result;
}

template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr T approx_inv_normal_cdf(const T& p){
  constexpr T one_div_sqrt_2pi = static_cast<T>(0.398942280401);
  constexpr T half = static_cast<T>(0.5);
  constexpr T one = static_cast<T>(1.0);
  constexpr T zero = static_cast<T>(0.0);
  constexpr T dx = static_cast<T>(1e-3);

  auto pdf = [](const T& x){
    return one_div_sqrt_2pi * exp_compile_time(-half * x * x);
  };

  const T tgt = (p >= half) ? p : (one - p);

  T y = half;
  T x = zero;
  while(y < tgt){
    y += pdf(x) * dx;
    x += dx;
  }

  return (p >= half) ? x : -x;
}

template<typename T> inline constexpr T level = static_cast<T>(0.95);
template<typename T> inline constexpr T z_star = approx_inv_normal_cdf(level<T>);

template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
struct random_variable{
  T mu_, sigma_;

  constexpr const T& mu() const { return mu_; }
  constexpr const T& sigma() const { return sigma_; }
  constexpr random_variable<T> operator-() const { return random_variable(-mu_, sigma_); }


  bool gt(const random_variable<T>& other) const {
    const T s_total = std::sqrt(sigma() * sigma() + other.sigma() * other.sigma());
    return z_star<T> <= (mu() - other.mu()) / s_total;
  }

  bool lt(const random_variable<T>& other) const {
    return (-(*this)).gt(-other);
  }

  constexpr random_variable(T mu, T sigma) : mu_{mu}, sigma_{sigma} {}
};

}