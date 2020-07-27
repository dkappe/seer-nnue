#include <iostream>
#include <random_variable.h>

int main(){
  rv::random_variable<double> a(1.0, 0.5);
  rv::random_variable<double> b(-1.0, 2.0);
  std::cout << std::boolalpha << a.gt(b) << std::endl;
}