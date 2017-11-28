#include "PID.h"
#include <iostream>

using namespace std;

/*
 * TODO: Complete the PID class.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  Kp_ = Kp;
  Ki_ = Ki;
  Kd_ = Kd;
  
  d_error = 0.0;
  i_error = 0.0;
  p_error = 0.0;
}

void PID::UpdateError(double cte) {
  // d_error is difference between old cte and current cte
  d_error = (cte - p_error);
  // p_error is the current cte
  p_error = cte;
  // i_error is the accumulated cte's
  i_error += cte;
}

double PID::TotalError() {
  return -Kp_ * p_error - Kd_ * d_error - Ki_ * i_error;
}
