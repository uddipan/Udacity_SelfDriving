### The Model

I have used a simple version of kinematic model ignoring complications like
bumps on the road, or friction between tires and the road.
The model equations are as introduced in the lessons:

```
x[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
y[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
psi[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
v[t] = v[t-1] + a[t-1] * dt
cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(epsi[t-1]) * dt
epsi[t] = psi[t] - psides[t-1] + v[t-1] * delta[t-1] / Lf * dt
```

Where the following are state variables:

- `x, y` : position co-ordinates of the car.
- `psi`  : direction of the car (angle with horizontal).
- `v`    : velocity of the car .
- `cte`  : cross-track error.
- `epsi` : orientation error.

The model also requires the following control variables:

- `a`    : throttle (acceleration) of the car.
- `delta`: steering angle of the car.

and another constraint 

- `Lf`   : distance of car center  of mass to front wheels cog.


The objective of the MPC is:

Given an initial state (comprising of the state variables), the constraints 
and the solver, return a set of control inputs (in the specified set of timestamps)
that minimizes the cost function.
The cost function is a combination of `cte` and `epsi`.

### Timestep Length and Elapsed Duration (N & dt)

The number of points(`N`) and the time interval(`dt`) define the prediction horizon which is the duration over which future predictions are made. 
The number of points impacts the controller performance heavily. A large N leads to instability of the car especially near sharp turns and the controller runs slower. I have used `N` = 10 and `dt` = 0.08 after some experimentation.

### Polynomial Fitting and MPC Preprocessing

A degree 3 polynomial was fitted to the waypoints provided by the simulator after transforming them to car co-ordinate system (line 109 of main.cpp).
The polynomial coefficients are fed to the MPC solver along with the initial state.

### Model Predictive Control with Latency

Actuator latency of 100 ms were taken into account while calculating the state values
to be fed to the solver (line 132 of main.cpp).

The cost function calculation also involves cost related to reference state,
influence of actuators etc, which are tuned and explained in detail as 
code comments (lines 57-78 of MPC.cpp)

## Simulation

### The vehicle must successfully drive a lap around the track.

The vehicle successfully drives a lap without leaving the track
