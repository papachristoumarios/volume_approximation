// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef LANGEVIN_WALK_HPP
#define LANGEVIN_WALK_HPP

#include "generators/boost_random_number_generator.hpp"
#include "random_walks/gaussian_helpers.hpp"

// Implements the stochastic function of the Underdamped Langevin
template
<
  typename NT,
  typename Point,
  class RandomNumberGenerator,
  class func
>
struct LangevinStochasticFunction {

  typedef std::vector<Point> pts;

  NT a, b, c;
  int index;
  RandomNumberGenerator rng;
  func f;

  Point dw;

  LangevinStochasticFunction(
    func f_,
    NT a_ = NT(0),
    NT b_ = NT(1),
    NT c_ = NT(0),
    int index_ = 0) :
    f(f_),
    a(a_), // Self multiplier
    b(b_), // Oracle multiplier
    c(c_), // Brownian motion multiplier
    index(index_) {
      rng = RandomNumberGenerator(1);
    }

  Point operator() (pts xs, NT t) {
    Point res(xs[index].dimension());
    dw = GetDirection<Point>::apply(xs[index].dimension(), rng, false);
    res = res + a * xs[index];
    res = res + b * f(xs, t);
    res = res + c * dw;

    return res;

  }

};

struct UnderdampedLangevinWalk {

  template
  <
    typename NT
  >
  struct parameters {
    NT L = NT(1); // smoothness constant
    NT m = NT(1); // strong-convexity constant
    NT epsilon = NT(1e-4); // tolerance in mixing
    NT eta = NT(0); // step size
    NT kappa = NT(1); // condition number
    NT u = NT(1);
  };

  template
  <
    typename Point,
    class Polytope,
    class RandomNumberGenerator
  >
  struct Walk {

    typedef std::vector<Point> pts;
    typedef typename Point::FT NT;
    typedef std::function <Point(pts, NT)> func;
    typedef LangevinStochasticFunction<NT, Point, RandomNumberGenerator, func> sfunc;
    typedef std::vector<sfunc> sfuncs;
    typedef std::vector<Polytope*> bounds;
    typedef LeapfrogODESolver<Point, NT, Polytope, sfunc> Solver;

    parameters<NT> params;

    // Numerical ODE solver
    Solver *solver;

    unsigned int dim;

    // References to xs
    Point x, v;
    // Proposal points
    Point x_tilde, v_tilde;

    // Function oracles Fs[0] contains grad_K = x
    // Fs[1] contains - grad f(x)
    sfuncs Fs;
    std::function<NT(Point)> f;

    NT H_tilde, H, log_prob, u_logprob;

    Walk(Polytope *P,
      Point &initial_x,
      Point &initial_v,
      func neg_grad_f,
      std::function<NT(Point)>
      density_exponent,
      parameters<NT> &param)
    {
      initialize(P, initial_x, initial_v, neg_grad_f, density_exponent, param);
    }

    void initialize(Polytope *P,
      Point &initial_x,
      Point &initial_v,
      func neg_grad_f,
      std::function<NT(Point)>
      density_exponent,
      parameters<NT> &param)
    {

      // ODE related-stuff
      params = param;
      params.kappa = params.L / params.m;
      params.eta = 1.0 /
        sqrt(20 * params.L);

      params.u = 1.0 / params.L;

      // Define Kinetic and Potential Energy gradient updates
      // Kinetic energy gradient grad_K = v
      func temp_grad_K = [](pts xs, NT t) { return xs[1]; };

      sfunc stoch_temp_grad_K(temp_grad_K, NT(0), NT(1), NT(0), 0);
      sfunc stoch_neg_grad_f(neg_grad_f, NT(-2), NT(params.u), NT(2 * sqrt(params.u)), 1);


      Fs.push_back(stoch_temp_grad_K);
      Fs.push_back(stoch_neg_grad_f);

      // Define exp(-f(x)) where f(x) is convex
      f = density_exponent;


      // Starting point is provided from outside
      x = initial_x;
      v = initial_v;

      dim = initial_x.dimension();

      solver = new Solver(0, params.eta, pts{initial_x, initial_v}, Fs, bounds{P, NULL});

    };

    inline void apply(
      RandomNumberGenerator &rng,
      int walk_length=1,
      bool metropolis_filter=false)
    {
      solver->set_state(0, x);
      solver->set_state(1, v);

      // Get proposals
      solver->steps(walk_length);
      x_tilde = solver->get_state(0);
      v_tilde = solver->get_state(1);

      if (metropolis_filter) {
        // Calculate initial Hamiltonian
        H = hamiltonian(x, v);

        // Calculate new Hamiltonian
        H_tilde = hamiltonian(x_tilde, v_tilde);

        // Log-sum-exp trick
        log_prob = H - H_tilde < 0 ? H - H_tilde : 0;

        // Decide to switch
        u_logprob = log(rng.sample_urdist());
        if (u_logprob < log_prob) {
          x = x_tilde;
          v = v_tilde;
        }
      } else {
        x = x_tilde;
        v = v_tilde;
      }

    }

    inline NT hamiltonian(Point &pos, Point &vel) const {
      return f(pos) + 1.0 / (2 * params.u) * vel.dot(vel);
    }

  };
};

#endif // LANGEVIN_WALK_HPP
