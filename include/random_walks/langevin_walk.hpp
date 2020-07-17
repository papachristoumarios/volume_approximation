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
  typename RandomNumberGenerator,
  typename func
>
struct LangevinStochasticFunctor {

  typedef std::vector<Point> pts;

  std::vector<NT> a, b, c;
  RandomNumberGenerator rng;
  func f;
  Point dw;

  LangevinStochasticFunctor(
    func f_,
    std::vector<NT> a_,
    std::vector<NT> b_,
    std::vector<NT> c_) :
    f(f_),
    a(a_), // Self multipliers
    b(b_), // Oracle multipliers
    c(c_) // Brownian motion multipliers
    {
      rng = RandomNumberGenerator(1);
    }

  Point operator() (unsigned int &i, pts const& xs, NT const& t) const {
    Point res(xs[i].dimension());
    dw = GetDirection<Point>::apply(xs[i].dimension(), rng, false);
    res = res + a[i] * xs[i];
    res = res + b[i] * f(i, xs, t);
    res = res + c[i] * dw;
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
    typename Polytope,
    typename RandomNumberGenerator,
    typename neg_gradient_func,
    typename neg_logprob_func
  >
  struct Walk {

    typedef std::vector<Point> pts;
    typedef typename Point::FT NT;
    typedef std::vector<Polytope*> bounds;
    typedef RandomizedMipointSDESolver<Point, NT, Polytope, neg_gradient_func, RandomNumberGenerator> Solver;

    parameters<NT> params;

    // Numerical ODE solver
    Solver *solver;

    unsigned int dim;

    // References to xs
    Point x, v;
    // Proposal points
    Point x_tilde, v_tilde;

    // Gradient Function
    neg_gradient_func F;

    // Density exponent
    neg_logprob_func f;

    NT H_tilde, H, log_prob, u_logprob;

    Walk(Polytope *P,
      Point &initial_x,
      Point &initial_v,
      neg_gradient_func neg_grad_f,
      neg_logprob_func density_exponent,
      parameters<NT> &param)
    {
      initialize(P, initial_x, initial_v, neg_grad_f, density_exponent, param);
    }

    void initialize(Polytope *P,
      Point &initial_x,
      Point &initial_v,
      neg_gradient_func neg_grad_f,
      neg_logprob_func density_exponent,
      parameters<NT> &param)
    {

      // ODE related-stuff
      params = param;
      params.kappa = params.L / params.m;
      params.eta = 1.0 /
        sqrt(20 * params.L);

      params.u = 1.0 / params.L;

      F = neg_grad_f;

      // Define exp(-f(x)) where f(x) is convex
      f = density_exponent;

      // Starting point is provided from outside
      x = initial_x;
      v = initial_v;

      dim = initial_x.dimension();

      solver = new Solver(0, params.eta, pts{initial_x, initial_v}, F, bounds{P, NULL}, params.u);

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
