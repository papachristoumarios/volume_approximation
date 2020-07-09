// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef HAMILTONIAN_MONTE_CARLO_WALK_HPP
#define HAMILTONIAN_MONTE_CARLO_WALK_HPP


#include "generators/boost_random_number_generator.hpp"
#include "random_walks/gaussian_helpers.hpp"
#include "ode_solvers/ode_solvers.hpp"

struct HamiltonianMonteCarloWalk {

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
    typedef std::vector<func> funcs;
    typedef std::vector<Polytope*> bounds;

    // Use Leapfrog ODE solver (other solvers can be used as well)
    typedef LeapfrogODESolver<Point, NT, Polytope> Solver;

    // Hyperparameters of the sampler
    parameters<NT> params;

    // Numerical ODE solver
    Solver *solver;

    // Dimension
    unsigned int dim;

    // References to xs
    Point x, v;

    // Proposal points
    Point x_tilde, v_tilde;

    // Function oracles Fs[0] contains grad_K = x
    // Fs[1] contains - grad f(x)
    funcs Fs;

    // Density exponent
    std::function<NT(Point)> f;

    Walk(Polytope *P,
      Point &p,
      func neg_grad_f,
      std::function<NT(Point)>
      density_exponent,
      parameters<NT> &param)
    {
      initialize(P, p, neg_grad_f, density_exponent, param);
    }

    void initialize(Polytope *P,
      Point &p,
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

      // Define Kinetic and Potential Energy gradient updates
      // Kinetic energy gradient grad_K = v
      func temp_grad_K = [](pts xs, NT t) { return xs[1]; };
      Fs.push_back(temp_grad_K);
      Fs.push_back(neg_grad_f);

      // Define exp(-f(x)) where f(x) is convex
      f = density_exponent;

      // Starting point is provided from outside
      x = p;
      dim = p.dimension();

      // Initialize solver
      solver = new Solver(0, params.eta, pts{x, x}, Fs, bounds{P, NULL});

    };


    inline void apply(
      RandomNumberGenerator &rng,
      int walk_length=1,
      bool metropolis_filter=true)
    {
      // Pick a random velocity
      v = GetDirection<Point>::apply(dim, rng, false);

      solver->set_state(0, x);
      solver->set_state(1, v);

      // Get proposals
      solver->steps(walk_length);
      x_tilde = solver->get_state(0);
      v_tilde = solver->get_state(1);


      if (metropolis_filter) {
        // Calculate initial Hamiltonian
        NT H = hamiltonian(x, v);

        // Calculate new Hamiltonian
        NT H_tilde = hamiltonian(x_tilde, v_tilde);

        // Log-sum-exp trick
        NT log_prob = H - H_tilde < 0 ? H - H_tilde : 0;

        // Decide to switch
        NT u_logprob = log(rng.sample_urdist());
        if (u_logprob < log_prob) {
          x = x_tilde;
        }
      } else {
        x = x_tilde;
      }
    }

    inline NT hamiltonian(Point &pos, Point &vel) const {
      return f(pos) + 0.5 * vel.dot(vel);
    }
  };
};

#endif // HAMILTONIAN_MONTE_CARLO_WALK_HPP
