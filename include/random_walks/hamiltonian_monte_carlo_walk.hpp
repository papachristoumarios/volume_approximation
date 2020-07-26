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
    NT L; // smoothness constant
    NT m; // strong-convexity constant
    NT epsilon; // tolerance in mixing
    NT eta; // step size
    NT kappa; // condition number

    parameters() :
      L(NT(1)),
      m(NT(1)),
      epsilon(NT(1e-4)),
      eta(NT(0)),
      kappa(NT(1))
    {}

    parameters(
      NT L_,
      NT m_,
      NT epsilon_,
      NT eta_) :
      L(L_),
      m(m_),
      epsilon(epsilon_),
      eta(eta_),
      kappa(L_ / m_)
    {}
  };

  template
  <
    typename Point,
    typename Polytope,
    typename RandomNumberGenerator,
    typename neg_gradient_func,
    typename neg_logprob_func,
    typename Solver
  >
  struct Walk {

    typedef std::vector<Point> pts;
    typedef typename Point::FT NT;
    typedef std::vector<Polytope*> bounds;

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

    // Gradient function
    neg_gradient_func F;

    // Helper variables
    NT H, H_tilde, log_prob, u_logprob;

    // Density exponent
    neg_logprob_func f;

    Walk(Polytope *P,
      Point &p,
      neg_gradient_func neg_grad_f,
      neg_logprob_func density_exponent,
      parameters<NT> &param)
    {
      initialize(P, p, neg_grad_f, density_exponent, param);
    }

    void initialize(Polytope *P,
      Point &p,
      neg_gradient_func neg_grad_f,
      neg_logprob_func density_exponent,
      parameters<NT> &param)
    {
      dim = p.dimension();

      // ODE related-stuff
      params = param;
      params.kappa = params.L / params.m;
      params.eta = 1.0 /
        (sqrt(20 * params.L * pow(dim, 3)));

      // Set order to 2
      F = neg_grad_f;
      F.params.order = 2;

      // Define exp(-f(x)) where f(x) is convex
      f = density_exponent;

      // Starting point is provided from outside
      x = p;

      // Initialize solver
      solver = new Solver(0, params.eta, pts{x, x}, F, bounds{P, NULL});

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
        H = hamiltonian(x, v);

        // Calculate new Hamiltonian
        H_tilde = hamiltonian(x_tilde, v_tilde);

        // Log-sum-exp trick
        log_prob = H - H_tilde < 0 ? H - H_tilde : 0;

        // Decide to switch
        u_logprob = log(rng.sample_urdist());
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
    //
    // ~Walk() {
    //   delete solver->bounds[0];
    //   delete sovlver;
    // }
  };
};

#endif // HAMILTONIAN_MONTE_CARLO_WALK_HPP
