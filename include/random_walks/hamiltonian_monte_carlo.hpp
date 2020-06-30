// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef HAMILTONIAN_MONTE_CARLO_HPP
#define HAMILTONIAN_MONTE_CARLO_HPP


#include "generators/boost_random_number_generator.hpp"
#include "random_walks/gaussian_helpers.hpp"

struct HamiltonianMonteCarloWalk
{
  struct parameters {
    NT L;
    NT m;
    NT kappa;
    NT epsilon;
  };

  parameters param;

  template
  <
      typename Polytope,
      typename RandomNumberGenerator,
      typename Solver,
      typename func
  >

  struct Walk {

    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef HPolytope<Point> Hpolytope;
    typedef Zonotope<Point> zonotope;
    typedef ZonoIntersectHPoly <zonotope, Hpolytope> ZonoHPoly;
    typedef Ball<Point> BallType;
    typedef BallIntersectPolytope<Polytope,BallType> BallPolytope;
    typedef std::vector<func> funcs;
    typedef std::vector<Polytope*> bounds;


    template <typename GenericPolytope>
    Walk(GenericPolytope const& P,
         Point const& p,
         RandomNumberGenerator &rng,
         parameters const& params)
    {
       initialize(params);
    }

    template <typename GenericPolytope>
    inline void apply(GenericPolytope const& P,
                      Point &p,
                      unsigned int const& walk_length,
                      RandomNumberGenerator &rng)
    {
      // Pick a random velocity
      v = GetDirection<Point>::apply(dim, rng, false);

      // Calculate initial Hamiltonian
      NT H = hamiltonian(x, v);
      solver->set_state(0, x);
      solver->set_state(1, v);

      // Get proposals
      solver->steps(walk_length);
      x_tilde = solver->get_state(0);
      v_tilde = solver->get_state(1);

      // Calculate new Hamiltonian
      NT H_tilde = hamiltonian(x_tilde, v_tilde);

      // Log-sum-exp trick
      NT log_prob = H - H_tilde < 0 ? H - H_tilde : 0;

      // Decide to switch
      NT u_logprob = log(rng.sample_urdist());
      if (u_logprob < log_prob) {
        x = x_tilde;
      }
    }

    inline void update_eta(NT eta_) {
      eta = eta_;
    }

    inline NT hamiltonian(Point const& pos, Point const& vel) {
      return f(pos) + 0.5 * vel.dot(vel);
    }

private:

  template<typename GenericPolytope>
  inline void initialize(GenericPolytope const& P,
                         Point const& p,
                         RandomNumberGenerator &rng,
                         func neg_grad_f,
                         func neg_logprob,
                         parameters const& params)
  {
    param.L = params.L;
    param.m = params.m;
    param.kappa = params.L / params.m;
    param.epsilon = params.epsilon;
    param.delta = params.delta;

    eta = 1.0 / sqrt(2000 * L * p.dimension() * log(param.kappa / param.epsilon));

    func temp_grad_K = [](pts xs, NT t) { return xs[1]; };
    Fs.push_back(temp_grad_K);
    Fs.push_back(neg_grad_f);

    // Define exp(-f(x)) where f(x) is convex
    f = neg_logprob;

    // Create boundaries for K and U
    // Boundary for K is given in the constructor
    Ks.push_back(boundary);

    // Support of kinetic energy is R^d
    Ks.push_back(NULL);

    // Starting point is provided from outside
    x = p;

    dim = initial.dimension();

    solver = new Solver(0, eta, 2, initial.dimension(), Fs, Ks);
  }


  NT eta;
  unsigned int dim;

  // References to xs
  Point x, v;

  // Proposal points
  Point x_tilde, v_tilde;

  // Contains K x R^d
  bounds Ks;

  // Function oracles Fs[0] contains grad_K = x
  // Fs[1] contains - grad f(x)
  funcs Fs;
  std::function<NT(Point)> f; // Potential energy

  Solver *solver;

};

};

#endif // HAMILTONIAN_MONTE_CARLO_HPP
