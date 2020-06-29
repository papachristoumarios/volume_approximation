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
         Solver &solver)
    {
        initialize(P, p, eta, rng, solver);
    }

    template <typename GenericPolytope>
    Walk(GenericPolytope const& P,
         Point const& p,
         NT const& eta,
         RandomNumberGenerator &rng,
         Solver &solver,
         parameters const& params)
    {
       initialize(P, p, eta, rng, solver);
       // TODO initialize params
    }

    template <typename GenericPolytope>
    inline void apply(GenericPolytope const& P,
                      Point &p,
                      unsigned int const& walk_length,
                      RandomNumberGenerator &rng)
    {
      // TODO (re)-implement
    }

    inline void update_eta(NT eta_) {
      // TODO implement
    }

    inline NT hamiltonian(Point const& pos, Point const& vel) {
      return f(pos) + 0.5 * vel.dot(vel);
    }

private:

  template<typename GenericPolytope>
  inline void initialize(GenericPolytope const& P,
                         Point const& p,
                         RandomNumberGenerator &rng)
  {
    // TODO Implement

  }

  NT eta;
  NT L, m, kappa;
  NT delta, epsilon;
  Solver &solver;
  unsigned int dim;

  // xs[0] contains position xs[1] contains velocity
  pts xs;

  // References to xs
  Point &x, &v;
  // Proposal points
  Point x_tilde, v_tilde;

  // Contains K x R^d
  bounds Ks;

};

};

#endif // HAMILTONIAN_MONTE_CARLO_HPP
