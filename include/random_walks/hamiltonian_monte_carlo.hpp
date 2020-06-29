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

private:

  template<typename GenericPolytope>
  inline void initialize(GenericPolytope const& P,
                         Point const& p,
                         RandomNumberGenerator &rng)
  {
    // TODO Implement

  }

};

};

#endif // HAMILTONIAN_MONTE_CARLO_HPP
