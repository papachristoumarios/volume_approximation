// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include <iostream>
#include <cmath>
#include <functional>
#include <vector>
#include <unistd.h>
#include <string>
#include <typeinfo>

#include "doctest.h"
#include "Eigen/Eigen"

#include "ode_solvers.hpp"
#include "random.hpp"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "generators/known_polytope_generators.h"


template <typename NT>
void test_hmc(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> neg_gradient_func;
    typedef IsotropicQuadraticFunctor::FunctionFunctor<Point> neg_logprob_func;
    typedef LeapfrogODESolver<Point, NT, Hpolytope, neg_gradient_func> Solver;

    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;
    params.alpha = NT(1);

    neg_gradient_func F(params);
    neg_logprob_func f(params);

    RandomNumberGenerator rng(1);
    HamiltonianMonteCarloWalk::parameters<NT> hmc_params;
    unsigned int dim = 1;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0(dim);

    HamiltonianMonteCarloWalk::Walk
      <Point, Hpolytope, RandomNumberGenerator, neg_gradient_func, neg_logprob_func, Solver>
      hmc(&P, x0, F, f, hmc_params);

    Point sum(dim);
    int n_samples = 150000;

    for (int i = 0; i < n_samples; i++) {
      hmc.apply(rng, 1);
      if (i > n_samples / 2) {
        sum = sum + hmc.x;
        std::cout << hmc.x.getCoefficients().transpose() << std::endl;
      }
    }

    sum = (1.0 / n_samples) * sum;

    // CHECK(sum.dot(sum) < 1e-5);

}


template <typename NT>
void test_uld(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> neg_gradient_func;
    typedef IsotropicQuadraticFunctor::FunctionFunctor<Point> neg_logprob_func;
    typedef LeapfrogODESolver<Point, NT, Hpolytope, neg_gradient_func> Solver;

    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;
    params.alpha = NT(1);

    neg_gradient_func F(params);
    neg_logprob_func f(params);

    RandomNumberGenerator rng(1);
    UnderdampedLangevinWalk::parameters<NT> hmc_params;
    unsigned int dim = 1;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0(dim);

    UnderdampedLangevinWalk::Walk
      <Point, Hpolytope, RandomNumberGenerator, neg_gradient_func, neg_logprob_func>
      hmc(&P, x0, F, f, hmc_params);

    Point sum(dim);
    int n_samples = 150000;

    for (int i = 0; i < n_samples; i++) {
      hmc.apply(rng, 1);
      if (i > n_samples / 2) {
        sum = sum + hmc.x;
        std::cout << hmc.x.getCoefficients().transpose() << std::endl;
      }
    }

    sum = (1.0 / n_samples) * sum;

    // CHECK(sum.dot(sum) < 1e-5);

}

template <typename NT>
void call_test_hmc() {
  test_hmc<NT>();
}

template <typename NT>
void call_test_uld() {
  test_uld<NT>();
}

TEST_CASE("hmc") {
  call_test_hmc<double>();
}

TEST_CASE("uld") {
  call_test_uld<double>();
}
