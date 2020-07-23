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

template <typename Sampler, typename RandomNumberGenerator, typename NT, typename Point>
void check_ergodic_mean_norm(
    Sampler &sampler,
    RandomNumberGenerator &rng,
    Point &mean,
    unsigned int dim,
    int n_samples=1500,
    NT target=NT(0),
    NT tol=1e-3) {


  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_samples; i++) {
    sampler.apply(rng, 1);
    mean = mean + sampler.x;

    #ifdef VOLESTI_DEBUG
      std::cout << sampler.x.getCoefficients().transpose() << std::endl;
    #endif
  }

  auto stop = std::chrono::high_resolution_clock::now();

  long ETA = (long) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

  mean = (1.0 / n_samples) * mean;

  NT error = abs(NT(sqrt(mean.dot(mean))) - target);

  if (target != NT(0)) error /= abs(target);

  std::cout << "Dimensionality: " << dim << std::endl;
  std::cout << "Target ergodic mean norm: " << target << std::endl;
  std::cout << "Error (relative if possible) after " << n_samples << " samples: " << error << std::endl;
  std::cout << "ETA (us): " << ETA << std::endl << std::endl;

  CHECK(error < tol);

}


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

    neg_gradient_func F(params);
    neg_logprob_func f(params);

    RandomNumberGenerator rng(1);
    HamiltonianMonteCarloWalk::parameters<NT> hmc_params;
    unsigned int dim = 50;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0(dim);

    HamiltonianMonteCarloWalk::Walk
      <Point, Hpolytope, RandomNumberGenerator, neg_gradient_func, neg_logprob_func, Solver>
      hmc(&P, x0, F, f, hmc_params);

    Point mean(dim);
    check_ergodic_mean_norm(hmc, rng, mean, dim, 15000, NT(0));
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
    unsigned int dim = 10;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0(dim);

    UnderdampedLangevinWalk::Walk
      <Point, Hpolytope, RandomNumberGenerator, neg_gradient_func, neg_logprob_func>
      uld(&P, x0, F, f, hmc_params);

    Point mean(dim);
    check_ergodic_mean_norm(uld, rng, mean, dim, 15000, NT(0));

}

template <typename NT>
void call_test_hmc() {
  std::cout << "--- Testing Hamiltonian Monte Carlo" << std::endl;
  test_hmc<NT>();
}

template <typename NT>
void call_test_uld() {
  std::cout << "--- Testing Underdamped Langevin Diffusion" << std::endl;
  test_uld<NT>();
}

TEST_CASE("hmc") {
  call_test_hmc<double>();
}

TEST_CASE("uld") {
  call_test_uld<double>();
}
