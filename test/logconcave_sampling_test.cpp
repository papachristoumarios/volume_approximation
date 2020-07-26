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

struct CustomFunctor {

  // Custom density with neg log prob equal to || x ||^2 + 1^T x
  template <
      typename NT
  >
  struct parameters {
    unsigned int order;

    parameters() : order(2) {};

    parameters(unsigned int order_) :
      order(order)
    {}
  };

  template
  <
      typename Point
  >
  struct GradientFunctor {
    typedef typename Point::FT NT;
    typedef std::vector<Point> pts;

    parameters<NT> params;

    GradientFunctor() {};

    // The index i represents the state vector index
    Point operator() (unsigned int const& i, pts const& xs, NT const& t) const {
      if (i == params.order - 1) {
        Point y = (-1.0) * Point::all_ones(xs[0].dimension());
        y = y + (-2.0) * xs[0];
        return y;
      } else {
        return xs[i + 1]; // returns derivative
      }
    }

  };

  template
  <
    typename Point
  >
  struct FunctionFunctor {
    typedef typename Point::FT NT;

    parameters<NT> params;

    FunctionFunctor() {};

    // The index i represents the state vector index
    NT operator() (Point const& x) const {
      return x.dot(x) + x.sum();
    }

  };

};

template <typename Sampler, typename RandomNumberGenerator, typename NT, typename Point>
void check_ergodic_mean_norm(
    Sampler &sampler,
    RandomNumberGenerator &rng,
    Point &mean,
    unsigned int dim,
    int n_samples=1500,
    int skip_samples=750,
    NT target=NT(0),
    NT tol=1e-1) {

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_samples; i++) {
    sampler.apply(rng, 1);
    if (i >= skip_samples) {
      mean = mean + sampler.x;
    }

    #ifdef VOLESTI_DEBUG
      std::cout << sampler.x.getCoefficients().transpose() << std::endl;
    #endif
  }

  auto stop = std::chrono::high_resolution_clock::now();

  long ETA = (long) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

  mean = (1.0 / (n_samples - skip_samples)) * mean;

  NT error = abs(NT(mean.dot(mean)) - target);

  if (target != NT(0)) error /= abs(target);

  std::cout << "Dimensionality: " << dim << std::endl;
  std::cout << "Target ergodic mean norm: " << target << std::endl;
  std::cout << "Error (relative if possible) after " << n_samples << " samples: " << error << std::endl;
  std::cout << "ETA (us): " << ETA << std::endl << std::endl;

  CHECK(error < tol);

}

template <typename NT>
void benchmark_hmc(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;
    typedef CustomFunctor::GradientFunctor<Point> neg_gradient_func;
    typedef CustomFunctor::FunctionFunctor<Point> neg_logprob_func;
    typedef LeapfrogODESolver<Point, NT, Hpolytope, neg_gradient_func> Solver;

    neg_gradient_func F;
    neg_logprob_func f;
    RandomNumberGenerator rng(1);
    HamiltonianMonteCarloWalk::parameters<NT> hmc_params;
    unsigned int dim_min = 1;
    unsigned int dim_max = 100;
    int n_samples = 100;

    for (unsigned int dim = dim_min; dim <= dim_max; dim++) {
      Hpolytope P = gen_cube<Hpolytope>(dim, false);
      Point x0(dim);
      HamiltonianMonteCarloWalk::Walk
      <Point, Hpolytope, RandomNumberGenerator, neg_gradient_func, neg_logprob_func, Solver>
      hmc(&P, x0, F, f, hmc_params);

      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n_samples; i++) hmc.apply(rng, 1);
      auto stop = std::chrono::high_resolution_clock::now();

      long ETA = (long) std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
      std::cout << ETA << std::endl;
    }

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
    check_ergodic_mean_norm(hmc, rng, mean, dim, 50000, 25000, NT(0));
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
    check_ergodic_mean_norm(uld, rng, mean, dim, 50000, 25000, NT(0));

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

template <typename NT>
void call_test_benchmark_hmc() {
  benchmark_hmc<NT>();
}

TEST_CASE("hmc") {
  call_test_hmc<double>();
}

TEST_CASE("uld") {
  call_test_uld<double>();
}

TEST_CASE("benchmark_hmc") {
  call_test_benchmark_hmc<double>();
}
