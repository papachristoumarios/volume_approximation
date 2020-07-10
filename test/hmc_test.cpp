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

#include "convex_bodies/barriers.h"

template <typename Point>
Point all_ones(int dim) {
  Point p(dim);
  for (int i = 0; i < dim; i++) p.set_coord(i, 1.0);
  return p;
}

template <typename NT>
void test_hmc(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef std::function<Point(pts, NT)> func;
    typedef std::vector<func> funcs;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

    RandomNumberGenerator rng(1);


    HamiltonianMonteCarloWalk::parameters<NT> params;

    func neg_grad_f = [](pts x, NT t) {
      Point p = all_ones<Point>(x[0].dimension());
      Point z = (-2.0) * x[0];
      z = z - p;
      return z;
     };
    std::function<NT(Point)> f = [](Point x) {
      return x.dot(x) + x.sum();
    };
    unsigned int dim = 1;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0 = GetDirection<Point>::apply(dim, rng, false);
    Point z = all_ones<Point>(dim);
    z = (-0.5) * z;
    x0 = x0 - z;

    HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, RandomNumberGenerator> hmc(&P, x0, neg_grad_f, f, params);

    for (int i = 0; i < 20000; i++) {
      hmc.apply(rng);
      std::cout << hmc.x.getCoefficients().transpose() << std::endl;
    }

}

template <typename NT>
void test_underdamped_langevin(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef std::function<Point(pts, NT)> func;
    typedef std::vector<func> funcs;
    typedef HPolytope<Point> Hpolytope;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

    RandomNumberGenerator rng(1);


    UnderdampedLangevinWalk::parameters<NT> params;

    func neg_grad_f = [](pts x, NT t) {
      Point p = all_ones<Point>(x[0].dimension());
      Point z = (-2.0) * x[0];
      z = z - p;
      return z;
     };
    std::function<NT(Point)> f = [](Point x) {
      return x.dot(x) + x.sum();
    };
    unsigned int dim = 1;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0 = GetDirection<Point>::apply(dim, rng, false);
    Point v0 = GetDirection<Point>::apply(dim, rng, false);
    Point z = all_ones<Point>(dim);
    z = (-0.5) * z;
    x0 = x0 - z;

    UnderdampedLangevinWalk::Walk<Point, Hpolytope, RandomNumberGenerator>
      uld(&P, x0, v0, neg_grad_f, f, params);

    for (int i = 0; i < 1000; i++) {
      uld.apply(rng, 1, true);
      std::cout << uld.x.getCoefficients().transpose() << std::endl;
    }

}

template <typename NT>
void call_test_hmc() {
  test_hmc<NT>();
}

template <typename NT>
void call_test_langevin() {
  test_underdamped_langevin<NT>();
}

TEST_CASE("hmc") {
  call_test_hmc<double>();
}

TEST_CASE("langevin") {
  call_test_langevin<double>();
}
