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
#include <chrono>

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

template <typename NT, typename Solver>
void check_norm(Solver &solver, int num_steps, NT target, NT tol=1e-4) {

  auto start = std::chrono::high_resolution_clock::now();

  #ifndef VOLESTI_DEBUG
    solver.steps(num_steps);
  #else
    for (int i = 0; i < num_steps; i++) {
      solver.step();
      solver.print_state();
    }
  #endif

  auto stop = std::chrono::high_resolution_clock::now();

  long ETA = (long) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

  NT norm = NT(0);

  for (unsigned int i = 0; i < solver.xs.size(); i++) {
    norm += solver.xs[i].dot(solver.xs[i]);
  }

  norm = sqrt(norm);
  NT error = abs(norm - target);

  std::cout << "Dimensionality: " << solver.dim << std::endl;
  std::cout << "Norm of states after " << num_steps << " steps: ";
  std::cout << norm << std::endl;
  std::cout << "Target Norm: " << target << std::endl;

  if (target != NT(0)) error /= target;

  std::cout << "Error (relative if applicable): " << error << std::endl;
  std::cout << "ETA (us): " << ETA << std::endl << std::endl;

  CHECK(error < tol);

}

template <typename NT>
void test_euler(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
		func F;

    Point q0 = Point::all_ones(100);
    q0 = 0.5 * q0;

    pts q;
    q.push_back(q0);
    EulerODESolver<Point, NT, Hpolytope, func> euler_solver =
      EulerODESolver<Point, NT, Hpolytope, func>(0, 0.01, q, F, bounds{NULL});

    check_norm(euler_solver, 1000, NT(0));

}

template <typename NT>
void test_richardson(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    func F;

    Point q0 = Point::all_ones(100);
    pts q;
    q.push_back(q0);
    RichardsonExtrapolationODESolver<Point, NT, Hpolytope, func> bs_solver =
      RichardsonExtrapolationODESolver<Point, NT, Hpolytope, func>(0, 0.1, q, F, bounds{NULL});

    check_norm(bs_solver, 1000, NT(0));

}

template <typename NT>
void test_rk4(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;

    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
		func F;

    Point q0 = Point::all_ones(100);
    pts q;
    q.push_back(q0);
    RKODESolver<Point, NT, Hpolytope, func> rk_solver =
      RKODESolver<Point, NT, Hpolytope, func>(0, 0.1, q, F, bounds{NULL});
    rk_solver.steps(1000);

    check_norm(rk_solver, 1000, NT(0));

}

template <typename NT>
void test_leapfrog_constrained(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    unsigned int dim = 4;

    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;
    params.alpha = NT(1);

		func F(params);

    // Solve in P x R for
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    bounds Ks{&P, NULL};

    Point x0 = Point(dim);
    Point v0 = Point::all_ones(dim);
    pts q{x0, v0};
    LeapfrogODESolver<Point, NT, Hpolytope, func> leapfrog_solver =
      LeapfrogODESolver<Point, NT, Hpolytope, func>(0, 0.1, q, F, Ks);

    for (int i = 0; i < 1000; i++) {
      leapfrog_solver.step();
      CHECK(leapfrog_solver.xs[0].dot(leapfrog_solver.xs[0]) < 1.1 * dim);
    }

}


template <typename NT>
void test_leapfrog(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;

    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;

    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
		func F(params);

    Point x0 = Point(1);
    Point v0 = Point(1);
    x0.set_coord(0, 0);
    v0.set_coord(0, 1.0);
    pts q{x0, v0};
    LeapfrogODESolver<Point, NT, Hpolytope, func> leapfrog_solver =
      LeapfrogODESolver<Point, NT, Hpolytope, func>(0, 0.01, q, F, bounds{NULL, NULL});

    for (int i = 0; i < 1000; i++) {
      leapfrog_solver.step();
      CHECK(leapfrog_solver.xs[0].dot(leapfrog_solver.xs[0]) < 1.1);
    }

}

template <typename NT>
void test_euler_constrained(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;

    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
		func F(params);

    unsigned int dim = 1;

    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0(dim);
    Point v0 = Point::all_ones(dim);

    EulerODESolver<Point, NT, Hpolytope, func> euler_solver =
      EulerODESolver<Point, NT, Hpolytope, func>(0, 0.01, pts{x0, v0}, F, bounds{&P, NULL});

    for (int i = 0; i < 1000; i++) {
      euler_solver.step();
      // CHECK(euler_solver.xs[0].dot(euler_solver.xs[0]) < 1.1 * dim);
    }

}

template <typename NT>
void test_richardson_constrained(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    IsotropicQuadraticFunctor::parameters<NT> params;
    unsigned int dim = 4;
    params.order = 2;
    func F(params);

    Hpolytope P = gen_cube<Hpolytope>(dim, false);

    Point x0(dim);
    Point v0 = Point::all_ones(dim);

    RichardsonExtrapolationODESolver<Point, NT, Hpolytope, func> r_solver =
      RichardsonExtrapolationODESolver<Point, NT, Hpolytope, func>
        (0, 0.01, pts{x0, v0}, F, bounds{&P, NULL});

    for (int i = 0; i < 1000; i++) {
      r_solver.step();
      CHECK(r_solver.xs[0].dot(r_solver.xs[0]) < 1.1 * dim);
    }
}

template <typename NT>
void test_rk4_constrained(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    IsotropicQuadraticFunctor::parameters<NT> params;
    params.alpha = NT(-1);
    func F(params);

    Point q0 = Point(1);
    q0.set_coord(0, 0.2);
    pts q;
    q.push_back(q0);

    Hpolytope P = gen_cube<Hpolytope>(1, false);

    bounds Ks{&P};
    RKODESolver<Point, NT, Hpolytope, func> rk_solver = RKODESolver<Point, NT, Hpolytope, func>(0, 0.01, q, F, Ks);

    check_norm(rk_solver, 1000, NT(1));
}

template <typename NT>
void test_euler_2d(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;

    bounds Ks;
    IsotropicQuadraticFunctor::parameters<NT> params;
    params.order = 2;
    func F(params);
    Point q0 = Point(2);
    q0.set_coord(0, 0.2);
    q0.set_coord(1, 0.2);
    pts q;
    q.push_back(q0);
    EulerODESolver<Point, NT, Hpolytope, func> euler_solver =
      EulerODESolver<Point, NT, Hpolytope, func>(0, 0.1, q, F, bounds{NULL});
    euler_solver.steps(1000);
    CHECK(euler_solver.xs[0].dot(euler_solver.xs[0]) < 1.1);

}

template <typename NT>
void call_test_first_order() {

  std::cout << "--- Testing solution to dx / dt = -x" << std::endl;
  test_euler<NT>();
  test_rk4<NT>();
  test_richardson<NT>();

  std::cout << "--- Testing solution to dx / dt = x in [-1, 1]" << std::endl;
  test_rk4_constrained<NT>();
  // test_richardson_constrained<NT>();

}

template <typename NT>
void call_test_second_order() {
  std::cout << "--- Testing solution to d^2x / dt^2 = -x" << std::endl;
  test_leapfrog<NT>();

  std::cout << "--- Testing solution to dx / dt = v, dv / dt = -x w/ Euler" << std::endl;
  test_euler_constrained<NT>();

  std::cout << "--- Testing solution to d^2x / dt^2 = x in [-1, 1]" << std::endl;
  test_leapfrog_constrained<NT>();

}

TEST_CASE("first_order") {
  call_test_first_order<double>();
}

TEST_CASE("second_order") {
  call_test_second_order<double>();
}

#ifndef DISABLE_NLP_ORACLES

template <typename NT>
void test_collocation(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;

    typedef std::function<NT(NT, NT, unsigned int, unsigned int)> bfunc;
    typedef std::vector<NT> coeffs;

    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
		func F;

    Point q0 = Point(1);
    q0.set_coord(0, 1.0);
    pts q;
    q.push_back(q0);

    bfunc phi = [](NT t, NT t0, unsigned int j, unsigned int order) {
      return pow(t - t0, (NT) j);
    };

    bfunc grad_phi = [](NT t, NT t0, unsigned int j, unsigned int order) {
      return ((NT) j) * pow(t - t0, (NT) (j - 1));
    };

    // Trapezoidal collocation
    coeffs cs{0.0, 0.0, 1.0};
    CollocationODESolver<Point, NT, Hpolytope, bfunc, func> c_solver =
      CollocationODESolver<Point, NT, Hpolytope, bfunc, func>
      (0, 1.0, q, F, bounds{NULL}, cs, phi, grad_phi);
    c_solver.steps(100);
    NT err=0.001;
    NT error = c_solver.xs[0].dot(c_solver.xs[0]);
    CHECK(error < err);
}

template <typename NT>
void test_integral_collocation(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;

    typedef std::vector<NT> coeffs;

    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;


    func F;
    unsigned int dim = 3;

    Point x0 = Point::all_ones(dim);
    IntegralCollocationODESolver<Point, NT, Hpolytope, func> c_solver =
      IntegralCollocationODESolver<Point, NT, Hpolytope, func>(0, 0.01, pts{x0}, F, bounds{NULL}, 8);

    check_norm(c_solver, 1000, NT(0));

}

template <typename NT>
void test_collocation_constrained(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef std::function<NT(NT, NT, unsigned int, unsigned int)> bfunc;
    typedef std::vector<NT> coeffs;
    typedef HPolytope<Point>  Hpolytope;
    typedef std::vector<Hpolytope*> bounds;
    typedef IsotropicQuadraticFunctor::GradientFunctor<Point> func;
    IsotropicQuadraticFunctor::parameters<NT> params;
    params.alpha = NT(-1);
    func F(params);

    bounds Ks;

    Hpolytope P = gen_cube<Hpolytope>(1, false);
    Ks.push_back(&P);

    bfunc phi = [](NT t, NT t0, unsigned int j, unsigned int order) {
      return pow(t - t0, (NT) j);
    };

    bfunc grad_phi = [](NT t, NT t0, unsigned int j, unsigned int order) {
      return ((NT) j) * pow(t - t0, (NT) (j - 1));
    };

    // Trapezoidal collocation
    coeffs cs{0.0, 0.0, 1.0};

    Point q0 = Point(1);
    q0.set_coord(0, 0.5);
    pts q;
    q.push_back(q0);
    CollocationODESolver<Point, NT, Hpolytope, bfunc, func> c_solver =
      CollocationODESolver<Point, NT, Hpolytope, bfunc, func>
      (0, 0.05, q, F, Ks, cs, phi, grad_phi);
    c_solver.steps(1000);
    NT err=0.1;
    NT target = 1.0;
    NT error = std::abs((c_solver.xs[0].dot(c_solver.xs[0]) - target) / target);
    CHECK(error < err);
}

template <typename NT>
void call_test_collocation() {

  std::cout << "--- Testing solution to dx / dt = -x w/ collocation" << std::endl;
  test_collocation<NT>();
  test_integral_collocation<NT>();

  // std::cout << "--- Testing solution to dx / dt = x in [-1, 1] w/ collocation" << std::endl;
  // test_collocation_constrained<NT>();

}

TEST_CASE("collocation") {
  call_test_collocation<double>();
}

#endif
