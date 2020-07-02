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

// template <typename NT>
// void test_hmc_isotropic_gaussian(){
//     typedef Cartesian<NT>    Kernel;
//     typedef typename Kernel::Point    Point;
//     typedef std::vector<Point> pts;
//     typedef std::function<Point(pts, NT)> func;
//     typedef std::vector<func> funcs;
//     typedef boost::mt19937    RNGType;
//     typedef HPolytope<Point> Hpolytope;
//     typedef EulerODESolver<Point, NT, Hpolytope> Solver;
//     typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;
//
//     RandomNumberGenerator rng(1);
//
//     // Isotropic gaussian
//     NT L = 3;
//     NT m = L;
//     func neg_grad_f = [](pts x, NT t) { return (-1.0 / (2 * 3 * 3)) * x[0]; };
//     std::function<NT(Point)> f = [](Point x) { return (0.5 / (2 * 3 * 3)) * x.dot(x); };
//     unsigned int dim = 2;
//     Point x0 = GetDirection<Point>::apply(dim, rng, false);
//     x0 = (1.0 / sqrt(3)) * x0;
//
//     HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, Solver, RandomNumberGenerator> hmc(neg_grad_f, f, x0, m, L, 0.1, 0.01, NULL);
//
//     hmc.mix();
//
//     for (int i = 0; i < 1000; i++) {
//       Point p = hmc.apply();
//       for (int j = 0; j < p.dimension(); j++) std::cout << p[j] << " ";
//       std::cout << std::endl;
//     }
//
// }

template <typename NT>
void test_hmc_isotropic_gaussian_leapfrog(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef std::vector<Point> pts;
    typedef std::function<Point(pts, NT)> func;
    typedef std::vector<func> funcs;
    typedef HPolytope<Point> Hpolytope;
    typedef LeapfrogODESolver<Point, NT, Hpolytope> Solver;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

    RandomNumberGenerator rng(1);


    HamiltonianMonteCarloWalk::parameters<NT> params;

    func neg_grad_f = [](pts x, NT t) { return (-1.0) * x[0]; };
    std::function<NT(Point)> f = [](Point x) { return 0.5 * x.dot(x); };
    unsigned int dim = 1;
    Hpolytope P = gen_cube<Hpolytope>(dim, false);
    Point x0 = GetDirection<Point>::apply(dim, rng, false);

    HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, Solver, RandomNumberGenerator> hmc(neg_grad_f, f, x0, params);

    for (int i = 0; i < 20000; i++) {
      hmc.apply(rng);
      if (i > 2000) std::cout << hmc.x.getCoefficients().transpose() << std::endl;
    }

}
//
// template <typename NT>
// void test_hmc_isotropic_truncated_gaussian(){
//     typedef Cartesian<NT>    Kernel;
//     typedef typename Kernel::Point    Point;
//     typedef std::vector<Point> pts;
//     typedef std::function<Point(pts, NT)> func;
//     typedef std::vector<func> funcs;
//     typedef boost::mt19937    RNGType;
//     typedef HPolytope<Point> Hpolytope;
//     typedef LeapfrogODESolver<Point, NT, Hpolytope> Solver;
//     // Isotropic gaussian
//     NT L = 1;
//     NT m = L;
//     func neg_grad_f = [](pts x, NT t) { return (-1.0) * x[0]; };
//     std::function<NT(Point)> f = [](Point x) { return (0.5) * x.dot(x); };
//     unsigned int dim = 2;
//     Hpolytope P = gen_cube<Hpolytope>(dim, false);
//     Point x0(dim);
//
//     HamiltonianMonteCarloWalk::Walk<Point, NT, RNGType, Hpolytope, Solver> hmc(neg_grad_f, f, x0, L, m, 0.01, 0.001, &P);
//
//     hmc.applys(1000);
//
//     for (int i = 0; i < 1000; i++) {
//       Point p = hmc.apply();
//       for (int j = 0; j < p.dimension(); j++) std::cout << p[j] << " ";
//       std::cout << std::endl;
//     }
//
// }

template <typename NT>
void call_test_hmc() {
  // test_hmc_isotropic_gaussian<double>();
  // test_hmc_isotropic_truncated_gaussian<double>();
  test_hmc_isotropic_gaussian_leapfrog<double>();
}

TEST_CASE("hmc") {
  call_test_hmc<double>();
}
