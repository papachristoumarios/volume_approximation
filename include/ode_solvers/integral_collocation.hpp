// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

// Refers to the integral collocation method with Lagrange Polynomials
// from Lee, Yin Tat, Zhao Song, and Santosh S. Vempala.
//"Algorithmic theory of ODEs and sampling from well-conditioned
// logconcave densities." arXiv preprint arXiv:1812.06243 (2018).


#ifndef INTEGRAL_COLLOCATION_HPP
#define INTEGRAL_COLLOCATION_HPP

#include "nlp_oracles/nlp_hpolyoracles.hpp"
#include "nlp_oracles/nlp_vpolyoracles.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/math/special_functions/chebyshev.hpp"
#include "boost/math/special_functions/chebyshev_transform.hpp"

template <
  typename Point,
  typename NT,
  class Polytope,
  class func=std::function <Point(std::vector<Point>&, NT&)>
>
class IntegralCollocationODESolver {
public:

  // Vectors of points
  typedef std::vector<Point> pts;
  typedef std::vector<pts> ptsv;

  // typedef from existing templates
  typedef typename Polytope::MT MT;
  typedef typename Polytope::VT VT;
  typedef std::vector<MT> MTs;
  typedef std::vector<func> funcs;
  typedef std::vector<Polytope*> bounds;
  typedef std::vector<NT> coeffs;
  typedef boost::numeric::ublas::vector<NT> boost_vector;
  typedef boost::math::chebyshev_transform<NT> chebysev_transform_boost;

  unsigned int dim;

  NT eta;
  NT t, t_prev, dt, temp_node, a, b;
  const NT tol = 1e-6;

  // Function oracles x'(t) = F(x, t)
  funcs Fs;
  bounds Ks;

  // Contains the sub-states
  pts xs, X_temp;

  // Temporal coefficients
  coeffs cs;

  VT Ar, Av, X_op, nodes;

  MT A_phi, X0, X, X_prev, F_op;

  unsigned int _order;

  int prev_facet = -1;
  Point prev_point;

  IntegralCollocationODESolver(NT initial_time, NT step, pts initial_state,
    funcs oracles, bounds boundaries, unsigned int order_) :
    t(initial_time), xs(initial_state), Fs(oracles), eta(step), Ks(boundaries),
    _order(order_) {
      dim = xs[0].dimension();
      initialize_matrices();
    };

  unsigned int order() const {
    return _order;
  }

  void initialize_matrices() {

    A_phi.resize(order(), order());
    nodes.resize(order());

    boost_vector b_vec(order());

    for (unsigned int i = 0; i < order(); i++) b_vec(i) = NT(0);

    // Calculate integrals of basis functions based on the Discrete Chebyshev Transform
    for (unsigned int i = 0; i < order(); i++) {
      b_vec(i) = NT(1);
      if (i > 0) {
        b_vec(i-1) = NT(0);
      }
      for (unsigned int j = 0; j < order(); j++) {
        nodes(j) = NT(cos((2 * (1 + j) - 1) * M_PI / (2 * order())));
        if (nodes(j) < NT(-1)) {
          a = nodes(j);
          b = NT(-1);
        } else {
          a = NT(-1);
          b = nodes(j);
        }

        chebysev_transform_boost transform(b_vec, a, b, 1e-6, 5);
        std::cout << "result" <<  NT(transform.integrate()) << std::endl;
      }
    }

    // std::cout << A_phi << std::endl;

    X.resize(xs.size() * dim, order());
    X0.resize(xs.size() * dim, order());
    X_prev.resize(xs.size() * dim, order());
    X_op.resize(xs.size() * dim);
  }

  void initialize_fixed_point() {
    for (unsigned int ord = 0; ord < order(); ord++) {
      for (unsigned int i = 0; i < xs.size(); i++) {

          X0.col(ord).seqN(i * dim, (i + 1) * dim) = xs[i].getCoefficients();
      }
    }
  }
  //
  // void step() {
  //   initialize_fixed_point();
  //
  //   X = X0;
  //
  //   // TODO change with paper iters T / eps * max (F)
  //   for (int iter = 0; iter < 10; iter++) {
  //     for (unsigned int ord = 0; ord < order(); ord++) {
  //       for (unsigned int i = 0; i < xs.size(); i++) {
  //         X_temp[i] = Point(X.col(ord));
  //       }
  //       for (unsigned int i = 0; i < xs.size(); i++) {
  //         temp_node = nodes(ord) * eta;
  //         F_op.col(ord).seqN(i * dim, (i + 1) * dim) =
  //           Fs[i](X_temp, temp_node).getCoefficients();
  //       }
  //     }
  //
  //     X = X0 + F_op * A_phi;
  //
  //   }
  //
  //   X_op = X0;
  //
  //   for (unsigned int i = 0; i < xs.size(); i++) {
  //     xs[i] = Point(X_op.seqN(i * dim, (i + 1) * dim));
  //   }
  //
  // }

  void step() {

  }

  void print_state() {
    for (int j = 0; j < xs.size(); j++) {
      for (unsigned int i = 0; i < xs[j].dimension(); i++) {
        std::cout << xs[j][i] << " ";
      }
    }
    std::cout << std::endl;
  }

  void steps(int num_steps) {
    for (int i = 0; i < num_steps; i++) step();
  }

  Point get_state(int index) {
    return xs[index];
  }

  void set_state(int index, Point p) {
    xs[index] = p;
  }
};


#endif
