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


#ifndef COLLOCATION_HPP
#define COLLOCATION_HPP

#include "nlp_oracles/nlp_hpolyoracles.hpp"
#include "nlp_oracles/nlp_vpolyoracles.hpp"

template <
  typename Point,
  typename NT,
  class Polytope,
  class bfunc,
  class func=std::function <Point(std::vector<Point>, NT)>,
  class NontLinearOracle=MPSolveHPolyoracle<
    Polytope,
    bfunc
  >
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

  unsigned int dim;

  NT eta;
  NT t, t_prev, dt;
  const NT tol = 1e-6;

  // Function oracles x'(t) = F(x, t)
  funcs Fs;

  // Basis functions
  LagrangeBasis phi, grad_phi, integral_phi;


  bounds Ks;

  // Contains the sub-states
  pts xs, xs_prev;

  Point &x, &v;
  Point y;

  // Temporal coefficients
  coeffs cs;

  VT Ar, Av;

  MT A_phi;

  unsigned int _order;

  int prev_facet = -1;
  Point prev_point;

  IntegralCollocationODESolver(NT initial_time, NT step, pts initial_state, funcs oracles,
    bounds boundaries, unsigned int order_) :
    t(initial_time), xs(initial_state), Fs(oracles), eta(step), Ks(boundaries) :
    _order(order_) {
      dim = xs[0].dimension();
      initialize_matrices();

    };

  unsigned int order() const {
    return order_;
  }

  void initialize_matrices() {

    // Determine Chebyshev nodes
    NT temp_node;
    for (int i = 0; i < order(); i++) {
      temp_node = cos((2 * i - 1) / (2 * order()) * M_PI);
      cs.push_back(temp_node);
    }

    phi = LagrangeBasis::LagrangePolynomial<Point>(cs, LagrangeBasis::BaseType.FUNCTION);
    grad_phi = LagrangeBasis::LagrangePolynomial<Point>(cs, LagrangeBasis::BaseType.DERIVATIVE);
    integral_phi = LagrangeBasis::LagrangePolynomial<Point>(cs, LagrangeBasis::BaseType.INTEGRAL);

    A_phi.resize(order(), order());

    // Calculate integrals of Chebyshev nodes based on "Fast Polynomial Interpolation"
    for (int i = 0; i < order(); i++) {
      for (int j = 0; j < order(); j++) {
        A_phi(i, j) = integral_phi(cs[j], 0, i, order());
      }
    }


  }

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




class LagrangePolynomial : public LagrangeBasis {

  template <typename Point>
  class Basis {
    typedef template Point::FT NT;

    std::vector<Point> &coeffs;
    int order;
    BasisType basis_type;

    LagrangePolynomial(std::vector<Point> coeffs_, BasisType basis_type_) :
      coeffs(coeffs_), order((int) order.size()), basis_type(basis_type_) {}

    NT operator() (NT t, NT t0, int j, NT ord) {
      NT result;
      NT mult _den = NT(1);
      NT mult_num = NT(1);

      for (int i = 0; i < order; i++) {
        if (i != j) {
          mult_den *= (coeffs[j] - coeffs[i]);
          mult_num *= (t - coeffs[i]);
        }
      }

      switch(basis_type) {
        case FUNCTION:
          result = mult_num / mult_den;
          break;
        case DERIVATIVE:
          result = NT(0);
          for (int i = 0; i < order; i++) {
            if (i != j) result += mult_num / (t - coeffs[i]);
          }
          result *= mult_den;
          break;
        case INTEGRAL:
          result = NT(0);
          // TODO add implementation
      }

      return result;
    }
  };

  enum BasisType {
    DERIVATIVE,
    FUNCTION,
    INTEGRAL
  };

};

#endif
