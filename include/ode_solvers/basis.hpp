// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef BASIS_HPP
#define BASIS_HPP

enum BasisType {
  DERIVATIVE = 0,
  FUNCTION = 1,
  INTEGRAL = 2
};

template <typename NT, class bfunc>
class RationalFunctionBasis {
  bfunc p, q;
  bfunc grad_p, grad_q;
  NT reg = 1e-6;
  BasisType basis_type;
  NT num, den, grad_num, grad_den;

  RationalFunctionBasis(bfunc num, bfunc grad_num, bfunc den, bfunc grad_den, BasisType basis_type_) :
    p(num), grad_p(grad_num), q(den), grad_q(grad_den), basis_type(basis_type_) {};

  NT operator()(NT t, NT t0, unsigned int j, unsigned int ord) {

    switch (basis_type) {
      case FUNCTION:
        num = p(t, t0, j, ord);
        den = q(t, t0, j, ord);
        if (std::abs(den) < reg) den += reg;
        return num / den;
      case DERIVATIVE:
        num = p(t, t0, j, ord);
        grad_num = grad_p(t, t0, j, ord);
        den = q(t, t0, j, ord);
        grad_den = grad_q(t, t0, j, ord);
        if (std::abs(den * den) < reg) den += reg;
        return (grad_num  / den)  - (grad_den * num) / den;
      case INTEGRAL:
        throw true;
    }
  }

};


template <typename NT>
class LagrangeBasis {

  std::vector<NT> &coeffs;
  int order;
  BasisType basis_type;

  NT result, mult_num, mult_den;

  LagrangeBasis(std::vector<NT> coeffs_, BasisType basis_type_) :
    coeffs(coeffs_), basis_type(basis_type_) {
      order = (int) coeffs_.size();
    }

  NT operator() (NT t, NT t0, int j, NT ord) {

    mult_den = NT(1);
    mult_num = NT(1);

    for (int i = 0; i < order; i++) {
      if (i != j) {
        mult_den *= (coeffs[j] - coeffs[i]);
        mult_num *= (t - coeffs[j]);
      }
    }

    switch(basis_type) {
      case FUNCTION:
        result = mult_num / mult_den;
        break;
      case DERIVATIVE:
        result = NT(0);
        for (int i = 0; i < order; i++) {
          if (i != j) result += mult_num / (t - coeffs[j]);
        }
        result *= mult_den;
        break;
      case INTEGRAL:
        result = NT(0);
        throw true;
        // TODO add implementation
    }

    return result;
  }
};

#endif
