// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file


struct LogarithmicBarrierAugmenter {

  template <typename Point, class func>
  struct LogarithmicBarrierObjective {
    typedef HPolytope<Point> Hpolytope;
    typedef typename Point::FT NT;

    func f;
    Hpolytope &P;

    LogarithmicBarrierObjective(func f_, Hpolytope P_) :
      f(f_), P(P_) {}

    NT operator() (Point &x) {
      return f(x) + P.log_barrier(x);
    }

  };

  template <typename Point, class func>
  struct LogarithmicBarrierGradient {
    typedef HPolytope<Point> Hpolytope;
    typedef typename Point::FT NT;

    func f;
    Hpolytope &P;

    LogarithmicBarrierGradient(func f_, Hpolytope P_) :
      f(f_), P(P_) {}

    Point operator() (Point &x) {
      return f(x) + P.grad_log_barrier(x);
    }

  };

};
