// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Contributed and/or modified by Marios Papachristou, as part of Google Summer of Code 2020 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef LEAPFROG_H
#define LEAPFROG_H

template <typename Point, typename NT, class Polytope, class func=std::function <Point(std::vector<Point>, NT)>>
class LeapfrogODESolver {
public:
  typedef std::vector<Point> pts;
  typedef std::vector<func> funcs;
  typedef std::vector<Polytope*> bounds;
  typedef typename Polytope::VT VT;

  unsigned int dim;

  VT Ar, Av;

  NT eta;
  NT t;

  funcs Fs;
  bounds Ks;

  // Contains the sub-states
  pts xs;
  pts xs_prev;

  LeapfrogODESolver(NT initial_time, NT step, pts initial_state, funcs oracles, bounds boundaries) :
    t(initial_time), xs(initial_state), Fs(oracles), eta(step), Ks(boundaries) {
      dim = xs[0].dimension();
    };


  LeapfrogODESolver(NT initial_time, NT step, int num_states, unsigned int dimension, funcs oracles, bounds boundaries) :
    t(initial_time), Fs(oracles), eta(step), Ks(boundaries) {
      xs = pts(num_states, Point(dimension));
    };


  LeapfrogODESolver(NT initial_time, NT step, pts initial_state, funcs oracles) :
    t(initial_time), xs(initial_state), Fs(oracles), eta(step) {
      Ks = bounds(xs.size(), NULL);
      dim = xs[0].dimension();
    };


  void step() {
    xs_prev = xs;
    t += eta;
    unsigned int x_index, v_index;
    bool flag;
    for (unsigned int i = 1; i < xs.size(); i += 2) {
      flag = false;

      x_index = i - 1;
      v_index = i;

      // v' <- v + eta / 2 F(x)
      Point z = Fs[v_index](xs_prev, t);
      z = (eta / 2) * z;
      xs[v_index] = xs[v_index] + z;

      // x <- x + eta v'
      Point y = xs[v_index];
      y = (eta) * y;

      if (Ks[x_index] == NULL) {
        xs[x_index] = xs[x_index] + y;
      }
      else {
        // Find intersection (assuming a line trajectory) between x and y
        do {
          std::pair<NT, int> pbpair = Ks[x_index]->line_positive_intersect(xs[x_index], y, Ar, Av);

          if (pbpair.first >= 0 && pbpair.first <= 1) {
            xs[x_index] += (pbpair.first * 0.99) * y;
            Ks[x_index]->compute_reflection(y, xs[x_index], pbpair.second);
            xs[x_index] += y;
          }
          else {
            if (flag) break;
            xs[x_index] += y;
            flag = true;
          }
        } while (!Ks[x_index]->is_in(xs[x_index]));
      }

      // tilde v <- v + eta / 2 F(x)
      z = Fs[v_index](xs, t);
      z = (eta / 2) * z;
      xs[v_index] = xs[v_index] + z;

    }

  }

  void print_state() {
    for (int j = 0; j < xs.size(); j ++) {
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