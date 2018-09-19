// VolEsti (volume computation and sampling library)

// Copyright (c) 20012-2018 Vissarion Fisikopoulos
// Copyright (c) 2018 Apostolos Chalkis

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ZONOTOPE_EXACT_VOL_H
#define ZONOTOPE_EXACT_VOL_H

#include <algorithm>
#include <iostream>
#include <string>

// From rosetta code at http://rosettacode.org/wiki/Combinations#C.2B.2B
// We made some adjustments to vectorize the output
// Compute all the N combinations from N elements
std::vector< std::vector<int> > comb(int N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    std::vector<int> iter_comb(K,0);
    std::vector<std::vector<int> > combs;
    int count;

    // print integers and permute bitmask
    do {
        count = 0;
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]){
                iter_comb[count] = i;
                count++;
            }
        }
        combs.push_back(iter_comb);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return combs;
}


template <typename NT, class Polytope>
NT exact_zonotope_vol(Polytope ZP){

    typedef typename Polytope::MT 	MT;
    typedef typename Polytope::VT 	VT;
    typedef std::vector< std::vector<int> >::iterator  IntMatIt;
    typedef std::vector<int>::iterator  IntIt;

    int n = ZP.dimension();
    int k = ZP.num_of_generators();
    MT V = ZP.get_mat().transpose();
    NT vol = 0.0;

    std::vector< std::vector<int> > combs = comb(k, n);
    IntMatIt iter_combs;
    IntIt it;

    int col;
    MT SubV(n,n);

    iter_combs = combs.begin();
    for ( ;  iter_combs!=combs.end(); ++iter_combs) {
        it = (*iter_combs).begin();
        col = 0;
        // construct all the nxn submatrices
        for ( ; it!=(*iter_combs).end(); ++it, ++col) {
            SubV.col(col) = V.col(*it);
        }
        vol += std::abs(SubV.determinant());
    }
    return vol;
}

template <typename NT>
NT vol_Ali(std::vector<NT> plane, NT zit, int dim){

    NT vol;
    int i,j,J,counter,K,k;
    //double min,max,step,z,zit;

    //std::cout<<"hello"<<std::endl;
    //min=min_coeff(plane);
    //max=max_coeff(plane);
    //step=(max-min)/100;
    //std::cout<<"helloooo"<<std::endl;
    //double *Y = (double *)malloc((dim+2) * sizeof(double));
    //double *X = (double *)malloc((dim+2) * sizeof(double));
    //double *a = (double *)malloc((dim+2) * sizeof(double));

    std::vector<NT> Y(dim+2 , 0.0), X(dim+2 , 0.0), a(dim+2 , 0.0);

    //std::cout<<min<<" "<<max<<" "<<step<<" "<<plane.size()<<std::endl;

    //z=min;
    //while (z<=max){

    J=0; K=0; counter=0;
    if (zit<0){
        X[0]=zit; J++;
    }else{
        Y[0]=zit; counter++;
    }

    for (i=0; i<dim; i++){

        a[i]=0.0;

        if (plane[i]+zit<0){
            X[J]=plane[i]+zit;
            J++;
        }else{
            Y[counter]=plane[i]+zit;
            counter++;
        }
    }
    K=dim+1-J;
    a[0]=1.0; a[dim]=0.0; a[dim+1]=0.0;

    for (i=0; i<J; i++){
        for (k=1; k<K+1; k++){
            a[k]=( Y[k-1]*a[k] - X[i]*a[k-1] ) / ( Y[k-1]-X[i] );
        }
    }

    //results.push_back(a[K]);
    //z+=step;

    vol=a[K];

    //free(Y); free(X); free(a);
    return vol;


}

#endif