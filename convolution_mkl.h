
#include <iostream>

#include "mkl.h"
#define min(x,y) (((x) < (y)) ? (x) : (y))
using namespace std;
double* matrix_mult_mkl(double* A,double* B,int m,int k,int n){
    
    int i, j;
    double alpha, beta;

    //cout<< "This example computes real matrix C=alpha*A*B+beta*C using "<<
      //      " Intel(R) MKL function dgemm, where A, B, and  C are matrices and "<<
        //    " alpha and beta are double precision scalars"<<endl;

    //cout<<" Initializing data for matrix multiplication C=A*B for matrix "<<
      //      " A(%ix%i) and matrix B(%ix%i)\n\n";
    alpha = 1.0; beta = 0.0;

    //printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
      //      " performance \n\n");
    double * C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
     // printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return C;
    }

    //printf (" Intializing matrix data \n\n");
   

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

   // printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    //printf ("\n Computations completed.\n\n");

    //printf (" Top left corner of matrix A: \n");
    // for (i=0; i<min(m,6); i++) {
    //   for (j=0; j<min(k,6); j++) {
    //    cout<< A[j+i*k];
    //   }
    // cout<<endl;
    // }

   // printf ("\n Top left corner of matrix B: \n");
    // for (i=0; i<min(k,6); i++) {
    //   for (j=0; j<min(n,6); j++) {
    //     cout<<B[j+i*n];
    //   }
    //   cout<<endl;
    //}
    
    //printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        cout<< C[j+i*n];
      }
      cout<<endl;
    }

   // printf ("\n Deallocating memory \n\n");
    return C;
}