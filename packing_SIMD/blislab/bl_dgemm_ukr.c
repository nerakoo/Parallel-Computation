#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <omp.h>

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
                   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for (l = 0; l < k; ++l)
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
                c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
        }
    }
}



// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
void dgemm_2x4x4 (  int k,
                    int m,
                    int n,
                    const double *restrict A,
                    const double *restrict B,
                    double *C,
                    unsigned long long ldc,
                    aux_t* data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    register float64_t aval;
    int lda = DGEMM_MR, ldb = DGEMM_NR, kk = 0;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C + 0 * ldc);
    c1x = svld1_f64(npred, C + 1 * ldc);
    c2x = svld1_f64(npred, C + 2 * ldc);
    c3x = svld1_f64(npred, C + 3 * ldc);
    for (; kk < (k >> 2) << 2; kk += 4){
        aval = *(A + kk * lda + 0);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + kk * ldb);
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + kk * lda + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + kk * lda + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + kk * lda + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);

        aval = *(A + (kk + 1) * lda + 0);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + (kk + 1) * ldb);
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + (kk + 1) * lda + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + (kk + 1) * lda + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + (kk + 1) * lda + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);

        aval = *(A + (kk + 2) * lda + 0);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + (kk + 2) * ldb);
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + (kk + 2) * lda + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + (kk + 2) * lda + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + (kk + 2) * lda + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);

        aval = *(A + (kk + 3) * lda + 0);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + (kk + 3) * ldb);
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + (kk + 3) * lda + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + (kk + 3) * lda + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + (kk + 3) * lda + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);
    }

    for (; kk < k; kk++){
        aval = *(A + kk * lda + 0);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + kk * ldb);
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + kk * lda + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + kk * lda + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + kk * lda + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);
    }

    svst1_f64(npred, C + 0 * ldc, c0x);
    svst1_f64(npred, C + 1 * ldc, c1x);
    svst1_f64(npred, C + 2 * ldc, c2x);
    svst1_f64(npred, C + 3 * ldc, c3x);
}
