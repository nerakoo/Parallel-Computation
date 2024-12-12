/*
 * --------------------------------------------------------------------------
 * BLISLAB
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order
 *      handle arbitrary  size C
 * */

#include <stdio.h>
#include <limits.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"

const char* dgemm_desc = "my blislab ";


#define LOW_2_BIT_MASK (INT_MAX - 3)


/*
 * pack one subpanel of A
 *
 * pack like this
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses).
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 *     0 0 0 0
 */
static inline
void packA_mcxkc_d(
        int    m,
        int    k,
        double *restrict XA,
        int    ldXA,
        double *packA
        )
{
  int i = 0, j = 0;
  double *a_index[DGEMM_MR];
  for (i = 0; i < (m & LOW_2_BIT_MASK); i += 4) {
    a_index[i + 0] = XA + (i + 0) * ldXA;
    a_index[i + 1] = XA + (i + 1) * ldXA;
    a_index[i + 2] = XA + (i + 2) * ldXA;
    a_index[i + 3] = XA + (i + 3) * ldXA;
  }
  for (; i < m; i++)
    a_index[i] = XA + i * ldXA;
  for (i = m; i < DGEMM_MR; i++)
    a_index[i] = NULL;
  for (j = 0; j < k; j++) {
    for (i = 0; i < (m & LOW_2_BIT_MASK); i += 4) {
      *(packA + 0) = *a_index[i + 0]++;
      *(packA + 1) = *a_index[i + 1]++;
      *(packA + 2) = *a_index[i + 2]++;
      *(packA + 3) = *a_index[i + 3]++;
      packA += 4;
    }
    for (; i < m; i++) {
      *packA++ = *a_index[i]++;
    }
    for (; i < DGEMM_MR; i++)
      *packA++ = 0.0;
    // memset(packA, 0, sizeof(double) * (DGEMM_MR - i));
    // packA += (DGEMM_MR - i);
  }
}




/*
 * --------------------------------------------------------------------------
 */

/*
 * pack one subpanel of B
 *
 * pack like this
 * if B is
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses).
 *
 * Then pack
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */
static inline
void packB_kcxnc_d(
        int    n,
        int    k,
        double *restrict XB,
        int    ldXB, // ldXB is the original k
        double *packB
        )
{
  int i, j;
  double *b_index[DGEMM_NR];
  for (j = 0; j < (n & LOW_2_BIT_MASK); j += 4) {
    b_index[j + 0] = XB + j + 0;
    b_index[j + 1] = XB + j + 1;
    b_index[j + 2] = XB + j + 2;
    b_index[j + 3] = XB + j + 3;
  }
  for (j = 0; j < n; j++)
    b_index[j] = XB + j;
  for (j = n; j < DGEMM_NR; j++)
    b_index[j] = NULL;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < (n & LOW_2_BIT_MASK); j += 4) {
      *(packB + 0) = *b_index[j + 0];
      *(packB + 1) = *b_index[j + 1];
      *(packB + 2) = *b_index[j + 2];
      *(packB + 3) = *b_index[j + 3];
      packB += 4;
      b_index[j + 0] += ldXB;
      b_index[j + 1] += ldXB;
      b_index[j + 2] += ldXB;
      b_index[j + 3] += ldXB;
    }
    for (; j < n; j++) {
        *packB++ = *b_index[j];
        b_index[j] += ldXB;
    }
    for (; j < DGEMM_NR; j++)
      *packB++ = 0.0;
    // memset(packB, 0, sizeof(double) * (DGEMM_NR - j));
    // packB += (DGEMM_NR - j);
  }
}

/*
 * --------------------------------------------------------------------------
 */

static
inline
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        const double *restrict packA,
        const double *restrict packB,
        double * C,
        int    ldc
        )
{
    int    i, j;
    aux_t  aux;
    for ( i = 0; i < m; i += DGEMM_MR ) {                      // 2-th loop around micro-kernel
      for ( j = 0; j < n; j += DGEMM_NR ) {                    // 1-th loop around micro-kernel
 ( *bl_micro_kernel ) (
         k,
         min(m-i, DGEMM_MR),
         min(n-j, DGEMM_NR),
        //  &packA[i * ldc],          // assumes sq matrix, otherwise use lda
        //  &packB[j],                //

         // what you should use after real packing routine implmemented
                  &packA[ i * k ],
                  &packB[ j * k ],
         &C[ i * ldc + j ],
         (unsigned long long) ldc,
         &aux
         );
      }                                                        // 1-th loop around micro-kernel
    }                                                          // 2-th loop around micro-kernel
}

void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *restrict XA,
        int    lda,
        double *restrict XB,
        int    ldb,
        double *C,
        int    ldc
        )
{
  int    ic, ib, jc, jb, pc, pb;
  double *packA, *packB;
  
  // Allocate packing buffers
  //
  // FIXME undef NOPACK when you implement packing
  //
// #define NOPACK
#ifndef NOPACK
  packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC/DGEMM_MR + 1 )* DGEMM_MR, sizeof(double) );
  packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC/DGEMM_NR + 1 )* DGEMM_NR, sizeof(double) );
#endif
  for ( ic = 0; ic < m; ic += DGEMM_MC ) {              // 5-th loop around micro-kernel
      ib = min( m - ic, DGEMM_MC );
      for ( pc = 0; pc < k; pc += DGEMM_KC ) {          // 4-th loop around micro-kernel
 pb = min( k - pc, DGEMM_KC );
 

#ifdef NOPACK
 packA = &XA[pc + ic * lda ];
#else
 int    i, j;
 for ( i = 0; i < ib; i += DGEMM_MR ) {
   packA_mcxkc_d(
   min( ib - i, DGEMM_MR ), /* m */
   pb,                      /* k */
   &XA[ pc + lda*(ic + i)], /* XA - start of micropanel in A */
   k,                       /* ldXA */
   &packA[ 0 * DGEMM_MC * pb + i * pb ] /* packA */);
   
 }
#endif
 for ( jc = 0; jc < n; jc += DGEMM_NC ) {        // 3-rd loop around micro-kernel
   jb = min( m - jc, DGEMM_NC );

#ifdef NOPACK
   packB = &XB[ldb * pc + jc ];
#else
   for ( j = 0; j < jb; j += DGEMM_NR ) {
     packB_kcxnc_d(
     min( jb - j, DGEMM_NR ) /* n */,
     pb                      /* k */,
     &XB[ ldb * pc + jc + j]     /* XB - starting row and column for this panel */,
     n, // should be ldXB instead /* ldXB */
     &packB[ j * pb ]        /* packB */
     );
   }
#endif

   bl_macro_kernel(
     ib,
     jb,
     pb,
     packA,
     packB,
     &C[ ic * ldc + jc ],
     ldc
     );
 }                                               // End 3.rd loop around micro-kernel
      }                                                 // End 4.th loop around micro-kernel
  }                                                     // End 5.th loop around micro-kernel
  
#ifndef NOPACK
  free( packA );
  free( packB );
#endif
}

void square_dgemm(int lda, double *A, double *B, double *C){
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C,  lda);
}
