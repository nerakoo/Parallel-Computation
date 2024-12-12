// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

// #define NAIVE

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y * blockDim.y + threadIdx.y;
    int J =  blockIdx.x * blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}
#else
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) { 
    int tx = threadIdx.x, ty = threadIdx.y;   // Built-in variable indicating the position of the current thread in the thread Block.
    int bx = blockIdx.x, by = blockIdx.y;     // Built-in variable to indicate the position of the current thread Block in the Grid.
    int bm_x = blockDim.x, bm_y = blockDim.y; // Represents the number of threads per Block.

    int row = by * bm_y * TILESCALE_M + ty;   // Calculate the position index of the current thread in the global matrix
    int col = bx * bm_x * TILESCALE_N + tx;   
    extern __shared__ _FTYPE_ sharedmem[];

    //__shared__ _FTYPE_ As[TILEDIM_M][TILEDIM_K];
    //__shared__ _FTYPE_ Bs[TILEDIM_K][TILEDIM_N];
    _FTYPE_ *As = (_FTYPE_ *)sharedmem;
    _FTYPE_ *Bs = (_FTYPE_ *)sharedmem + TILEDIM_M * TILEDIM_K;
    register _FTYPE_ C_saved[TILESCALE_M * TILESCALE_N] = {0.0f};

    int xIter = (TILEDIM_K + bm_x - 1) / bm_x; // This result indicates that in the x direction, each thread block needs to be divided into xIter small blocks for iteration.
    int yIter = (TILEDIM_K + bm_y - 1) / bm_y; 

    #pragma unroll
    for (int kk = 0; kk < N; kk += TILEDIM_K) {
        
        for (int xx = 0; xx < xIter; xx++)         
            for (int yy = 0; yy < TILESCALE_M; yy += 2) { 
                int a_row = ty + yy * bm_y;
                int a_col = tx + xx * bm_x; 
                int A_row = row + yy * bm_y;
                int A_col = kk + a_col;

                _FTYPE_ Aval = (A_col < N && A_row < N) ? A[A_row * N + A_col] : 0;
                As[a_row * TILEDIM_K + a_col] = Aval;

                a_row += bm_y;
                A_row += bm_y;

                Aval = (A_col < N && A_row < N) ? A[A_row * N + A_col] : 0;
                As[a_row * TILEDIM_K + a_col] = Aval;
            }
            
        __syncthreads();

        for (int xx = 0; xx < yIter; xx++)        
            for (int yy = 0; yy < TILESCALE_N; yy += 2) { 
                int b_row = ty + xx * bm_y;
                int b_col = tx + yy * bm_x; 
                int B_row = kk + b_row;
                int B_col = col + yy * bm_x;

                _FTYPE_ Bval = (B_col < N && B_row < N) ? B[B_row * N + B_col] : 0;
                Bs[b_row * TILEDIM_N + b_col] = Bval;

                b_col += bm_x;
                B_col += bm_x;

                Bval = (B_col < N && B_row < N) ? B[B_row * N + B_col] : 0;
                Bs[b_row * TILEDIM_N + b_col] = Bval;
            }

        __syncthreads();

        for (int k = 0; k < TILEDIM_K; k++) {
            #pragma unroll
            for (int j = 0; j < TILESCALE_N; j++) { // The value of Bs is preloaded to avoid repeated access in the inner loop
                _FTYPE_ bs_value = Bs[k * TILEDIM_N + tx + j * bm_x];
                #pragma unroll
                for (int i = 0; i < TILESCALE_M; i++) {
                    _FTYPE_ a_value = As[(ty + i * bm_y) * TILEDIM_K + k];
                    C_saved[i * TILESCALE_N + j] += a_value * bs_value;
                }
            }
        }
        
        __syncthreads();
    }

    // The result is written back to global memory
    #pragma unroll
    for (int j = 0; j < TILESCALE_N; j++) {
        #pragma unroll
        for (int i = 0; i < TILESCALE_M; i++) {
            int C_row = row + i * bm_y;
            int C_col = col + j * bm_x;
            if (C_row < N && C_col < N) {
                C[C_row * N + C_col] = C_saved[i * TILESCALE_N + j];
            }
        }
    }

}
#endif
