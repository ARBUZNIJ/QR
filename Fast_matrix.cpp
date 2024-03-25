#include "QR_header.h"
#include <iostream>
#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include <time.h>
#include <chrono>
#include <iomanip> 
#include <thread>

void QR::mult_Q_by_R(size_t i_start)
{
    //Q is (n-i_start)*(n-i_start) matrix starting from [0][0] (F)

    size_t Q_row = n - i_start;
    size_t Q_col = n - i_start;

    //R_tmp is (n-i_start)*(n-(i_start+block_size)) matrix starting from [i_start][i_start+block_size] (S)

    size_t R_tmp_row = n - i_start;
    size_t R_tmp_col = n - (i_start + block_size);

    const size_t block_size_row = 96, block_size_col = 192;

    const size_t sub_block_size = 6, sub_block_size2 = 48, sub_block_size3 = 8; //sub_sub_block_size3 == 2 * sizeof(__m512); (byte)
    // +- same time for sub_block_size2 = 48, 96

    size_t t = Q_row - (Q_row % sub_block_size);// i
    size_t l = R_tmp_col - (R_tmp_col % sub_block_size3);// j
    size_t s = Q_col - (Q_col % sub_block_size2);// k

#pragma omp parallel for
    for (size_t i1 = 0; i1 < Q_row; i1 += block_size_row)
        for (size_t k1 = 0; k1 < Q_col; k1 += block_size_col)
            for (size_t j1 = 0; j1 < R_tmp_col; j1 += block_size_row)

                for (size_t i3 = i1; i3 < i1 + block_size_row && i3 < t; i3 += sub_block_size)
                    for (size_t k3 = k1; k3 < k1 + block_size_col && k3 < s; k3 += sub_block_size2)
                        for (size_t j3 = j1; j3 < j1 + block_size_row && j3 < l; j3 += sub_block_size3)
                        {
                            //theoretically, with this implementation, compiler should generate code that makes better use of the superscalar architecture


                            /*
                            __m256 c[(sub_block_size << 1)];
                            __m256 a1, a2, b1, b2, b3, b4;

                            for (int i4 = 0; i4 < sub_block_size; i4++)
                                for (int j4 = 0; j4 < 2; j4++)
                                    c[(i4<<1) + j4] = _mm256_loadu_pd(&RES[(i4 + i3) * RES.col + j3 + j4 * (sub_block_size3 >> 1)]);

                            for (int k5 = 0; k5 < sub_block_size2; k5+=2)
                            {
                                b1 = _mm256_loadu_pd(&S[(k3 + k5) * S.col + j3]);
                                b2 = _mm256_loadu_pd(&S[(k3 + k5) * S.col + j3 + (sub_block_size3 >> 1)]);
                                b3 = _mm256_loadu_pd(&S[(k3 + k5 + 1) * S.col + j3]);
                                b4 = _mm256_loadu_pd(&S[(k3 + k5 + 1) * S.col + j3 + (sub_block_size3 >> 1)]);

                                for (int i5 = 0; i5 < sub_block_size; i5++)
                                {
                                    int i = i5 << 1;
                                    a1 = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);
                                    c[i] = _mm256_fmadd_pd(a1, b1, c[i]);
                                    c[i + 1] = _mm256_fmadd_pd(a1, b2, c[i + 1]);

                                    a2 = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5 + 1)]);
                                    c[i] = _mm256_fmadd_pd(a2, b3, c[i]);
                                    c[i + 1] = _mm256_fmadd_pd(a2, b4, c[i + 1]);
                                }
                            }

                            for (int i6 = 0; i6 < sub_block_size; i6++)
                                for (int j6 = 0; j6 < 2; j6++)
                                    _mm256_storeu_pd(&RES[(i6 + i3) * RES.col + j3 + j6 * (sub_block_size3 >> 1)], c[(i6<<1) + j6]);
                           */



                            __m256d c[(sub_block_size << 1)];
                            __m256d a, b1, b2;

                            for (size_t i4 = 0; i4 < sub_block_size; i4++)
                                for (size_t j4 = 0; j4 < 2; j4++)
                                    c[(i4 << 1) + j4] = _mm256_loadu_pd(&r((i4 + i3),j3 + j4 * (sub_block_size3 >> 1)));

                            for (size_t k5 = 0; k5 < sub_block_size2; k5++)
                            {
                                b1 = _mm256_loadu_pd(&r_tmp((k3 + k5),j3));
                                b2 = _mm256_loadu_pd(&r_tmp((k3 + k5),j3 + (sub_block_size3 >> 1)));
#pragma unroll(sub_block_size)
                                for (size_t i5 = 0; i5 < sub_block_size; i5++)
                                {
                                    size_t i = i5 << 1;

                                    a = _mm256_set1_pd(q((i3 + i5),(k3 + k5)));

                                    c[i] = _mm256_fmadd_pd(a, b1, c[i]);
                                    c[i + 1] = _mm256_fmadd_pd(a, b2, c[i + 1]);
                                }
                            }

                            for (size_t i6 = 0; i6 < sub_block_size; i6++)
                                for (size_t j6 = 0; j6 < 2; j6++)
                                    _mm256_storeu_pd(&r((i6 + i3),j3 + j6 * (sub_block_size3 >> 1)), c[(i6 << 1) + j6]);

                        }

    if (R_tmp_col == l)
    {
        if ((Q_row != t) && (Q_col != s))
        {
            //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = s; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
            //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
            for (size_t i = t; i < Q_row; i++)
                for (size_t k = 0; k < s; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
        else if ((Q_row != t) && (Q_col == s))
        {
            //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
            for (size_t i = t; i < Q_row; i++)
                for (size_t k = 0; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
        else if ((Q_row == t) && (Q_col != s))
        {
            //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = s; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
    }
    else if (R_tmp_col != l)
    {
        if ((Q_row != t) && (Q_col != s))
        {
            //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = s; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);

            //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
            for (size_t i = t; i < Q_row; i++)
                for (size_t k = 0; k < s; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);

            //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
            for (size_t i = 0; i < t; i++)
                for (size_t k = 0; k < s; k++)
#pragma omp simd
                    for (size_t j = l; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
        else if ((Q_row != t) && (Q_col == s))
        {
            //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
            for (size_t i = t; i < Q_row; i++)
                for (size_t k = 0; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);

            //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
            for (size_t i = 0; i < t; i++)
                for (size_t k = 0; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = l; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
        else if ((Q_row == t) && (Q_col != s))
        {
            //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = s; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = 0; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);

            //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = 0; k < s; k++)
#pragma omp simd
                    for (size_t j = l; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
        else if ((Q_row == t) && (Q_col == s))
        {
            //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
            for (size_t i = 0; i < Q_row; i++)
                for (size_t k = 0; k < Q_col; k++)
#pragma omp simd
                    for (size_t j = l; j < R_tmp_col; j++)
                        r(i, j) += q(i, k) * r_tmp(k, j);
        }
    }
}