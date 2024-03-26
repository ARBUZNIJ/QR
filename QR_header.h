#pragma once
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <mkl.h>

//const size_t bs = 32;
using namespace std;
typedef double T;
class QR
{
private:

	size_t n, block_size, i_start;
	T* Q, * R, * REF_A, * u, * factor_block, * w, * R_tmp, * v_vector, * v;
	T eps = 1e-5;

public:

	QR(bool flag, size_t size, size_t b_s) : n(size), block_size(b_s) //flag = 0 - ââîä ðàíäîìíûõ ÷èñåë, èíà÷å ââîä ñ êëàâèàòóðû
	{
		REF_A = new T[n * n];
		Q = new T[n * n]();
		R = new T[n * (n + 1)];
		R_tmp = new T[n * n];
		u = new T[n];
		v = new T[n * block_size]();
		v_vector = new T[n];
		factor_block = new T[n];
		w = new T[n * block_size]();

		if (flag)
		{

			cout << "Enter matrix:" << endl;
			for (size_t i = 0; i < n; i++)
				for (size_t j = 0; j < n; j++)
					cin >> R[i * n + j];

			for (size_t i = 0; i < n; i++)

				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

		}
		else {


			for (size_t i = 0; i < n; i++)
				for (size_t j = 0; j < n; j++)

					R[i * n + j] = T(rand()) / RAND_MAX;

			for (size_t i = 0; i < n; i++)
				R[i * n + i] += n;

			for (size_t i = 0; i < n; i++)
				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

			/*auto start{ chrono::steady_clock::now() };
			LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, n, n, REF_A, n, v_vector);
			LAPACKE_dorgqr(LAPACK_ROW_MAJOR, n, n, n, REF_A, n, v_vector);
			auto end{ chrono::steady_clock::now() };
			chrono::duration<double> elapsed_seconds = end - start;
			cout << "Time spent (MKL): " << elapsed_seconds.count() << endl;*/

			for (size_t i = 0; i < n; i++)
				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

		}

	}
	void mult_W_by_V(size_t size);

	//indexing for mult_W_by_V

	inline 	T& W(size_t i, size_t j)
	{
		//W full size is n * block_size
		//starting from (i_start;0)
		return w[(i + i_start) * block_size + j];
	}
	inline T& V(size_t i, size_t j)
	{
		//V full size is n * block_size
		//starting from (0;0)

		return v[i * n + i_start + j];
	}
	inline T& q_tmp(size_t i, size_t j)
	{
		//Q full size is n * n
		//starting from (0;i_start)

		return Q[(i + i_start) * n + (j + i_start)];
	}

	void mult_Q_by_R(size_t i_start);

	//indexing for mult_Q_by_R

	inline 	T& q(size_t i, size_t j)
	{
		//Q full size is n*n
		//starting from (0;0)
		return Q[i * n + j];
	}
	inline T& r_tmp(size_t i, size_t j)
	{
		//R_tmp full size is n*n
		//starting from [i_start][i_start+block_size]
		return R_tmp[(i + i_start) * n + j + i_start + block_size];
	}
	inline T& r(size_t i, size_t j)
	{
		//R full size is n*n
		//starting from [i_start][i_start+block_size]
		return R[(i + i_start) * n + j + i_start + block_size];
	}

	void count_v_gamma(size_t column, size_t num_in_block)
	{
		T scl, gamma;
		//#pragma omp parallel for simd reduction(+: scl) 
		scl = 0;

		for (size_t i = 0; i < n - column; i++)
		{
			u[i] = R[(i + column) * n + column];
			scl += u[i] * u[i];
		}

		if (scl < eps)
		{
			v_vector[column] = 1;
			gamma = 0.5;
			return;
		}

		else
		{
			scl = 1 / sqrt(scl);
			u[0] *= scl;
			gamma = (1 + abs(u[0]));
			v[num_in_block * n + column] = v_vector[column] = sgn(u[0]) * gamma;

			//#pragma omp parallel for simd
			for (size_t i = column + 1; i < n; i++)
			{
				v[num_in_block * n + i] = v_vector[i] = u[i - column] * scl;
			}
		}
	}

	T sgn(T val)
	{
		if (val >= 0)
			return 1;
		else return -1;
	}

	T scal(size_t v_ind, size_t a_col)
	{
		T res = 0;
#pragma omp simd
		for (size_t i = v_ind; i < n; i++)
			res += R[i * n + a_col] * v_vector[i];
		return res;
	}
	void HHolder_R()
	{
		size_t m = 0;

		for (; m < (n / block_size); m++)

			HHolder_Block(m * block_size);

		if (n % block_size != 0)
			HHolder_Block_finish(m * block_size, n - m * block_size);
	}
	void HHolder_Block(size_t i_st)
	{
		i_start = i_st;

		for (size_t i = 0; i < block_size; i++)
			memset(v + i * n, 0, n * sizeof(T));

		for (size_t j = i_start; j < i_start + block_size; j++)
		{
			count_v_gamma(j, j - i_start);

			if (n - i_start >= 256)
			{
#pragma  omp parallel for //512/64 slowdown, 1024/64 minor boost (simd in scal - slowdown)
				for (size_t k = j; k < i_start + block_size; k++)
					factor_block[k - j] = scal(j, k) / abs(v_vector[j]);
			}
			else
				for (size_t k = j; k < i_start + block_size; k++)
					factor_block[k - j] = scal(j, k) / abs(v_vector[j]);

			if (n - i_start >= 256)
			{
#pragma omp parallel for //512/64 slowdown, 1024/64 minor boost
				for (size_t i = j; i < n; i++)
#pragma omp simd
					for (size_t k = j; k < i_start + block_size; k++)
						R[i * n + k] -= v_vector[i] * factor_block[k - j];
			}
			else
				for (size_t i = j; i < n; i++)
					for (size_t k = j; k < i_start + block_size; k++)
						R[i * n + k] -= v_vector[i] * factor_block[k - j];

			for (size_t i = j; i < n; i++)
				R[(i + 1) * n + j] = v_vector[i];

		}

		W_count(i_start);

		Q_count(i_start);

		R_recount(i_start);
	}

	void W_count(size_t i_start)
	{
		memset(w, 0, n * block_size * sizeof(T));
		memset(factor_block, 0, n * sizeof(T));

		for (size_t i = i_start; i < i_start + block_size; i++)
			//#pragma omp parallel for simd private(j)
			for (size_t j = i; j < n; j++)

				factor_block[i] += R[(j + 1) * n + i] * R[(j + 1) * n + i];

		for (size_t i = i_start; i < i_start + block_size; i++)

			factor_block[i] = (-2) / factor_block[i];

		for (size_t i = i_start; i < n; i++)
			w[i * block_size] = factor_block[i_start] * R[(i + 1) * n + i_start];

		for (size_t i = 1; i < block_size; i++)
		{
			memset(Q, 0, n * n * sizeof(T));

			mult_W_by_V(i);

			/*#pragma omp parallel for simd
						for (size_t j = i_start; j < n; j++)
							for (size_t k = i_start; k < n; k++)
								for (size_t m = 0; m < i; m++)
									Q[j * n + k] += w[j * block_size + m] * v[m * n + k];*/
			
						for (size_t j = 0; j < n; j++)
							Q[j * n + j] += 1;
			
						#pragma omp parallel for
						for (size_t j = 0; j < n; j++)
							for (size_t k = i + i_start; k < n; k++)
								w[j * block_size + i] += Q[j * n + k] * R[(k + 1) * n + i + i_start] * factor_block[i + i_start];

			//#pragma omp parallel for /*simd*/ private(j,k,m) //512/64 MULTIPLE boost WHY IS IT CORRECT AND FAST WITHOUT SIMD
			//				for (j = 0; j < n; j++)
			//					for (k = i + i_start; k < n; k++)
			//						for (m = 0; m < i; m++)
			//
			//							w[j * block_size + i] += w[j * block_size + m] * R[(k + 1) * n + m + i_start] * R[(k + 1) * n + i + i_start] * factor_block[i + i_start];
			//
			//			for (j = i + i_start; j < n; j++)
			//				w[j * block_size + i] += R[(j + 1) * n + i + i_start] * factor_block[i + i_start];
		}

	}

	void R_recount(size_t i_start)
	{
#pragma omp parallel for //512/64 no result
		for (size_t i = 0; i < n; i++)
			copy(R + i * n, R + (i + 1) * n, R_tmp + i * n);

		for (size_t i = i_start; i < n; i++)
			memset(R + i * n + i_start + block_size, 0, (n - i_start - block_size) * sizeof(T));

		//		if (n - i_start >= 64)
		//		{
		//#pragma omp parallel for simd //512/64 obvious boost WHY IS IT CORRECT
		//			for (size_t k = i_start + block_size; k < n; k++)
		//				for (size_t j = i_start; j < n; j++)
		//					for (size_t m = i_start; m < n; m++)
		//
		//						R[j * n + k] += Q[(j - i_start) * n + m - i_start] * R_tmp[m * n + k];
		//		}
		//		else {
		//			for (size_t k = i_start + block_size; k < n; k++)
		//				for (size_t j = i_start; j < n; j++)
		//					for (size_t m = i_start; m < n; m++)
		//
		//						R[j * n + k] += Q[(j - i_start) * n + m - i_start] * R_tmp[m * n + k];;
		//		}
		mult_Q_by_R(i_start);
	}

	void Q_count(size_t i_start)
	{
		memset(Q, 0, n * n * sizeof(T));



		if (n - i_start >= 64)
		{
#pragma omp parallel for //512/64 slowdown
			for (size_t j = i_start; j < n; j++)
				for (size_t k = i_start; k < n; k++)
#pragma omp simd
					for (size_t m = i_start; m < i_start + min((j + 1 - i_start), block_size); m++)

						Q[(j - i_start) * n + k - i_start] += R[(j + 1) * n + m] * w[k * block_size + (m - i_start)];
		}
		else
		{
			for (size_t j = i_start; j < n; j++)
				for (size_t k = i_start; k < n; k++)
					for (size_t m = i_start; m < i_start + min((j + 1 - i_start), block_size); m++)

						Q[(j - i_start) * n + k - i_start] += R[(j + 1) * n + m] * w[k * block_size + (m - i_start)];
		}

		for (size_t j = 0; j < n; j++)
			Q[j * n + j] += 1;
	}

	void HHolder_Block_finish(size_t i_start, size_t b_size)
	{
		for (size_t j = i_start; j < i_start + b_size; j++)
		{
			count_v_gamma(j, j - i_start);

			for (size_t k = j; k < i_start + b_size; k++)
				factor_block[k - j] = scal(j, k) / abs(v_vector[j]);

			for (size_t i = j; i < n; i++)
				for (size_t k = j; k < i_start + b_size; k++)
					R[i * n + k] -= v_vector[i] * factor_block[k - j];
		}
	}


	void HHolder_Q()
	{
		//#pragma omp parallel for private(i)	//512/64, 1024/64 no result
		for (size_t i = 0; i < n; i++)
			copy(REF_A + i * n, REF_A + (i + 1) * n, Q + i * n);

		for (size_t i = 0; i < n; i++)
		{
#pragma omp parallel for //512/64 no result, 1024/64 small boost
			for (size_t j = 0; j < n; j++)
				Q[j * n + i] /= R[i * n + i];

			copy(R + i * (n + 1) + 1, R + (i + 1) * n, R_tmp);

#pragma omp parallel for //512/64 obvious boost
			for (size_t k = 0; k < n; k++)
				for (size_t j = i + 1; j < n; j++)

					Q[k * n + j] -= R_tmp[j - (i + 1)] * Q[k * n + i];

		}
	}

	bool check()
	{
		memset(R_tmp, 0, n * n * sizeof(T));

		for (size_t i = 0; i < n; i++)
			for (size_t j = 0; j < n; j++)
				for (size_t k = 0; k <= j; k++)
					R_tmp[i * n + j] += Q[i * n + k] * R[k * n + j];

		for (size_t i = 0; i < n; i++)
			for (size_t j = 0; j < n; j++)
				if (abs(REF_A[i * n + j] - R_tmp[i * n + j]) >= eps)
					return false;

		memset(R_tmp, 0, n * n * sizeof(T));

		for (size_t i = 0; i < n; i++)										//is ortogonal?
			for (size_t k = 0; k < n; k++)
				for (size_t j = 0; j < n; j++)
					R_tmp[i * n + j] += Q[i * n + k] * Q[j * n + k];

		for (size_t i = 0; i < n; i++)
		{
			for (size_t k = 0; k < i; k++)
				if (abs(R_tmp[i * n + k]) >= eps)
					return false;

			if (abs(R_tmp[i * n + i] - 1) >= eps)
				return false;

			for (size_t k = i + 1; k < n; k++)
				if (abs(R_tmp[i * n + k]) >= eps)
					return false;

		}

		return true;
	}

	void out(char s)
	{
		cout << endl;
		if (s == 'R')

			for (size_t i = 0; i < n; i++)
			{
				for (size_t j = 0; j < n; j++)
					cout << R[i * n + j] << ' ';
				cout << endl;

			}

		else  if (s == 'Q')

			for (size_t i = 0; i < n; i++)
			{
				for (size_t j = 0; j < n; j++)
					cout << Q[i * n + j] << ' ';
				cout << endl;
			}
		cout << endl;
	}

	void transpv()
	{
		for (size_t i = 0; i < n; i++)
			for (size_t j = i + 1; j < n; j++)
				swap(Q[i * n + j], Q[j * n + i]);

		return;
	}

	~QR()
	{
		delete[]Q;
		delete[]R;
		delete[]u;
		delete[]REF_A;
		delete[]factor_block;
		delete[]w;
		delete[]R_tmp;

		Q = R = u = REF_A = factor_block = w = R_tmp = nullptr;
	}
};