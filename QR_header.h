#pragma once
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <mkl.h>

//const size_t bs = 32;
using namespace std;
template <typename T>
class QR
{
private:

	size_t i, j, k, n, m, block_size;
	T* Q, * R, * REF_A, * u, * factor_block, * w, * z, * R_tmp, * v_vector;
	T norm, eps = 1e-10;

public:

	QR(bool flag, size_t size, size_t b_s) : n(size), block_size(b_s) //flag = 0 - ââîä ðàíäîìíûõ ÷èñåë, èíà÷å ââîä ñ êëàâèàòóðû
	{
		REF_A = new T[n * n];
		Q = new T[n * n]();
		R = new T[n * (n + 1)];
		R_tmp = new T[n * n];
		u = new T[n];
		v_vector = new T[n];
		factor_block = new T[n];
		w = new T[n * block_size]();

		if (flag)
		{

			cout << "Enter matrix:" << endl;
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++)
					cin >> R[i * n + j];

			for (i = 0; i < n; i++)

				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

		}
		else {


			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++)

					R[i * n + j] = T(rand()) / RAND_MAX;

			for (i = 0; i < n; i++)
				R[i * n + i] += n;

			for (i = 0; i < n; i++)
				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

			/*auto start{ chrono::steady_clock::now() };
			LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, n, n, REF_A, n, v_vector);
			LAPACKE_dorgqr(LAPACK_ROW_MAJOR, n, n, n, REF_A, n, v_vector);
			auto end{ chrono::steady_clock::now() };
			chrono::duration<double> elapsed_seconds = end - start;
			cout << "Time spent (MKL): " << elapsed_seconds.count() << endl;*/

			for (i = 0; i < n; i++)
				copy(R + i * n, R + (i + 1) * n, REF_A + i * n);

		}

	}

	double* returnR()
	{
		return R;
	}

	double* returnV()
	{
		return v;
	}

	void count_v_gamma(size_t column)
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
			v_vector[column] = sgn(u[0]) * gamma;

			//#pragma omp parallel for simd
			for (size_t i = column + 1; i < n; i++)
				v_vector[i] = u[i - column] * scl;
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
	void HHolder_Block(size_t i_start)
	{
		size_t num_in_block;

		for (j = i_start; j < i_start + block_size; j++)
		{
			num_in_block = j - i_start;
			count_v_gamma(j);

			if (n - i_start >= 512)
			{
#pragma  omp parallel for private (k) //512/64 slowdown, 1024/64 minor boost (simd in scal - slowdown)
				for (k = j; k < i_start + block_size; k++)
					factor_block[k - j] = scal(j, k) / abs(v_vector[j]);
			}
			else
				for (k = j; k < i_start + block_size; k++)
					factor_block[k - j] = scal(j, k) / abs(v_vector[j]);

			if (n - i_start >= 512)
			{
#pragma omp parallel for simd private (i) //512/64 slowdown, 1024/64 minor boost
				for (i = j; i < n; i++)
					for (k = j; k < i_start + block_size; k++)
						R[i * n + k] -= v_vector[i] * factor_block[k - j];
			}
			else
			for (i = j; i < n; i++)
				for (k = j; k < i_start + block_size; k++)
					R[i * n + k] -= v_vector[i] * factor_block[k - j];

			for (i = j; i < n; i++)
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

		for (i = i_start; i < i_start + block_size; i++)
			for (j = i; j < n; j++)

				factor_block[i] += R[(j + 1) * n + i] * R[(j + 1) * n + i];

		for (i = i_start; i < i_start + block_size; i++)

			factor_block[i] = (-2) / factor_block[i];

		for (i = i_start; i < n; i++)
			w[i * block_size] = factor_block[i_start] * R[(i + 1) * n + i_start];

		for (i = 1; i < block_size; i++)
		{
			//memset(Q, 0, n * n * sizeof(T));

//#pragma omp parallel for simd private(j,k,m)
//			for (j = 0; j < n; j++)
//				for (k = i + i_start; k < n; k++)
//					for (m = 0; m < i; m++)
//						Q[j * n + k] += w[j * block_size + m] * R[(k + 1) * n + m + i_start];
//
//			for (j = 0; j < n; j++)
//				Q[j * n + j] += 1;
//
//			#pragma omp parallel for private(j)
//			for (j = 0; j < n; j++)
//				for (k = i + i_start; k < n; k++)
//					w[j * block_size + i] += Q[j * n + k] * R[(k + 1) * n + i + i_start] * factor_block[i + i_start];

#pragma omp parallel for /*simd*/ private(j,k,m) //512/64 MULTIPLE boost WHY IS IT CORRECT AND FAST WITHOUT SIMD
			for (j = 0; j < n; j++)
				for (k = i + i_start; k < n; k++)
					for (m = 0; m < i; m++)

						w[j * block_size + i] += w[j * block_size + m] * R[(k + 1) * n + m + i_start] * R[(k + 1) * n + i + i_start] * factor_block[i + i_start];

			for (j = i + i_start; j < n; j++)
				w[j * block_size + i] += R[(j + 1) * n + i + i_start] * factor_block[i + i_start];
		}

	}
	
	void R_recount(size_t i_start)
	{
#pragma omp parallel for private (i) //512/64 no result
		for (i = 0; i < n; i++)
			copy(R + i * n, R + (i + 1) * n, R_tmp + i * n);

		for (i = i_start; i < n; i++)
			memset(R + i * n + i_start + block_size, 0, (n - i_start - block_size) * sizeof(T));

#pragma omp parallel for simd private(k,j) //512/64 obvious boost WHY IS IT CORRECT
		for (k = i_start + block_size; k < n; k++)
			for (j = i_start; j < n; j++)
				for (m = 0; m < n; m++)

					R[j * n + k] += Q[j * n + m] * R_tmp[m * n + k];
	}

	void Q_count(size_t i_start)
	{
		memset(Q, 0, n * n * sizeof(T));

#pragma omp parallel for simd private(k,j,m) //512/64 slowdown
		for (j = i_start; j < n; j++)
			for (k = i_start; k < n; k++)
				for (m = i_start; m < i_start + min((j + 1 - i_start), block_size); m++)

					Q[j * n + k] += R[(j + 1) * n + m] * w[k * block_size + (m - i_start)];

		for (j = 0; j < n; j++)
			Q[j * n + j] += 1;
	}

	void HHolder_Block_finish(size_t i_start, size_t b_size)
	{
		size_t num_in_block;

		for (j = i_start; j < i_start + b_size; j++)
		{
			num_in_block = j - i_start;
			count_v_gamma(j);

			for (k = j; k < i_start + b_size; k++)
				factor_block[k - j] = scal(j, k) / abs(v_vector[j]);

			for (i = j; i < n; i++)
				for (k = j; k < i_start + b_size; k++)
					R[i * n + k] -= v_vector[i] * factor_block[k - j];
		}
	}


	void HHolder_Q()
	{
		//#pragma omp parallel for private(i)	//512/64, 1024/64 no result
		for (i = 0; i < n; i++)
			copy(REF_A + i * n, REF_A + (i + 1) * n, Q + i * n);

		for (i = 0; i < n; i++)
		{
#pragma omp parallel for private(j) //512/64 no result, 1024/64 small boost
			for (j = 0; j < n; j++)
				Q[j * n + i] /= R[i * n + i];

			copy(R + i * (n + 1) + 1, R + (i + 1) * n, R_tmp);

#pragma omp parallel for private(k,j) //512/64 obvious boost
			for (k = 0; k < n; k++)
				for (j = i + 1; j < n; j++)

					Q[k * n + j] -= R_tmp[j - (i + 1)] * Q[k * n + i];

		}
	}

	bool check()
	{
		memset(R_tmp, 0, n * n * sizeof(T));

		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				for (k = 0; k <= j; k++)
					R_tmp[i * n + j] += Q[i * n + k] * R[k * n + j];

		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				if (abs(REF_A[i * n + j] - R_tmp[i * n + j]) >= eps)
					return false;

		memset(R_tmp, 0, n * n * sizeof(T));

		for (i = 0; i < n; i++)										//is ortogonal?
			for (k = 0; k < n; k++)
				for (j = 0; j < n; j++)
					R_tmp[i * n + j] += Q[i * n + k] * Q[j * n + k];

		for (i = 0; i < n; i++)
		{
			for (k = 0; k < i; k++)
				if (abs(R_tmp[i * n + k]) >= eps)
					return false;

			if (abs(R_tmp[i * n + i] - 1) >= eps)
				return false;

			for (k = i + 1; k < n; k++)
				if (abs(R_tmp[i * n + k]) >= eps)
					return false;

		}

		return true;
	}

	void out(char s)
	{
		cout << endl;
		if (s == 'R')

			for (i = 0; i < n; i++)
			{
				for (j = 0; j < n; j++)
					cout << R[i * n + j] << ' ';
				cout << endl;

			}

		else  if (s == 'Q')

			for (i = 0; i < n; i++)
			{
				for (j = 0; j < n; j++)
					cout << Q[i * n + j] << ' ';
				cout << endl;
			}
		cout << endl;
	}

	void transpQ()
	{
		for (i = 0; i < n; i++)

			//#pragma omp parallel for private(j)
			for (j = i + 1; j < n; j++)
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
		i = j = k = m = 0;
	}
};