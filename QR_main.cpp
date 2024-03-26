#include <omp.h>
#include "QR_header.h"

int main()
{
	//int count;
	size_t n, block_size;
	//cout << "Enter matrix dimentions: " << endl;
	//cin >> n;
	//while (1)
	//{
	n = 1024;
		//cout << "Enter block size: ";
		block_size = 40;
		//cin >> block_size;
		//count = 0;
		/*while (count < 5)
		{*/

			QR t(0, n, block_size);
			//auto start_A{ chrono::steady_clock::now() };

			t.HHolder_R();

			//t.out('R');

			//auto end_A{ chrono::steady_clock::now() };
			//chrono::duration<double> elapsed_seconds_A = end_A - start_A;
			//auto start_Q = chrono::steady_clock::now();

			t.HHolder_Q();

			//t.out('Q');

			//auto end_Q = chrono::steady_clock::now();
			//chrono::duration<double>  elapsed_seconds_Q = end_Q - start_Q;
			//cout << "Time spent: " << elapsed_seconds_A.count() << " sec + " << elapsed_seconds_Q.count() << " sec = ";
			//elapsed_seconds_A += elapsed_seconds_Q;
			//cout << elapsed_seconds_A.count() << endl;
			//count++;

			/*if (t.check())

				cout << "QR-decomposition is correct";
			else cout << "QR-decomposition is incorrect";

			cout << endl;*/
			////}
	//}
}