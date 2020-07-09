/***************************************************************************
 *            test_procedure.cpp
 *
 *  Copyright  2010-20  Pieter Collins
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "config.hpp"

#include "function/procedure.hpp"
#include "function/procedure.tpl.hpp"

#include "cuda/cuda_lib.hpp"

#include "../test.hpp"

using namespace std;
using namespace Ariadne;

template<class X> decltype(auto) mag(Covector<X> const& u) { return norm(transpose(u)); }

class TestCudaProcedure
{
  public:
    TestCudaProcedure();
    Void test();
  private:
    Void test_matrix_moltiplication();
};

TestCudaProcedure::TestCudaProcedure()
{
}

Void TestCudaProcedure::test()
{
    ARIADNE_TEST_CALL(test_matrix_moltiplication());
}

Void TestCudaProcedure::test_matrix_moltiplication()
{
    const int N = 5;
    int* h_matrixA = new int[N * N];
    int* h_matrixB = new int[N * N];
    int* h_matrixC = new int[N * N];
    int* h_matrixC_host = new int[N * N];

    for (int i = 0; i < N * N; i++) {
        h_matrixA[i] = i;
        h_matrixB[i] = i+1;
    }

    ariadne_cuda::function(N, h_matrixA, h_matrixB, h_matrixC);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                 sum += h_matrixA[i * N + k] * h_matrixB[k * N + j];
            h_matrixC_host[i * N + j] = sum;
        }
    }
    for (int i = 0; i < N * N; i++) {
        if (h_matrixC_host[i] != h_matrixC[i]) {
            ARIADNE_TEST_FAILURES++;
        }
    }

}



Int main() {
    TestCudaProcedure().test();
    return ARIADNE_TEST_FAILURES;
}

