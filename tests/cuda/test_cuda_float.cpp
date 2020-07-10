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

#include "../test.hpp"
#include "cuda/cuda_lib.hpp"

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

#define round_up 0
#define round_down 1
#define round_to_nearest 2
#define round_toward_zero 3

using namespace std;
using namespace Ariadne;

template<class X> decltype(auto) mag(Covector<X> const& u) { return norm(transpose(u)); }

class TestCudaFloat
{
  public:
    TestCudaFloat();
    Void test();
  private:
    Void test_cuda_rounding();
};

TestCudaFloat::TestCudaFloat()
{
}

Void TestCudaFloat::test()
{
  std::cout<<std::setprecision(20);
  ARIADNE_TEST_CALL(test_cuda_rounding());
}

Void TestCudaFloat::test_cuda_rounding()
{   

  volatile double one   = 1;
  volatile double two  = 2;
  volatile double three = 3;
  volatile double five  = 5;

  const double onethirddown    = 0.33333333333333331483;
  const double onethirdup      = 0.33333333333333337034;
  // This value are not used because this type of ruonding is not implemented
  //const double onethirdchop    = 0.33333333333333331483;
  //const double onethirdnearest = 0.33333333333333331483;
  const double twofifthsdown   = 0.39999999999999996669;
  const double twofifthsup     = 0.40000000000000002220;
  //const double twofifthschop   = 0.39999999999999996669;
  //const double twofifthsnearest= 0.40000000000000002220;

  double result = ariadne_cuda::double_approximation(one, three, DIV, round_down);
  ARIADNE_TEST_EQUAL(result, onethirddown);
  result = ariadne_cuda::double_approximation(one, three, DIV, round_up);
  ARIADNE_TEST_EQUAL(result, onethirdup);

  /*
    NOT IMPLEMENTED -- ROUNDING "toward_zero", ROUNDING "round_to_nearest"
  result = ariadne_cuda::double_approximation(one, three, DIV, toward_zero);
  ARIADNE_TEST_EQUAL(result, onethirdchop);
  result = ariadne_cuda::double_approximation(one, three, DIV, round_to_nearest);
  ARIADNE_TEST_EQUAL(result, onethirdnearest);
  */

  result = ariadne_cuda::double_approximation(two, five, DIV, round_down);
  ARIADNE_TEST_EQUAL(result, twofifthsdown);
  result = ariadne_cuda::double_approximation(two, five, DIV, round_up);
  ARIADNE_TEST_EQUAL(result, twofifthsup);

  /*
    NOT IMPLEMENTED -- ROUNDING "toward_zero", ROUNDING "round_to_nearest"
  result = ariadne_cuda::double_approximation(two, five, DIV, toward_zero);
  ARIADNE_TEST_EQUAL(result, twofifthschop);
  result = ariadne_cuda::double_approximation(two, five, DIV, round_to_nearest);
  ARIADNE_TEST_EQUAL(result, twofifthsnearest);
  */

}



Int main() {
    TestCudaFloat().test();
    return ARIADNE_TEST_FAILURES;
}

