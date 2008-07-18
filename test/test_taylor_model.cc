/***************************************************************************
 *            test_taylor_model.cc
 *
 *  Copyright  2007  Pieter Collins
 *  Email Pieter.Collins@cwi.nl
 ****************************************************************************/

/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Library General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */


#include <iostream>
#include <fstream>
#include <cassert>

#include "numeric/rational.h"
#include "test/test_float.h"
#include "test/test.h"

#include "linear_algebra/vector.h"
#include "linear_algebra/matrix.h"
#include "differentiation/multi_index.h"
#include "function/taylor_model.h"
#include "output/latexstream.h"

using namespace std;
using namespace Ariadne;


template<class R> 
class TestTaylorModel
{
  typedef typename traits<R>::arithmetic_type F;

 public:
  
  TestTaylorModel() {
  }
  
  void test_build() {
    std::cout << "test_build()\n";
    TaylorModel<R> p(1,2,3);
    std::cout << p << std::endl;
    assert(p.data().size()==10);
    MultiIndex j(4);
    p.set(0,j,1);
    p.set(0,++j,2);
    p.set(0,++j,3);
    p.set(0,++j,4);
    p.set(0,++j,5);
    p.set(0,++j,6);
    std::cout << p << std::endl;
  }

  void test_input() {
    std::cout << "test_input()\n";
    double a[10]={1,2,3,4,5,6,7,8,9,10};
    TaylorModel<R> p(1,2,3,3,a);
    std::cout << p << std::endl;
  }

  void test_output() {
  }

  void test_evaluate() {
    std::cout << "test_evaluate()\n";
    double a[10]={1,2,3,4,5,6,7,8,9,10};
    TaylorModel<R> p1(1,2,3,3,a);
    Vector<R> x(2,a,1);
    Vector<R> r=p1.evaluate(x);
    std::cout << r << std::endl;
  }

  void test_component() {
    std::cout << "test_component()\n";
    double a[12]={1,2,3,4,5,6,7,8,9,10,11,12};
    double a0[6]={1,3,5,7,9,11};
    double a1[6]={2,4,6,8,10,12};
    TaylorModel<R> p(2,2,2,2,a);
    TaylorModel<R> p0(1,2,2,2,a0);
    TaylorModel<R> p1(1,2,2,2,a1);
    ARIADNE_TEST_CHECK(p.component(0),p0);
    ARIADNE_TEST_CHECK(p.component(1),p1);
  }

  void test_add() {
    std::cout << "test_add()\n";
    double a[10]={1,2,3,4,5,6,7,8,9,10};
    TaylorModel<R> p1(1,2,3,3,a);
    TaylorModel<R> p2(1,2,2,2,a);
    //TaylorModel<F> p=add(p1,p2);
    TaylorModel<F> p=p1+p2;
    std::cout << p << std::endl;
  }

  void test_mul() {
    std::cout << "test_mul()\n";
    double a[10]={1,2,3,4,5,6,7,8,9,10};
    TaylorModel<R> p1(1,2,3,3,a);
    TaylorModel<R> p2(1,2,2,2,a);
    TaylorModel<F> p=p1*p2;
    std::cout << p << std::endl;
  }

  void test_pow() {
    std::cout << "test_pow()\n";
    double a0[3]={1,0,0};
    double a1[6]={1,1,1,1,1,1};
    double a2[21]={1,2,2,3,4,3,2,4,4,2,1,2,3,2,1,0,0,0,0,0,0};
    double a3[28]={1,3,3,6,9,6,7,15,15,7,6,15,21,15,6,3,9,15,15,9,3,1,3,6,7,6,3,1};
    TaylorModel<R> p0(1,2,1,1,a0);
    TaylorModel<R> p1(1,2,2,2,a1);
    TaylorModel<R> p2(1,2,5,5,a2);
    TaylorModel<R> p3(1,2,6,6,a3);
    TaylorModel<R> p=p1;
    std::cout << "p=" << p << std::endl;
    std::cout << "pow(p,0)=" << pow(p,0) << std::endl;
    std::cout << "pow(p,1)=" << pow(p,1) << std::endl;
    std::cout << "pow(p,2)=" << pow(p,2) << std::endl;
    std::cout << "pow(p,3)=" << pow(p,3) << std::endl;
    ARIADNE_TEST_ASSERT(pow(p,0)==p0);
    ARIADNE_TEST_ASSERT(pow(p,1)==p1);
    ARIADNE_TEST_ASSERT(pow(p,2)==p2);
    ARIADNE_TEST_ASSERT(pow(p,3)==p3);
  }


  void test_compose() {
    std::cout << "test_compose()\n";
    double a1[6]={1,1,1,1,1,1};
    double a2[8]={1,1,1,1,1,1,1,1};
    double a3[20]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    double r12[6]={6,8,8,3,6,3};
    double r23[20]={3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    TaylorModel<R> p1(1,2,2,2,a1);
    TaylorModel<R> p2(2,2,1,1,a2);
    TaylorModel<R> p3(2,3,2,2,a3);
    //std::cout << "p1=" << p1 << std::endl;
    //std::cout << "p2=" << p2 << std::endl;
    //std::cout << "p3=" << p3 << std::endl;
    //std::cout << "compose(p1,p2)=" << compose(p1,p2) << std::endl;
    //std::cout << "compose(p2,p3)=" << compose(p2,p3) << std::endl;
    ARIADNE_TEST_CHECK(compose(p1,p2),TaylorModel<F>(1,2,2,2,r12));
    ARIADNE_TEST_CHECK(compose(p2,p3),TaylorModel<F>(2,3,2,2,r23));
  }


  void test_jacobian() {
    std::cout << "test_jacobian()\n";
    double av1[2]={1,1};
    double av2[2]={2,3};
    Vector<R> v0(2);
    Vector<R> v1(2,av1);
    Vector<R> v2(2,av2);
    double a1[6]={0,1,2,3,5,7};
    double a2[8]={1,1,1,1,1,1,1,1};
    double a3[20]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    double aj10[2]={1,2};
    double aj11[2]={12,21};
    double aj12[2]={28,54};
    TaylorModel<R> p1(1,2,2,2,a1);
    TaylorModel<R> p2(2,2,1,1,a2);
    TaylorModel<R> p3(2,3,2,2,a3);
    Matrix<R> j10(1,2,aj10);
    Matrix<R> j11(1,2,aj11);
    Matrix<R> j12(1,2,aj12);
    ARIADNE_TEST_PRINT(p1);
    ARIADNE_TEST_PRINT(v0);
    ARIADNE_TEST_PRINT(v1);
    ARIADNE_TEST_PRINT(v2);
    ARIADNE_TEST_CHECK(p1.jacobian(v0),j11);
    ARIADNE_TEST_CHECK(p1.jacobian(v1),j11);
    ARIADNE_TEST_CHECK(p1.jacobian(v2),j12);
  }

  void test_inverse() {
    std::cout << "test_inverse()\n";
    double av1[2]={0.125,0.0};
    Vector<R> v0(2);
    Vector<R> v1(2,av1);
    double a1[12]={0,0,2,3,5,7,1,2,3,4,6,8};
    TaylorModel<R> p1(2,2,2,2,a1);
    ARIADNE_TEST_PRINT(p1);
    ARIADNE_TEST_PRINT(v0);
    ARIADNE_TEST_PRINT(v1);
    ARIADNE_TEST_EVALUATE(inverse(p1,v0));
    //ARIADNE_TEST_EVALUATE(inverse(p1,v1));
  }

  void test_latex_output() {
    double a1[30]={1,-1,0,4,5,0,7,8,-9,-1,1,0,13,14,15,16,17,18,19,20,0,-1,23,24,25,26,27,28,29,30};
    TaylorModel<R> p1(3,3,2,2,a1);
    double a2[30]={1.5,0,0,1,-0.375,0,-1,0,0,0,0,0};
    TaylorModel<R> p2(2,2,2,2,a2);
    latexfstream tex; tex.open("test_taylor_model.tex","\\usepackage{amsmath}");
    tex << "\\begin{gather}\n" << p1 << "\\end{gather}\n";
    tex << "\\begin{gather}\n" << p2 << "\\end{gather}\n";
  }

  void test() {
    test_build();
    test_input();
    test_output();
    test_latex_output();
    test_evaluate();
    test_component();
    test_add();
    test_mul();
    test_pow();
    test_compose();
    test_jacobian();
    test_inverse();
  }

};


int main() {
  //TestTaylorModel<Float>().test();
  return ARIADNE_TEST_FAILURES;
}

