/***************************************************************************
 *            test_nonlinear_programming.cc
 *
 *  Copyright  2010  Pieter Collins
 *
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

#include "test.h"

#include "numeric.h"
#include "vector.h"
#include "function.h"
#include "nonlinear_programming.h"
#include "box.h"


using namespace std;
using namespace Ariadne;

class TestOptimiser
{
  private:
    scoped_ptr<OptimiserInterface> optimiser;
  public:
    TestOptimiser(const OptimiserInterface& opt)
        : optimiser(opt.clone()) { }

    void test() {
        //ARIADNE_TEST_CALL(test_mixed_constrained_optimisation()); return;
        ARIADNE_TEST_CALL(test_feasibility_check());
        ARIADNE_TEST_CALL(test_unconstrained_optimisation());
        ARIADNE_TEST_CALL(test_constrained_optimisation());
        ARIADNE_TEST_CALL(test_equality_constrained_optimisation());
        ARIADNE_TEST_CALL(test_linear_feasibility());
        ARIADNE_TEST_CALL(test_nonlinear_feasibility());
    }

    void test_unconstrained_optimisation() {
        // Test the feasibility of x0>0, x1>0, 2x1+x2<1 using box [0,2]x[0,2]
        List<RealScalarFunction> x=RealScalarFunction::coordinates(2);
        RealScalarFunction x0s = sqr(x[0]);
        RealScalarFunction f(x0s*(12+x0s*(6.3+x0s))+6*x[1]*(x[1]-x[0]));
        ARIADNE_TEST_PRINT(f);
        RealVectorFunction g(0u,2u);
        ARIADNE_TEST_PRINT(g);
        Box D(2, -1.0,2.0, -3.0,5.0);
        Box C(0);

        IntervalVector x_optimal=optimiser->minimise(f,D,g,C);
        ARIADNE_TEST_BINARY_PREDICATE(subset,x_optimal,D);
        ARIADNE_TEST_BINARY_PREDICATE(subset,g(x_optimal),C);
        //ARIADNE_TEST_LESS(norm(x_optimal),1e-8);
    }

    void test_equality_constrained_optimisation() {
        List<RealScalarFunction> x=RealScalarFunction::coordinates(2);
        RealScalarFunction f(+(sqr(x[0])+sqr(x[1])));
        ARIADNE_TEST_PRINT(f);
        RealVectorFunction h( (1.5+x[0]+2*x[1]+0.25*x[0]*x[1]) );
        ARIADNE_TEST_PRINT(h);

        Box D(2, -1.0,2.0, -3.0,5.0);

        IntervalVector x_optimal=optimiser->minimise(f,D,h);
        ARIADNE_TEST_BINARY_PREDICATE(subset,x_optimal,D);
        ARIADNE_TEST_LESS(norm(h(x_optimal)),1e-7);
    }

    void test_constrained_optimisation() {
        List<RealScalarFunction> x=RealScalarFunction::coordinates(3);
        RealScalarFunction x0s = sqr(x[0]);
        RealScalarFunction f(x0s*(12+x0s*(6.3+x0s))+6*x[1]*(x[1]-x[0])+x[2]);
        ARIADNE_TEST_PRINT(f);
        //RealVectorFunction g( (x[0]-1, x[0]+x[1]*x[1], x[1]*x[1]) );
        RealVectorFunction g( (2*x[1]+x[0], x[0]+x[1]*x[1]-0.875) );
        ARIADNE_TEST_PRINT(g);
        Box D(3, -1.0,2.0, -3.0,5.0, -3.0,5.0);
        Box C(2, 0.0,infty, 0.0,infty, 3.0,3.0);

        IntervalVector x_optimal=optimiser->minimise(f,D,g,C);
        ARIADNE_TEST_BINARY_PREDICATE(subset,x_optimal,D);
        ARIADNE_TEST_BINARY_PREDICATE(subset,g(x_optimal),C);
        //ARIADNE_TEST_LESS(norm(x_optimal),1e-6);
    }

    void test_mixed_constrained_optimisation() {
        List<RealScalarFunction> x=RealScalarFunction::coordinates(3);
        RealScalarFunction f(+(sqr(x[0])+sqr(x[1])+x[1]*x[2]));
        ARIADNE_TEST_PRINT(f);
        RealVectorFunction g( x[0]*x[1]-x[0]*1.25 );
        ARIADNE_TEST_PRINT(g);
        RealVectorFunction h( (1.5+x[0]+2*x[1]+0.25*x[0]*x[1]) );
        ARIADNE_TEST_PRINT(h);

        Box D(3, -1.0,2.0, -3.0,5.0, 1.25,2.25);
        Box C(1, -1.0,-0.5);

        IntervalVector x_optimal=optimiser->minimise(f,D,g,C,h);
        ARIADNE_TEST_LESS(norm(h(x_optimal)),1e-8);
    }

    void test_linear_feasibility() {
        // Test the feasibility of x0>0, x1>0, 2x1+x2<1 using box [0,2]x[0,2]
        List<RealScalarFunction> x=RealScalarFunction::coordinates(2);
        RealVectorFunction g=RealVectorFunction(1u, 2*x[0]+x[1]);
        ARIADNE_TEST_PRINT(g);
        Box D(2, 0.0,2.0, 0.0,2.0);
        Box C(1, -2.0,1.0);

        ARIADNE_TEST_ASSERT(optimiser->feasible(D,g,C));
        C=Box(1, 1.0,1.5);
        ARIADNE_TEST_ASSERT(optimiser->feasible(D,g,C));
        D=Box(2, 1.0,1.5,0.5,1.0);
        ARIADNE_TEST_ASSERT(!optimiser->feasible(D,g,C));
    }

    void test_nonlinear_feasibility() {
        // Test the feasibility of x0>0, x1>0, 2x1+x2<1 using box [0,2]x[0,2]
        List<RealScalarFunction> x=RealScalarFunction::coordinates(2);
        RealVectorFunction g=RealVectorFunction(1u, 2*x[0]+x[1]+0.125*x[0]*x[1]);
        ARIADNE_TEST_PRINT(g);
        Box D(2, 0.0,2.0, 0.0,2.0);
        Box C(1, -2.0,1.0);

        ARIADNE_TEST_ASSERT(optimiser->feasible(D,g,C));
        C=Box(1, 1.0,1.5);
        ARIADNE_TEST_ASSERT(optimiser->feasible(D,g,C));
        D=Box(2, 1.0,1.5,0.5,1.0);
        ARIADNE_TEST_ASSERT(!optimiser->feasible(D,g,C));
    }

    void test_nonlinear_equality_feasibility() {
        // Test the feasibility of x0>0, x1>0, 2x1+x2<1 using box [0,2]x[0,2]
        List<RealScalarFunction> x=RealScalarFunction::coordinates(2);
        RealVectorFunction h=RealVectorFunction(1u, 2*x[0]-x[1]+0.125*x[0]*x[1]);
        ARIADNE_TEST_PRINT(h);
        Box D(2, 0.0,2.0, 0.0,2.0);

        ARIADNE_TEST_ASSERT(optimiser->feasible(D,h));
    }

    void test_feasibility_check() {
        RealVectorFunction x=RealVectorFunction::identity(2);
        ARIADNE_TEST_CONSTRUCT( RealVectorFunction, g,(sqr(x[0])+2*sqr(x[1])-1) );
        ARIADNE_TEST_CONSTRUCT( IntervalVector, D,(2, -1.0, 1.0, -1.0,1.0) );
        ARIADNE_TEST_CONSTRUCT( IntervalVector, C,(1, 0.0, 0.0) );

        ARIADNE_TEST_CONSTRUCT( IntervalVector, X1,(2, 0.30,0.40, 0.60,0.70) );
        ARIADNE_TEST_ASSERT( definitely(optimiser->contains_feasible_point(D,g,C,X1)) );

        // The following test fails since it is difficult to find the feasible
        // point in the box.
        ARIADNE_TEST_CONSTRUCT( IntervalVector, X2,(2, 0.30,0.40, 0.65,0.66) );
        ARIADNE_TEST_ASSERT( optimiser->contains_feasible_point(D,g,C,X2) );
    }

};

int main(int argc, const char* argv[]) {
    uint optimiser_verbosity = get_verbosity(argc,argv);

    NonlinearInteriorPointOptimiser nlo;
    nlo.verbosity=optimiser_verbosity;
    TestOptimiser(nlo).test();

    ApproximateOptimiser appo;
    appo.verbosity=optimiser_verbosity;
    TestOptimiser(appo).test_nonlinear_equality_feasibility();

    IntervalOptimiser ivlo;
    ivlo.verbosity=optimiser_verbosity;
    //TestOptimiser(ivlo).test_nonlinear_equality_feasibility();
    return ARIADNE_TEST_FAILURES;
}

