/***************************************************************************
 *            test_textplot.cpp
 *
 *  Copyright 2009--17  Davide Bresolin
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

#include "function/function.hpp"
#include "output/textplot.hpp"
#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/zonotope.hpp"
#include "geometry/polytope.hpp"
#include "geometry/curve.hpp"
#include "taylor_set.hpp"
#include "geometry/function_set.hpp"
#include "geometry/grid_paving.hpp"

#include "function/user_function.hpp"

using namespace Ariadne;


struct RadiusSquare : EffectiveVectorFunctionData<1,2,1> {
    template<class R, class A, class P>
    static Void compute(R& r, const A& x, const P& p) {
        r[0]=sqr(x[0])+sqr(x[1])-sqr(p[0]);
    }
};



Int main(Int argc, char **argv)
{

    ExactBoxType bx1(2); bx1[0]=ExactIntervalType(-0.2,0.2); bx1[1]=ExactIntervalType(-0.1,0.10);
    ExactBoxType bx2(2); bx2[0]=ExactIntervalType(0.1,0.3); bx2[1]=ExactIntervalType(0.05,0.15);
    ExactBoxType bx3(2); bx3[0]=ExactIntervalType(0.2,0.4); bx3[1]=ExactIntervalType(0.10,0.25);
    ExactBoxType bx4(2); bx4[0]=ExactIntervalType(0.25,0.5); bx4[1]=ExactIntervalType(0.20,0.50);
    ExactBoxType bx5(2); bx5[0]=ExactIntervalType(0.4,0.8); bx5[1]=ExactIntervalType(0.40,1.1);
    double z1cdata[]={0.15,0.6}; double z1gdata[]={0.05,0.0,0.05, 0.0,0.05,0.05};
    Vector<FloatDP> z1c(2,z1cdata);
    Matrix<FloatDP> z1g(2,3,z1gdata);
    Zonotope z1(z1c,z1g);
    Vector<FloatDP> ts1c=z1c-Vector<FloatDP>(2,FloatDP(0.25));
    Matrix<FloatDP> ts1g=z1g;
    VectorAffineFunction afn1(ts1g,ts1c);
    TaylorConstrainedImageSet ts1(afn1,ExactBoxType::unit_box(3));

    VectorUserFunction<RadiusSquare> radius(Vector<FloatDP>(1u,0.5));
    ConstraintSet cs1(ExactBoxType(1u,ExactIntervalType(-1,0)),radius);

    std::cout << "Testing boxes.." << std::endl;
    TextPlot g("test_textplot-bx1.txt");
    g << bx1
      << bx2
      << bx3
      << bx4
      << bx5;
    g.close();

    g.open("test_textplot-bx2.txt");
    g.draw(bx1);
    g.draw(bx2);
    g.draw(bx5);
    g.close();

    std::cout << "Testing zonotopes and TaylorSets.." << std::endl;
    std::cerr << "WARNING: No output defined for Zonotopes and TaylorSets." << std::endl;
    //g.open("test_textplot-zts.txt");
    //g << z1
    //  << ts1;
    //g.close();

    std::cout << "Testing interpolated curves.." << std::endl;
    InterpolatedCurve cv(ExactPoint(2,0.0));
    for(Int i=1; i<=10; ++i) {
        ExactPoint pt(2); pt[0]=i/10.; pt[1]=sqr(pt[0]);
        cv.insert(i,pt);
    }

    g.open("test_textplot-cv.txt");
    g.draw(cv);
    g.close();

    std::cout << "Testing grid sets.." << std::endl;
    GridTreeSet gts(2);
    gts.adjoin_outer_approximation(ImageSet(bx1), 2);
    gts.adjoin_outer_approximation(ImageSet(bx2), 3);
    gts.adjoin_outer_approximation(ImageSet(bx3), 4);
    gts.recombine();

    std::cout << "outputting GridSets.." << std::endl;
    g.open("test_textplot-gts1.txt");
    g << gts;
    g.close();

    g.open("test_textplot-gts2.txt");
    draw(g,gts);
    g.close();

}
