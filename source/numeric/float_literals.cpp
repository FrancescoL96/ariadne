/***************************************************************************
 *            numeric/float_literals.cpp
 *
 *  Copyright  2008-20  Pieter Collins
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

#include "float_error.hpp"
#include "float_ball.hpp"
#include "float_lower_bound.hpp"
#include "float_upper_bound.hpp"
#include "float_approximation.hpp"

namespace Ariadne {

FloatError<DoublePrecision> operator"" _error(long double lx) {
    double x=lx;
    assert(x==lx);
    return FloatError<DoublePrecision>(FloatDP(x));
}


FloatBall<DoublePrecision> operator"" _near(long double lx) {
    double x=lx;
    long double le=std::abs((long double)x-lx);
    double e=le;
    while(e<le) { e=e*(1+std::numeric_limits<double>::epsilon()); }

    return FloatBall<DoublePrecision>(x,e);
}


FloatUpperBound<DoublePrecision> operator"" _upper(long double lx) {
    static const double eps = std::numeric_limits<double>::epsilon();
    static const double min = std::numeric_limits<double>::min();
    double x=lx;
    if(x<lx) { x+=min; }

    while (x<lx) { x+=std::abs(x)*eps; }

    return FloatUpperBound<DoublePrecision>(x);
}


FloatLowerBound<DoublePrecision> operator"" _lower(long double lx) {
    static const double eps = std::numeric_limits<double>::epsilon();
    static const double min = std::numeric_limits<double>::min();
    double x=lx;
    if(x>lx) { x-=min; }


    while (x>lx) { x-=std::abs(x)*eps; }

    return FloatLowerBound<DoublePrecision>(x);
}


FloatApproximation<DoublePrecision> operator"" _approx(long double lx) {
    double x=lx;
    return FloatApproximation<DoublePrecision>(x);
}

} // namespace Ariadne
