/***************************************************************************
 *            float-approximate.h
 *
 *  Copyright 2008-14  Pieter Collins
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

/*! \file float-approximate.h
 *  \brief Approximate floating-point number class.
 */

#ifndef ARIADNE_FLOAT_APPROXIMATE_H
#define ARIADNE_FLOAT_APPROXIMATE_H

#include <iostream> // For std::floor std::ceil etc
#include <iomanip> // For std::setprecision
#include <cmath> // For std::floor std::ceil etc
#include <algorithm> // For std::max, std::min
#include <limits> // For std::numeric_limits<double>

#include "numeric/rounding.h"
#include "numeric/float.h"

namespace Ariadne {

class Real;
class Dyadic;
class Decimal;

class Float;

class ApproximateFloat;
class LowerFloat;
class UpperFloat;
class ValidatedFloat;
class ExactFloat;


//! \ingroup NumericModule
//! \brief Floating point numbers (double precision) using approxiamate arithmetic.
//! \details
//! The \c %Float class represents floating-point numbers.
//! Unless otherwise mentioned, operations on floating-point numbers are performed approximately, with no guarantees
//! on the output.
//!
//! To implement <em>interval arithmetic</em>, arithmetical operations of \c %Float can be performed with guaranteed rounding by
//! specifying \c _up and \c _down suffixes to arithmetical functions \c add, \c sub, \c mul and \c div.
//! Additionally, operations can be performed in the current <em>rounding mode</em> by using the \c _rnd suffix,
//! or with rounding reversed using the \c _opp suffix.
//! Operations can be specified to return an \c %ExactInterval answer by using the \c _ivl suffix.
//! The \c _approx suffix is provided to specifically indicate that the operation is computed approximately.
//!
//! %Ariadne floating-point numbers can be constructed by conversion from built-in C++ types.
//! Note that the value of a built-in floating-point value may differ from the mathematical value of the literal.
//! For example, while <c>%Float(3.25)</c> is represented exactly, <c>%Float(3.3)</c> has a value of \f$3.2999999999999998224\ldots\f$.
//! \note In the future, the construction of a \c %Float from a string literal may be supported.
//! \sa ExactInterval, Real, ExactFloat
class ApproximateFloat {
  public:
    static Nat output_precision;
    Float a;
  public:
    typedef Approximate Paradigm;
    typedef ApproximateFloat NumericType;
  public:
    //! \brief Default constructor creates an uninitialised number.
    ApproximateFloat() : a() { }
    //! \brief Convert from a built-in double-precision floating-point number.
    ApproximateFloat(double x) : a(x) { }
    //! \brief Copy constructor.
    explicit ApproximateFloat(const Float& x) : a(x) { }
    //! \brief Convert approximately from a decimal number.
    explicit ApproximateFloat(const Dyadic& d);
    //! \brief Convert approximately from a decimal number.
    explicit ApproximateFloat(const Decimal& d);

    //! \brief Construct from an integer.
    explicit ApproximateFloat(const Integer& z);
    //! \brief Construct from a rational number.
    explicit ApproximateFloat(const Rational& q);
    //! \brief Convert from a general real number by generating a representable approximation,
    //! not necessarily the nearest.
    explicit ApproximateFloat(const Real& r);

    //! \brief Convert from a floating-point number with an exact representation.
    ApproximateFloat(const ExactFloat& x);
    //! \brief Convert from a floating-point bounds on a number.
    ApproximateFloat(const ValidatedFloat& x);
    //! \brief Convert from a floating-point upper bound on a number.
    ApproximateFloat(const UpperFloat& x);
    //! \brief Convert from a floating-point lower bound on a number.
    ApproximateFloat(const LowerFloat& x);
    //! \brief Explicit conversion to extract raw data.
    explicit operator Float const& () const { return this->a; }
    //! \brief An approximation by a built-in double-precision floating-point number.
    Float const& raw() const { return this->a; }
    Float& raw() { return this->a; }
    //! \brief An approximation by a built-in double-precision floating-point number.
    double get_d() const { return this->a.get_d(); }
    static Void set_output_precision(Nat p) { output_precision=p; }
    ApproximateFloat pm(ApproximateFloat e) { return *this; }
};


inline OutputStream& operator<<(OutputStream& os, const ApproximateFloat& x) {
    return os << std::showpoint << std::setprecision(ApproximateFloat::output_precision) << x.a; }
inline InputStream& operator>>(InputStream& is, ApproximateFloat& x) {
    Float a; is >> a; x=ApproximateFloat(a); return is; }

template<class R, class A> inline R integer_cast(const A& a);
template<> inline Int integer_cast(const ApproximateFloat& x) { return static_cast<Int>(x.a.dbl); }
template<> inline Nat integer_cast(const ApproximateFloat& x) { return static_cast<Nat>(x.a.dbl); }

// Discontinuous integer-valued functions
//! \related ApproximateFloat \brief The next lowest integer, represented as a floating-point type.
inline ApproximateFloat floor(ApproximateFloat x) { return ApproximateFloat(floor(x.a)); }
//! \related ApproximateFloat \brief The next highest integer, represented as a floating-point type.
inline ApproximateFloat ceil(ApproximateFloat x) { return ApproximateFloat(ceil(x.a)); }

// Non-smooth operations
//! \related ApproximateFloat \brief The magnitude of a floating-point number. Equal to the absolute value.
inline ApproximateFloat mag(ApproximateFloat x) { return ApproximateFloat(abs(x.a)); }
//! \related ApproximateFloat \brief The absolute value of a floating-point number. Can be computed exactly.
inline ApproximateFloat abs(ApproximateFloat x) { return ApproximateFloat(abs(x.a)); }
//! \related ApproximateFloat \brief The maximum of two floating-point numbers. Can be computed exactly.
inline ApproximateFloat max(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(max(x.a,y.a)); }
//! \related ApproximateFloat \brief The minimum of two floating-point numbers. Can be computed exactly.
inline ApproximateFloat min(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(min(x.a,y.a)); }

// Standard arithmetic functions
//! \related ApproximateFloat \brief The unary plus function \c +x.
inline ApproximateFloat pos(ApproximateFloat x) { return ApproximateFloat(pos_exact(x.a)); }
//! \related ApproximateFloat \brief The unary negation function \c -x.
inline ApproximateFloat neg(ApproximateFloat x) { return ApproximateFloat(neg_exact(x.a)); }
//! \related ApproximateFloat \brief The halving function \c x/2.
inline ApproximateFloat half(ApproximateFloat x) { return ApproximateFloat(half_exact(x.a)); }
//! \related ApproximateFloat \brief The square function \c x*x.
inline ApproximateFloat sqr(ApproximateFloat x) { return ApproximateFloat(mul_near(x.a,x.a)); }
//! \related ApproximateFloat \brief The reciprocal function \c 1/x.
inline ApproximateFloat rec(ApproximateFloat x) { return ApproximateFloat(div_near(1.0,x.a)); }
//! \related ApproximateFloat \brief The binary addition function \c x+y.
inline ApproximateFloat add(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(add_near(x.a,y.a)); }
//! \related ApproximateFloat \brief The subtraction function \c x-y.
inline ApproximateFloat sub(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(sub_near(x.a,y.a)); }
//! \related ApproximateFloat \brief The binary multiplication function \c x*y.
inline ApproximateFloat mul(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(mul_near(x.a,y.a)); }
//! \related ApproximateFloat \brief The division function \c x/y.
inline ApproximateFloat div(ApproximateFloat x, ApproximateFloat y) { return ApproximateFloat(div_near(x.a,y.a)); }

//! \related ApproximateFloat \brief The positive integer power operator \c x^m.
//! Note that there is no power operator in C++, so the named version must be used. In Python, the power operator is \c x**m.
inline ApproximateFloat pow(ApproximateFloat x, Nat m) { return ApproximateFloat(pow_approx(x.a,m)); }
//! \related ApproximateFloat \brief The integer power operator \c x^n.
//! Note that there is no power operator in C++, so the named version must be used. In Python, the power operator is \c x**n.
inline ApproximateFloat pow(ApproximateFloat x, Int n) { return ApproximateFloat(pow_approx(x.a,n)); }

// Standard algebraic and transcendental functions
//! \related ApproximateFloat \brief The square-root function. Not guaranteed to be correctly rounded.
inline ApproximateFloat sqrt(ApproximateFloat x) { return ApproximateFloat(sqrt_approx(x.a)); }
//! \related ApproximateFloat \brief The exponential function. Not guaranteed to be correctly rounded.
inline ApproximateFloat exp(ApproximateFloat x) { return ApproximateFloat(exp_approx(x.a)); }
//! \related ApproximateFloat \brief The logarithm function. Not guaranteed to be correctly rounded.
inline ApproximateFloat log(ApproximateFloat x) { return ApproximateFloat(log_approx(x.a)); }


//! \related ApproximateFloat \brief The sine function. Not guaranteed to be correctly rounded. Also available with \c _rnd suffix.
inline ApproximateFloat sin(ApproximateFloat x) { return ApproximateFloat(sin_approx(x.a)); }
//! \related ApproximateFloat \brief The cosine function. Not guaranteed to be correctly rounded. Also available with \c _rnd suffix.
inline ApproximateFloat cos(ApproximateFloat x) { return ApproximateFloat(cos_approx(x.a)); }
//! \related ApproximateFloat \brief The tangent function. Not guaranteed to be correctly rounded. Also available with \c _rnd suffix.
inline ApproximateFloat tan(ApproximateFloat x) { return ApproximateFloat(tan_approx(x.a)); }
//! \related ApproximateFloat \brief The arcsine function. Not guaranteed to be correctly rounded.
inline ApproximateFloat asin(ApproximateFloat x) { return ApproximateFloat(asin_approx(x.a)); }
//! \related ApproximateFloat \brief The arccosine function. Not guaranteed to be correctly rounded.
inline ApproximateFloat acos(ApproximateFloat x) { return ApproximateFloat(acos_approx(x.a)); }
//! \related ApproximateFloat \brief The arctangent function. Not guaranteed to be correctly rounded.
inline ApproximateFloat atan(ApproximateFloat x) { return ApproximateFloat(atan_approx(x.a)); }


// Arithmetic operators
//! \related ApproximateFloat \brief Unary plus (identity) operator. Guaranteed to be exact.
inline ApproximateFloat operator+(const ApproximateFloat& x) { return pos(x); }
//! \related ApproximateFloat \brief Unary negation operator. Guaranteed to be exact.
inline ApproximateFloat operator-(const ApproximateFloat& x) { return neg(x); }
//! \related ApproximateFloat \brief The addition operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat operator+(const ApproximateFloat& x1, const ApproximateFloat& x2) { return add(x1,x2); }
//! \related ApproximateFloat \brief The subtraction operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat operator-(const ApproximateFloat& x1, const ApproximateFloat& x2) { return sub(x1,x2); }
//! \related ApproximateFloat \brief The multiplication operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat operator*(const ApproximateFloat& x1, const ApproximateFloat& x2) { return mul(x1,x2); }
//! \related ApproximateFloat \brief The division operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat operator/(const ApproximateFloat& x1, const ApproximateFloat& x2) { return div(x1,x2); }

//template<class N, typename std::enable_if<std::is_integral<N>::value,Int>::type=0>
//    inline ApproximateFloat operator/(N n1, const ApproximateFloat& x2) { return div(ApproximateFloat(n1),x2); };
//template<class N, typename std::enable_if<std::is_integral<N>::value,Int>::type=0>
//    inline ApproximateFloat operator/(const ApproximateFloat& x1, N n2) { return div(x1,ApproximateFloat(n2)); };

//! \related ApproximateFloat \brief The in-place addition operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat& operator+=(ApproximateFloat& x, const ApproximateFloat& y) { x.a.dbl+=y.a.dbl; return x; }
//! \related ApproximateFloat \brief The in-place subtraction operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat& operator-=(ApproximateFloat& x, const ApproximateFloat& y) { x.a.dbl-=y.a.dbl; return x; }
//! \related ApproximateFloat \brief The in-place multiplication operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat& operator*=(ApproximateFloat& x, const ApproximateFloat& y) { x.a.dbl*=y.a.dbl; return x; }
//! \related ApproximateFloat \brief The in-place division operator. Guaranteed to respect the current rounding mode.
inline ApproximateFloat& operator/=(ApproximateFloat& x, const ApproximateFloat& y) { x.a.dbl/=y.a.dbl; return x; }

// Comparison operators
//! \related ApproximateFloat \brief The equality operator.
inline Bool operator==(const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a==x2.a; }
//! \related ApproximateFloat \brief The inequality operator.
inline Bool operator!=(const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a!=x2.a; }
//! \related ApproximateFloat \brief The less-than-or-equal-to comparison operator.
inline Bool operator<=(const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a<=x2.a; }
//! \related ApproximateFloat \brief The greater-than-or-equal-to comparison operator.
inline Bool operator>=(const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a>=x2.a; }
//! \related ApproximateFloat \brief The less-than comparison operator.
inline Bool operator< (const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a< x2.a; }
//! \related ApproximateFloat \brief The greater-than comparison operator.
inline Bool operator> (const ApproximateFloat& x1, const ApproximateFloat& x2) { return x1.a> x2.a; }


} // namespace Ariadne

#endif
