/***************************************************************************
 *            float64.hpp
 *
 *  Copyright 2008-17  Pieter Collins
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

/*! \file float64.hpp
 *  \brief RawTag floating-point number class based on double-precision floats.
 */

#ifndef ARIADNE_FLOAT64_HPP
#define ARIADNE_FLOAT64_HPP

#include <iosfwd> // For std::floor std::ceil etc
#include <cmath> // For std::floor std::ceil etc
#include <algorithm> // For std::max, std::min
#include <limits> // For std::numeric_limits<double>

#include "utility/declarations.hpp"
#include "numeric/operators.hpp"
#include "numeric/rounding.hpp"
#include "numeric/sign.hpp"
#include "numeric/number.decl.hpp"
#include "numeric/float.decl.hpp"

namespace Ariadne {

class Float64;
typedef Float64 RawFloat64;

class Rational;
enum class Comparison : char;

class Precision64 {
    friend constexpr Precision64 max(Precision64, Precision64) { return Precision64(); }
    friend constexpr Precision64 min(Precision64, Precision64) { return Precision64(); }
    friend constexpr bool operator<=(Precision64, Precision64) { return true; }
    friend constexpr bool operator==(Precision64, Precision64) { return true; }
    friend OutputStream& operator<<(OutputStream& os, Precision64 dp) { return os << "Precision64()"; }
};
static const Precision64 double_precision = Precision64();
static const Precision64 pr64 = Precision64();
static const Precision64 dp = Precision64();

using RoundingMode64 = RoundingModeType;

// Correctly rounded functions
double sqr_rnd(double x);
double rec_rnd(double x);
double add_rnd(double x1, double x2);
double sub_rnd(double x1, double x2);
double mul_rnd(double x1, double x2);
double div_rnd(double x1, double x2);
double fma_rnd(double x1, double x2, double x3);
double pow_rnd(double x, int n);
double sqrt_rnd(double x);
double exp_rnd(double x);
double log_rnd(double x);
double sin_rnd(double x);
double cos_rnd(double x);
double tan_rnd(double x);
double atan_rnd(double x);


//! \ingroup NumericModule
//! \brief Floating point numbers (double precision) using approxiamate arithmetic.
//! \details
//! The \c %Float64 class represents floating-point numbers.
//! Unless otherwise mentioned, operations on floating-point numbers are performed approximately, with no guarantees
//! on the output.
//!
//! To implement <em>rounded arithmetic</em>, arithmetical operations of \c %Float64 can be performed with guaranteed rounding by
//! specifying \c _up and \c _down suffixes to arithmetical functions \c add, \c sub, \c mul and \c div.
//! Additionally, operations can be performed in the current <em>rounding mode</em> by using the \c _rnd suffix,
//! or with rounding reversed using the \c _opp suffix.
//! The \c _approx suffix is provided to specifically indicate that the operation is computed approximately.
//!
//! %Ariadne floating-point numbers can be constructed by conversion from built-in C++ types.
//! Note that the value of a built-in floating-point value may differ from the mathematical value of the literal.
//! For example, while <c>%Float64(3.25)</c> is represented exactly, <c>%Float64(3.3)</c> has a value of \f$3.2999999999999998224\ldots\f$.
//! \note In the future, the construction of a \c %Float64 from a string literal may be supported.
//! \sa Real, Float64Value, Float64Bounds, Float64UpperBound, Float64LowerBound, Float64Approximation
class Float64 {
  public:
    volatile double dbl;
  public:
    typedef RawTag Paradigm;
    typedef Float64 NumericType;
    typedef Precision64 PrecisionType;
    typedef RoundingMode64 RoundingModeType;
  public:
    static const RoundingModeType ROUND_TO_NEAREST = Ariadne::ROUND_TO_NEAREST;
    static const RoundingModeType ROUND_DOWNWARD = Ariadne::ROUND_DOWNWARD;
    static const RoundingModeType ROUND_UPWARD = Ariadne::ROUND_UPWARD;
    static const RoundingModeType ROUND_TOWARD_ZERO = Ariadne::ROUND_TOWARD_ZERO;

    static RoundingModeType get_rounding_mode();
    static Void set_rounding_mode(RoundingModeType);
    static Void set_rounding_downward();
    static Void set_rounding_upward();
    static Void set_rounding_to_nearest();
    static Void set_rounding_toward_zero();
  public:
    static Precision64 get_default_precision();
    Precision64 precision() const;
    Void set_precision(Precision64);
  public:
    static Float64 nan(Precision64 pr);
    static Float64 inf(Precision64 pr);
    static Float64 max(Precision64 pr);
    static Float64 eps(Precision64 pr);
    static Float64 min(Precision64 pr);
  public:
    //! \brief Default constructor creates an uninitialised number.
    Float64() : dbl() { }
    explicit Float64(Precision64) : dbl() { }
    //! \brief Convert from a built-in double-precision floating-point number.
    Float64(double x) : dbl(x) { }
    explicit Float64(double x, Precision64) : dbl(x) { }
    explicit Float64(Dyadic const& x, Precision64);
    //! \brief Copy constructor.
    Float64(const Float64& x) : dbl(x.dbl) { }

    //! \brief Construct from a double number using given rounding
    explicit Float64(double d, RoundingModeType rnd, PrecisionType pr);
    //! \brief Construct from a number using given rounding
    explicit Float64(Float64 d, RoundingModeType rnd, PrecisionType pr);
    //! \brief Construct from a rational number with given rounding
    explicit Float64(const Rational& q, RoundingModeType rnd, PrecisionType pr);
    //! \brief Convert to a dyadic number.
    explicit operator Dyadic () const;
    //! \brief Convert to a rational number.
    explicit operator Rational () const;
  public:
    Float64 const& raw() const { return *this; }
    //! \brief An approximation by a built-in double-precision floating-point number.
    double get_d() const { return this->dbl; }
  public:
    friend Bool is_nan(Float64 x) { return std::isnan(x.dbl); }
    friend Bool is_inf(Float64 x) { return std::isinf(x.dbl); }
    friend Bool is_finite(Float64 x) { return std::isfinite(x.dbl); }

    friend Float64 next(RoundUpward rnd, Float64 x) { return add(rnd,x,Float64::min(x.precision())); }
    friend Float64 next(RoundDownward rnd, Float64 x) { return sub(rnd,x,Float64::min(x.precision())); }

    friend Float64 floor(Float64 x);
    friend Float64 ceil(Float64 x);
    friend Float64 round(Float64 x);
  public:
    // Correctly rounded arithmetic
    friend Float64 nul(Float64 x) { return +0.0; }
    friend Float64 pos(Float64 x) { volatile double xv=x.dbl; return +xv; }
    friend Float64 neg(Float64 x) { volatile double xv=x.dbl; return -xv; }
    friend Float64 hlf(Float64 x) { volatile double xv=x.dbl; return xv/2; }
    friend Float64 sqr(Float64 x) { volatile double xv=x.dbl; return xv*xv; }
    friend Float64 rec(Float64 x) { volatile double xv=x.dbl; return 1.0/xv; }
    friend Float64 add(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v+x2v; return r; }
    friend Float64 sub(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v-x2v; return r; }
    friend Float64 mul(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v*x2v; return r; }
    friend Float64 div(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v/x2v; return r; }
    friend Float64 fma(Float64 x1, Float64 x2, Float64 x3) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double x3v=x3.dbl;
        volatile double r=x1v*x2v+x3v; return r; }
    friend Float64 pow(Float64 x, Int n) { return pow_rnd(x.dbl,n); }
    friend Float64 sqrt(Float64 x) { return sqrt_rnd(x.dbl); }
    friend Float64 exp(Float64 x) { return exp_rnd(x.dbl); }
    friend Float64 log(Float64 x) { return log_rnd(x.dbl); }
    friend Float64 sin(Float64 x) { return sin_rnd(x.dbl); }
    friend Float64 cos(Float64 x) { return cos_rnd(x.dbl); }
    friend Float64 tan(Float64 x) { return tan_rnd(x.dbl); }
    friend Float64 asin(Float64 x) { return std::asin(x.dbl); }
    friend Float64 acos(Float64 x) { return std::acos(x.dbl); }
    friend Float64 atan(Float64 x) { return atan_rnd(x.dbl); }
    static Float64 pi(PrecisionType pr);

    friend Float64 max(Float64 x1, Float64 x2) { return std::max(x1.dbl,x2.dbl); }
    friend Float64 min(Float64 x1, Float64 x2) { return std::min(x1.dbl,x2.dbl); }
    friend Float64 abs(Float64 x) { return std::fabs(x.dbl); }
    friend Float64 mag(Float64 x) { return std::fabs(x.dbl); }

    // Operators
    friend Float64 operator+(Float64 x) { volatile double xv=x.dbl; return +xv; }
    friend Float64 operator-(Float64 x) { volatile double xv=x.dbl; return -xv; }
    friend Float64 operator+(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v+x2v; return r; }
    friend Float64 operator-(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v-x2v; return r; }
    friend Float64 operator*(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v*x2v; return r; }
    friend Float64 operator/(Float64 x1, Float64 x2) { volatile double x1v = x1.dbl; volatile double x2v=x2.dbl; volatile double r=x1v/x2v; return r; }
    friend Float64& operator+=(Float64& x1, Float64 x2) { volatile double& x1v = x1.dbl; volatile double x2v=x2.dbl; x1v+=x2v; return x1; }
    friend Float64& operator-=(Float64& x1, Float64 x2) { volatile double& x1v = x1.dbl; volatile double x2v=x2.dbl; x1v-=x2v; return x1; }
    friend Float64& operator*=(Float64& x1, Float64 x2) { volatile double& x1v = x1.dbl; volatile double x2v=x2.dbl; x1v*=x2v; return x1; }
    friend Float64& operator/=(Float64& x1, Float64 x2) { volatile double& x1v = x1.dbl; volatile double x2v=x2.dbl; x1v/=x2v; return x1; }

    template<class OP> friend Float64 apply(OP op, RoundingMode64 rnd, Float64 x1, Float64 x2, Float64 x3) {
        auto old_rnd=Float64::get_rounding_mode(); Float64::set_rounding_mode(rnd);
        Float64 r=op(x1,x2,x3); Float64::set_rounding_mode(old_rnd); return r;
    }

    template<class OP> friend Float64 apply(OP op, RoundingMode64 rnd, Float64 x1, Float64 x2) {
        auto old_rnd=Float64::get_rounding_mode(); Float64::set_rounding_mode(rnd);
        Float64 r=op(x1,x2); Float64::set_rounding_mode(old_rnd); return r;
    }

    template<class OP> friend Float64 apply(OP op, RoundingMode64 rnd, Float64 x) {
        auto old_rnd=Float64::get_rounding_mode(); Float64::set_rounding_mode(rnd);
        Float64 r=op(x); Float64::set_rounding_mode(old_rnd); return r;
    }

    template<class OP> friend Float64 apply(OP op, RoundingMode64 rnd, Float64 x, Int n) {
        auto old_rnd=Float64::get_rounding_mode(); Float64::set_rounding_mode(rnd);
        Float64 r=op(x,n); Float64::set_rounding_mode(old_rnd); return r;
    }

    friend Float64 add(RoundingModeType rnd, Float64 x1, Float64 x2) { return apply(Add(),rnd,x1,x2); }
    friend Float64 sub(RoundingModeType rnd, Float64 x1, Float64 x2) { return apply(Sub(),rnd,x1,x2); }
    friend Float64 mul(RoundingModeType rnd, Float64 x1, Float64 x2) { return apply(Mul(),rnd,x1,x2); }
    friend Float64 div(RoundingModeType rnd, Float64 x1, Float64 x2) { return apply(Div(),rnd,x1,x2); }
    friend Float64 fma(RoundingModeType rnd, Float64 x1, Float64 x2, Float64 x3); // x1*x2+x3
    friend Float64 pow(RoundingModeType rnd, Float64 x, Int n) { return apply(Pow(),rnd,x,n); }
    friend Float64 sqr(RoundingModeType rnd, Float64 x) { return apply(Sqr(),rnd,x); }
    friend Float64 rec(RoundingModeType rnd, Float64 x) { return apply(Rec(),rnd,x); }
    friend Float64 sqrt(RoundingModeType rnd, Float64 x) { return apply(Sqrt(),rnd,x); }
    friend Float64 exp(RoundingModeType rnd, Float64 x) { return apply(Exp(),rnd,x); }
    friend Float64 log(RoundingModeType rnd, Float64 x) { return apply(Log(),rnd,x); }
    friend Float64 sin(RoundingModeType rnd, Float64 x) { return apply(Sin(),rnd,x); }
    friend Float64 cos(RoundingModeType rnd, Float64 x) { return apply(Cos(),rnd,x); }
    friend Float64 tan(RoundingModeType rnd, Float64 x) { return apply(Tan(),rnd,x); }
    friend Float64 atan(RoundingModeType rnd, Float64 x) { return apply(Atan(),rnd,x); }
    static Float64 pi(RoundingModeType rnd, PrecisionType pr);

    friend Float64 fma(RoundingModeType rnd, Float64 x1, Float64 x2, Float64 x3) {
        return apply(Fma(),rnd,x1,x2,x3); }
    //! \related Float64 \brief The average of two values, computed with nearest rounding. Also available with \c _ivl suffix.
    friend Float64 med(RoundingModeType rnd, Float64 x1, Float64 x2) {
        rounding_mode_t rounding_mode=get_rounding_mode(); set_rounding_mode(rnd);
        Float64 r=hlf(add(x1,x2)); set_rounding_mode(rounding_mode); return r; }
    //! \related Float64 \brief Half of the difference of two values, computed with upward rounding. Also available with \c _ivl suffix.
    friend Float64 rad(RoundingModeType rnd, Float64 x1, Float64 x2) {
        rounding_mode_t rounding_mode=get_rounding_mode(); set_rounding_mode(rnd);
        Float64 r=hlf(sub(x2,x1)); set_rounding_mode(rounding_mode); return r; }

    friend Float64 sqrt(RoundApprox, Float64 x) { return std::sqrt(x.dbl); }
    friend Float64 exp(RoundApprox, Float64 x) { return std::exp(x.dbl); }
    friend Float64 log(RoundApprox, Float64 x) { return std::log(x.dbl); }
    friend Float64 sin(RoundApprox, Float64 x) { return std::sin(x.dbl); }
    friend Float64 cos(RoundApprox, Float64 x) { return std::cos(x.dbl); }
    friend Float64 tan(RoundApprox, Float64 x) { return std::tan(x.dbl); }
    friend Float64 asin(RoundApprox, Float64 x) { return std::asin(x.dbl); }
    friend Float64 acos(RoundApprox, Float64 x) { return std::acos(x.dbl); }
    friend Float64 atan(RoundApprox, Float64 x) { return std::atan(x.dbl); }

    // Discontinuous integer-valued functions
    friend Float64 floor(Float64 x) { return std::floor(x.dbl); }
    friend Float64 ceil(Float64 x) { return std::ceil(x.dbl); }
    friend Float64 round(Float64 x) { return std::round(x.dbl); }

    friend Comparison cmp(Float64 x1, Float64  const& x2);
    friend Bool operator==(Float64 x1, Float64 x2) { return x1.dbl == x2.dbl; }
    friend Bool operator!=(Float64 x1, Float64 x2) { return x1.dbl != x2.dbl; }
    friend Bool operator<=(Float64 x1, Float64 x2) { return x1.dbl <= x2.dbl; }
    friend Bool operator>=(Float64 x1, Float64 x2) { return x1.dbl >= x2.dbl; }
    friend Bool operator< (Float64 x1, Float64 x2) { return x1.dbl <  x2.dbl; }
    friend Bool operator> (Float64 x1, Float64 x2) { return x1.dbl >  x2.dbl; }

    friend Comparison cmp(Float64 x1, Dbl x2);
    friend Bool operator==(Float64 const& x1, Dbl x2) { return x1.dbl == x2; }
    friend Bool operator!=(Float64 const& x1, Dbl x2) { return x1.dbl != x2; }
    friend Bool operator<=(Float64 const& x1, Dbl x2) { return x1.dbl <= x2; }
    friend Bool operator>=(Float64 const& x1, Dbl x2) { return x1.dbl >= x2; }
    friend Bool operator< (Float64 const& x1, Dbl x2) { return x1.dbl <  x2; }
    friend Bool operator> (Float64 const& x1, Dbl x2) { return x1.dbl >  x2; }
    friend Comparison cmp(Dbl x1, Float64 const& x2);
    friend Bool operator==(Dbl x1, Float64 const& x2) { return x1 == x2.dbl; }
    friend Bool operator!=(Dbl x1, Float64 const& x2) { return x1 != x2.dbl; }
    friend Bool operator<=(Dbl x1, Float64 const& x2) { return x1 <= x2.dbl; }
    friend Bool operator>=(Dbl x1, Float64 const& x2) { return x1 >= x2.dbl; }
    friend Bool operator< (Dbl x1, Float64 const& x2) { return x1 <  x2.dbl; }
    friend Bool operator> (Dbl x1, Float64 const& x2) { return x1 >  x2.dbl; }

    friend Comparison cmp(Float64 x1, Rational const& x2);
    friend Bool operator==(Float64 const& x1, Rational const& x2) { return cmp(x1,x2)==Comparison::EQUAL; }
    friend Bool operator!=(Float64 const& x1, Rational const& x2) { return cmp(x1,x2)!=Comparison::EQUAL; }
    friend Bool operator<=(Float64 const& x1, Rational const& x2) { return cmp(x1,x2)<=Comparison::EQUAL; }
    friend Bool operator>=(Float64 const& x1, Rational const& x2) { return cmp(x1,x2)>=Comparison::EQUAL; }
    friend Bool operator< (Float64 const& x1, Rational const& x2) { return cmp(x1,x2)< Comparison::EQUAL; }
    friend Bool operator> (Float64 const& x1, Rational const& x2) { return cmp(x1,x2)> Comparison::EQUAL; }
    friend Comparison cmp(Rational const& x1, Float64 const& x2);
    friend Bool operator==(Rational const& x1, Float64 const& x2) { return cmp(x1,x2)==Comparison::EQUAL; }
    friend Bool operator!=(Rational const& x1, Float64 const& x2) { return cmp(x1,x2)!=Comparison::EQUAL; }
    friend Bool operator<=(Rational const& x1, Float64 const& x2) { return cmp(x1,x2)<=Comparison::EQUAL; }
    friend Bool operator>=(Rational const& x1, Float64 const& x2) { return cmp(x1,x2)>=Comparison::EQUAL; }
    friend Bool operator< (Rational const& x1, Float64 const& x2) { return cmp(x1,x2)< Comparison::EQUAL; }
    friend Bool operator> (Rational const& x1, Float64 const& x2) { return cmp(x1,x2)> Comparison::EQUAL; }

    friend OutputStream& operator<<(OutputStream& os, Float64 const&);
    friend InputStream& operator>>(InputStream& is, Float64&);
    friend OutputStream& write(OutputStream& os, Float64 const& x, DecimalPlaces dgts, RoundingModeType rnd);

  private:
    // Opposite rounded arithmetic
    friend Float64 pos_opp(Float64 x) { volatile double t=-x.dbl; return -t; }
    friend Float64 neg_opp(Float64 x) { volatile double t=x.dbl; return -t; }
    friend Float64 sqr_opp(Float64 x) { volatile double t=-x.dbl; t=t*x.dbl; return -t; }
    friend Float64 rec_opp(Float64 x) { volatile double t=-1.0/(volatile double&)x.dbl; return -t; }
    friend Float64 add_opp(Float64 x, Float64 y) { volatile double t=-x.dbl; t=t-y.dbl; return -t; }
    friend Float64 sub_opp(Float64 x, Float64 y) { volatile double t=-x.dbl; t=t+y.dbl; return -t; }
    friend Float64 mul_opp(Float64 x, Float64 y) { volatile double t=-x.dbl; t=t*y.dbl; return -t; }
    friend Float64 div_opp(Float64 x, Float64 y) { volatile double t=x.dbl; t=t/y.dbl; return -t; }
    friend Float64 pow_opp(Float64 x, int n);
};

static const Float64 inf = std::numeric_limits<double>::infinity();


struct Float32 {
    float flt;
  public:
    explicit Float32(Float64 x, BuiltinRoundingModeType rnd) { set_rounding_mode(rnd); (volatile float&)flt = (volatile double&)x.dbl; }
    explicit operator Float64() const { return Float64((double)this->flt); }
};



} // namespace Ariadne

#endif
