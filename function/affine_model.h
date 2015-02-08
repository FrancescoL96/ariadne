/***************************************************************************
 *            affine_model.h
 *
 *  Copyright 2008-10  Pieter Collins
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

/*! \file affine_model.h
 *  \brief Affine models defined on the unit box
 */

#ifndef ARIADNE_AFFINE_MODEL_H
#define ARIADNE_AFFINE_MODEL_H

#include <cstdarg>
#include <iosfwd>
#include <iostream>

#include "utility/macros.h"
#include "utility/pointer.h"
#include "utility/declarations.h"

#include "numeric/numeric.h"
#include "algebra/vector.h"
#include "algebra/covector.h"
#include "algebra/matrix.h"

namespace Ariadne {

template<class X> class Affine;
typedef Affine<Float64> FloatAffine;
typedef Affine<ExactInterval> IntervalAffine;
typedef Affine<ApproximateNumber> ApproximateAffine;
typedef Affine<ValidatedNumber> ValidatedAffine;

template<class X> class AffineModel;
typedef AffineModel<Float64> FloatAffineModel;
typedef AffineModel<ApproximateNumber> ApproximateAffineModel;
typedef AffineModel<ValidatedNumber> ValidatedAffineModel;

template<class P, class F> class TaylorModel;
typedef TaylorModel<Approximate,Float64> ApproximateTaylorModel;
typedef TaylorModel<Validated,Float64> ValidatedTaylorModel;


AffineModel<ValidatedNumber> affine_model(const Affine<ValidatedNumber>& affine);
AffineModel<ValidatedNumber> affine_model(const Affine<EffectiveNumber>& affine);
AffineModel<ValidatedNumber> affine_model(const TaylorModel<Validated,Float64>& taylor_model);
AffineModel<ValidatedNumber> affine_model(const ExactBox& domain, const ValidatedScalarFunction& function);
Vector< AffineModel<ValidatedNumber> > affine_models(const Vector< TaylorModel<Validated,Float64> >& taylor_models);
Vector< AffineModel<ValidatedNumber> > affine_models(const ExactBox& domain, const ValidatedVectorFunction& function);

//! An affine expression \f$f:\R^n\rightarrow\R\f$ given by \f$f(x) \approx \sum_{i=0}^{n-1} a_i x_i + b\f$.
template<>
class AffineModel<ApproximateNumber>
{
  public:
    typedef ApproximateFloat64 CoefficientType;

    explicit AffineModel() : _c(), _g() { }
    explicit AffineModel(Nat n) : _c(0.0), _g(n,0.0) { }
    explicit AffineModel(const ApproximateNumber& c, const Covector<ApproximateNumber>& g) : _c(c), _g(g) { }
    explicit AffineModel(ApproximateNumber c, InitializerList<ApproximateNumber> g) : _c(c), _g(g) { }

    AffineModel<ApproximateNumber>& operator=(const ApproximateNumber& c) {
        this->_c=c; for(Nat i=0; i!=this->_g.size(); ++i) { this->_g[i]=0.0; } return *this; }
    ApproximateNumber& operator[](Nat i) { return this->_g[i]; }
    const ApproximateNumber& operator[](Nat i) const { return this->_g[i]; }
    static AffineModel<ApproximateNumber> constant(Nat n, const ApproximateNumber& c) {
        return AffineModel<ApproximateNumber>(c,Covector<ApproximateNumber>(n,0.0)); }
    static AffineModel<ApproximateNumber> variable(Nat n, Nat j) {
        return AffineModel<ApproximateNumber>(0.0,Covector<ApproximateNumber>::unit(n,j)); }

    const Covector<ApproximateNumber>& a() const { return this->_g; }
    const ApproximateNumber& b() const { return this->_c; }

    const Covector<ApproximateNumber>& gradient() const { return this->_g; }
    const ApproximateNumber& gradient(Nat i) const { return this->_g[i]; }
    const ApproximateNumber& value() const { return this->_c; }

    Void resize(Nat n) { this->_g.resize(n); }

    Nat argument_size() const { return this->_g.size(); }
  private:
    ApproximateNumber _c;
    Covector<ApproximateNumber> _g;
};

//! \relates AffineModel
//! \brief Negation of an affine model.
AffineModel<ApproximateNumber> operator-(const AffineModel<ApproximateNumber>& f);
//! \relates AffineModel
//! \brief Addition of two affine models.
AffineModel<ApproximateNumber> operator+(const AffineModel<ApproximateNumber>& f1, const AffineModel<ApproximateNumber>& f2);
//! \relates AffineModel
//! \brief Subtraction of two affine models.
AffineModel<ApproximateNumber> operator-(const AffineModel<ApproximateNumber>& f1, const AffineModel<ApproximateNumber>& f2);
//! \relates AffineModel
//! \brief Multiplication of two affine models.
AffineModel<ApproximateNumber> operator*(const AffineModel<ApproximateNumber>& f1, const AffineModel<ApproximateNumber>& f2);
//! \relates AffineModel
//! \brief Addition of a constant to an affine model.
AffineModel<ApproximateNumber>& operator+=(AffineModel<ApproximateNumber>& f1, const ApproximateNumber& c2);
//! \relates AffineModel
//! \brief Scalar multiplication of an affine model.
AffineModel<ApproximateNumber>& operator*=(AffineModel<ApproximateNumber>& f1, const ApproximateNumber& c2);

//! \relates AffineModel
//! \brief Write to an output stream.
OutputStream& operator<<(OutputStream& os, const AffineModel<ApproximateNumber>& f);


//! An affine expression \f$f:[-1,+1]^n\rightarrow\R\f$ given by \f$f(x)=\sum_{i=0}^{n-1} a_i x_i + b \pm e\f$.
template<>
class AffineModel<ValidatedNumber>
{
  public:
    typedef ExactFloat64 CoefficientType;
    typedef ErrorFloat64 ErrorType;

    explicit AffineModel() : _c(), _g() { }
    explicit AffineModel(Nat n) : _c(0.0), _g(n,ExactFloat64(0.0)), _e(0u) { }
    explicit AffineModel(const ExactFloat64& c, const Covector<ExactFloat64>& g, const ErrorFloat64& e) : _c(c), _g(g), _e(e) { }
    explicit AffineModel(ExactFloat64 c, InitializerList<ExactFloat64> g) : _c(c), _g(g), _e(0u) { }

    AffineModel<ValidatedNumber>& operator=(const ExactFloat64& c) {
        this->_c=c; for(Nat i=0; i!=this->_g.size(); ++i) { this->_g[i]=0; } this->_e=0u; return *this; }
    static AffineModel<ValidatedNumber> constant(Nat n, const ExactFloat64& c) {
        return AffineModel<ValidatedNumber>(c,Covector<ExactFloat64>(n,0),ErrorFloat64(0u)); }
    static AffineModel<ValidatedNumber> variable(Nat n, Nat j) {
        return AffineModel<ValidatedNumber>(0,Covector<ExactFloat64>::unit(n,j),0u); }


    const Covector<ExactFloat64>& a() const { return this->_g; }
    const ExactFloat64& b() const { return this->_c; }
    const ErrorFloat64& e() const { return this->_e; }

    ExactFloat64& operator[](Nat i) { return this->_g[i]; }
    const ExactFloat64& operator[](Nat i) const { return this->_g[i]; }
    const Covector<ExactFloat64>& gradient() const { return this->_g; }
    const ExactFloat64& gradient(Nat i) const { return this->_g[i]; }
    const ExactFloat64& value() const { return this->_c; }
    const ErrorFloat64& error() const { return this->_e; }

    Void set_value(const ExactFloat64& c) { _c=c; }
    Void set_gradient(Nat j, const ExactFloat64& g) { _g[j]=g; }
    Void set_error(const ErrorFloat64& e) { _e=e; }

    Void resize(Nat n) { this->_g.resize(n); }

    Nat argument_size() const { return this->_g.size(); }
    template<class X> X evaluate(const Vector<X>& v) const {
        X r=v.zero_element()+static_cast<X>(this->_c);
        for(Nat i=0; i!=this->_g.size(); ++i) {
            r+=X(this->_g[i])*v[i]; }
        r+=ValidatedNumber(-_e,+_e);
        return r;
    }

  private:
    ExactFloat64 _c;
    Covector<ExactFloat64> _g;
    ErrorFloat64 _e;
};

//! \relates AffineModel
//! \brief Negation of an affine model.
AffineModel<ValidatedNumber> operator-(const AffineModel<ValidatedNumber>& f);
//! \relates AffineModel
//! \brief Addition of two affine models.
AffineModel<ValidatedNumber> operator+(const AffineModel<ValidatedNumber>& f1, const AffineModel<ValidatedNumber>& f2);
//! \relates AffineModel
//! \brief Subtraction of two affine models.
AffineModel<ValidatedNumber> operator-(const AffineModel<ValidatedNumber>& f1, const AffineModel<ValidatedNumber>& f2);
//! \relates AffineModel
//! \brief Multiplication of two affine models.
AffineModel<ValidatedNumber> operator*(const AffineModel<ValidatedNumber>& f1, const AffineModel<ValidatedNumber>& f2);
//! \relates AffineModel
//! \brief Addition of a constant to an affine model.
AffineModel<ValidatedNumber>& operator+=(AffineModel<ValidatedNumber>& f1, const ValidatedNumber& c2);
//! \relates AffineModel
//! \brief Scalar multiplication of an affine model.
AffineModel<ValidatedNumber>& operator*=(AffineModel<ValidatedNumber>& f1, const ValidatedNumber& c2);

//! \relates AffineModel \brief Scalar addition to an affine model.
AffineModel<ValidatedNumber> operator+(const ValidatedNumber& c1, const AffineModel<ValidatedNumber>& f2);
AffineModel<ValidatedNumber> operator+(const AffineModel<ValidatedNumber>& f1, const ValidatedNumber& c2);
//! \relates AffineModel \brief Subtraction of an affine model from a scalar.
AffineModel<ValidatedNumber> operator-(const ValidatedNumber& c1, const AffineModel<ValidatedNumber>& f2);
//! \relates AffineModel \brief Subtraction of a scalar from an affine model.
AffineModel<ValidatedNumber> operator-(const AffineModel<ValidatedNumber>& f1, const ValidatedNumber& c2);
//! \relates AffineModel \brief Scalar multiplication of an affine model.
AffineModel<ValidatedNumber> operator*(const ValidatedNumber& c1, const AffineModel<ValidatedNumber>& f2);

//! \relates AffineModel
//! \brief Write to an output stream.
OutputStream& operator<<(OutputStream& os, const AffineModel<ValidatedNumber>& f);

//! \relates AffineModel
//! \brief Create from a Taylor model.
AffineModel<ValidatedNumber> affine_model(const TaylorModel<Validated,Float64>& tm);




} // namespace Ariadne

#endif /* ARIADNE_AFFINE_MODEL_H */
