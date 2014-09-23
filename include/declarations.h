/***************************************************************************
 *            declarations.h
 *
 *  Copyright 2011  Pieter Collins
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

/*! \file declarations.h
 *  \brief Forward declarations of types and classes.
 */

#ifndef ARIADNE_DECLARATIONS_H
#define ARIADNE_DECLARATIONS_H

#include <iosfwd>
#include <boost/logic/tribool.hpp>

namespace Ariadne {

//! Internal name for output stream.
typedef std::ostream OutputStream;

//! Internal name for builtin boolean type.
typedef bool Bool;
//! Internal name for three-valued logical type.
typedef boost::tribool Tribool;
//! Internal name for string objects.
typedef std::string String;
//! Internal name for builtin unsigned integers.
typedef unsigned int Nat;
//! Internal name for builtin integers.
typedef int Int;



// Numeric declarations
class Integer;
class Rational;
class Real;

class Float;
class ExactFloat;
class ValidatedFloat;
class UpperFloat;
class LowerFloat;
class ApproximateFloat;

// Deprecated as numeric type
class Interval;

typedef Float RawFloat;
typedef Float RawFloatType;
typedef Float ApproximateFloatType;
typedef Float LowerFloatType;
typedef Float UpperFloatType;
typedef Interval ValidatedFloatType;
typedef Float ExactFloatType;

typedef ApproximateFloatType ApproximateNumberType;
typedef LowerFloatType LowerNumberType;
typedef UpperFloatType UpperNumberType;
typedef ValidatedFloatType ValidatedNumberType;
typedef Real EffectiveNumberType;
typedef ExactFloatType ExactNumberType;

typedef UpperFloatType PositiveUpperNumberType;

typedef PositiveUpperNumberType NormType;
typedef PositiveUpperNumberType ErrorType;
typedef ExactNumberType CoefficientType;

typedef ApproximateNumberType ApproximateNormType;
typedef ApproximateNumberType ApproximateErrorType;
typedef ApproximateNumberType ApproximateCoefficientType;

typedef ApproximateNumberType ApproximateNumber;
typedef ValidatedNumberType ValidatedNumber;
typedef EffectiveNumberType EffectiveNumber;
typedef ExactNumberType ExactNumber;

//typedef ApproximateFloat ApproximateNumberType;
//typedef ValidatedFloat ValidatedNumberType;
//typedef Real EffectiveNumberType;

// Information level declarations
struct ExactTag { };
typedef EffectiveNumberType EffectiveTag;
typedef ValidatedNumberType ValidatedTag;
typedef ApproximateNumberType ApproximateTag;

template<class I> struct CanonicalNumberTypedef;
template<> struct CanonicalNumberTypedef<ExactTag> { typedef ExactNumberType Type; };
template<> struct CanonicalNumberTypedef<EffectiveTag> { typedef EffectiveNumberType Type; };
template<> struct CanonicalNumberTypedef<ValidatedTag> { typedef ValidatedNumberType Type; };
template<> struct CanonicalNumberTypedef<ApproximateTag> { typedef ApproximateNumberType Type; };
template<class I> using CanonicalNumberType = typename CanonicalNumberTypedef<I>::Type;

// Concrete class declarations
template<class X> class Vector;
template<class X> class Matrix;
template<class X> class Differential;
template<class X> class Vector;
template<class X> class Matrix;
template<class X> class Differential;
template<class X> class Vector<Differential<X>>;
template<class X> class Series;


template<class X> class AffineModel;
template<class X> class TaylorModel;
template<class X> class Formula;
template<class X> class Algebra;

class Interval;
class Box;

typedef Interval IntervalType;

typedef Vector<Float> FloatVector;
typedef Vector<Interval> IntervalVector;

typedef Vector<RawFloatType> RawFloatVectorType;
typedef Vector<ApproximateFloatType> ApproximateFloatVectorType;
typedef Vector<ValidatedFloatType> ValidatedFloatVectorType;
typedef Vector<ExactFloatType> ExactFloatVectorType;

typedef Vector<ApproximateNumberType> ApproximateVectorType;
typedef Vector<ValidatedNumberType> ValidatedVectorType;
typedef Vector<EffectiveNumberType> EffectiveVectorType;
typedef Vector<ExactNumberType> ExactVectorType;

typedef Vector<ApproximateNumberType> ApproximatePointType;
typedef Vector<ValidatedNumberType> ValidatedPointType;
typedef Vector<EffectiveNumberType> EffectivePointType;
typedef Vector<ExactNumberType> ExactPointType;

// Function interface declarations
template<class X> class ScalarFunctionInterface;
template<class X> class VectorFunctionInterface;

typedef ScalarFunctionInterface<ApproximateTag> ApproximateScalarFunctionInterface;
typedef ScalarFunctionInterface<ValidatedTag> ValidatedScalarFunctionInterface;
typedef ScalarFunctionInterface<EffectiveTag> EffectiveScalarFunctionInterface;

typedef VectorFunctionInterface<ApproximateTag> ApproximateVectorFunctionInterface;
typedef VectorFunctionInterface<ValidatedTag> ValidatedVectorFunctionInterface;
typedef VectorFunctionInterface<EffectiveTag> EffectiveVectorFunctionInterface;

// Function declarations
template<class X> class ScalarFunction;
template<class X> class VectorFunction;

typedef ScalarFunction<ApproximateTag> ApproximateScalarFunction;
typedef ScalarFunction<ValidatedTag> ValidatedScalarFunction;
typedef ScalarFunction<EffectiveTag> EffectiveScalarFunction;
typedef ScalarFunction<Real> RealScalarFunction;

typedef VectorFunction<ApproximateTag> ApproximateVectorFunction;
typedef VectorFunction<ValidatedTag> ValidatedVectorFunction;
typedef VectorFunction<EffectiveTag> EffectiveVectorFunction;
typedef EffectiveVectorFunction RealVectorFunction;

// Deprecated typedefs
typedef ApproximateScalarFunction FloatScalarFunction;
typedef ValidatedScalarFunction IntervalScalarFunction;
typedef EffectiveScalarFunction RealScalarFunction;

typedef ApproximateVectorFunction FloatVectorFunction;
typedef ValidatedVectorFunction IntervalVectorFunction;
typedef EffectiveVectorFunction RealVectorFunction;

} // namespace Ariadne

#endif
