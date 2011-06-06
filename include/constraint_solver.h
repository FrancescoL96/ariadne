/***************************************************************************
 *            constraint_solver.h
 *
 *  Copyright 2009  Pieter Collins
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

/*! \file constraint_solver.h
 *  \brief Class for solving systems of constraints over a box.
 */

#ifndef ARIADNE_CONSTRAINT_SOLVER_H
#define ARIADNE_CONSTRAINT_SOLVER_H

#include "logging.h"
#include "container.h"

#include "tribool.h"
#include "numeric.h"

namespace Ariadne {

class Real;
class Interval;
typedef tribool Tribool;
class Point;
class Box;
class GridTreeSet;

template<class X, class R> class NonlinearConstraint;
typedef NonlinearConstraint<Real,Float> RealNonlinearConstraint;
typedef NonlinearConstraint<Interval,Float> IntervalNonlinearConstraint;

template<class X> class Procedure;
typedef Procedure<Interval> IntervalProcedure;

template<class X> class ScalarFunctionInterface;
typedef ScalarFunctionInterface<Interval> IntervalScalarFunctionInterface;
template<class X> class ScalarFunction;
typedef ScalarFunction<Interval> IntervalScalarFunction;
template<class X> class VectorFunctionInterface;
typedef VectorFunctionInterface<Interval> IntervalVectorFunctionInterface;
template<class X> class VectorFunction;
typedef VectorFunction<Interval> IntervalVectorFunction;

class VectorTaylorFunction;

template<class X> struct FeasibilityState {
    X t;
    Vector<X> x;
    Vector<X> y;
    Vector<X> z;
};


//! \ingroup EvaluationModule OptimisationModule
//! \brief A class for finding solutions of systems of constraints of the form \f$g(y) \leq c\f$.
class ConstraintSolverInterface {
  public:
    //! \brief Test if the image of the box \a domain under the function \a function intersects \a codomain.
    virtual Pair<Tribool,Point> feasible(const Box& domain, const IntervalVectorFunction& function, const Box& codomain) const = 0;
    //! \brief Test if \a point is in \a domain and the image of \a point under the function \a function lies in \a codomain.
    virtual Tribool check_feasibility(const Box& domain, const IntervalVectorFunction& function, const Box& codomain, const Point& point) const = 0;
    //! \brief Try to reduce the size of the domain by propagating interval constraints. Returns \c true if the reduced domain is empty.
    virtual bool reduce(Box& domain, const IntervalVectorFunction& function, const Box& codomain) const = 0;

};



//! \ingroup OptimisationModule
//! \brief A class for finding solutions of systems of constraints of the form \f$g(y) \leq c\f$.
class ConstraintSolver
    : public ConstraintSolverInterface, public Loggable
{
    typedef Vector<Float> FloatVector;
    typedef Vector<Interval> IntervalVector;
  public:
    //! \brief Test if the image of the box \a domain under the function \a function intersects \a codomain.
    virtual Pair<Tribool,Point> feasible(const Box& domain, const IntervalVectorFunction& function, const Box& codomain) const;
    //! \brief Test if \a point is in \a domain and the image of \a point under the function \a function lies in \a codomain.
    virtual Tribool check_feasibility(const Box& domain, const IntervalVectorFunction& function, const Box& codomain, const Point& point) const;
    //! \brief Try to reduce the size of the domain by propagating interval constraints.
    virtual bool reduce(Box& domain, const IntervalVectorFunction& function, const Box& codomain) const;


    //! \brief Test if the constraints are solvable using a nonlinear feasibility test. Returns an approximate feasible point if the result is true. (Deprecated)
    virtual Pair<Tribool,Point> feasible(const Box& domain, const List<IntervalNonlinearConstraint>& constraints) const;
    //! \brief Try to reduce the size of the domain by propagating interval constraints. (Deprecated)
    virtual bool reduce(Box& domain, const List<IntervalNonlinearConstraint>& constraints) const;

    //! \brief Try to enforce hull consistency by propagating several interval constraints at once.
    //! This method is sharp if each variable occurs at most once in the constraint.
    bool hull_reduce(Box& bx, const IntervalVectorFunctionInterface& function, const IntervalVector& codomain) const;
    bool hull_reduce(Box& bx, const Vector<IntervalProcedure>& procedure, const IntervalVector& codomain) const;
    //! \brief Try to enforce hull consistency by propagating an interval constraint.
    //! This method is sharp if each variable occurs at most once in the constraint.
    bool hull_reduce(Box& bx, const IntervalScalarFunctionInterface& function, const Interval& codomain) const;
    bool hull_reduce(Box& bx, const IntervalProcedure& procedure, const Interval& codomain) const;

    //! \brief Reduce the \a domain by testing intersection of \a multipliers inner product \a function(\a domain)
    //! with \a multipliers innner product \a codomain, centering at \a centre.
    //! Reduces \f$(\lambda\cdot f)(X) \cap (\lambda\cdot C)\f$, evaluating \f$g(x)=g(x^*)+Dg(X) (X-x^*)\f$.
    bool lyapunov_reduce(Box& domain, const VectorTaylorFunction& function, const IntervalVector& codomain,
                         FloatVector centre, FloatVector multpliers) const;
    //! \brief Try to enforce hull consistency by reducing a constraint with respect to one variable.
    bool box_reduce(Box& bx, const IntervalScalarFunctionInterface& function, const Interval&, uint j) const;
    //! \brief Try to enforce hull consistency by reducing an a monotone dimension.
    //! This method is sharp if each variable occurs at most once in the constraint.
    bool monotone_reduce(Box& bx, const IntervalScalarFunctionInterface& function, const Interval&, uint j) const;

    //! Split the domain into two pieces to help try to solve the constraints.
    Pair<Box,Box> split(const Box& domain, const IntervalVectorFunction& function, const IntervalVector& codomain) const;

    // Deprecated functions.
    bool hull_reduce(Box& bx, const IntervalNonlinearConstraint& constraint) const {
        return this->hull_reduce(bx,constraint.function(),constraint.bounds()); }
    bool box_reduce(Box& bx, const IntervalNonlinearConstraint& constraint, uint j) const {
        return this->box_reduce(bx,constraint.function(),constraint.bounds(),j); }
    bool monotone_reduce(Box& bx, const IntervalNonlinearConstraint& constraint, uint j) const {
        return this->monotone_reduce(bx,constraint.function(),constraint.bounds(),j); }

};


} //namespace Ariadne

#endif