/***************************************************************************
 *      function_set.h
 *
 *  Copyright 2008  Pieter Collins
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

/*! \file function_set.h
 *  \brief Images and preimages of boxes in Euclidean space.
 */

#ifndef ARIADNE_FUNCTION_SET_H
#define ARIADNE_FUNCTION_SET_H

#include <iosfwd>

#include "boost/shared_ptr.hpp"
#include "macros.h"
#include "numeric.h"
#include "vector.h"
#include "set_interface.h"
#include "function.h"
#include "constraint.h"
#include "graphics_interface.h"

#include "box.h"
#include "taylor_function.h"

namespace Ariadne {

class Zonotope;
class Polyhedron;
class AffineSet;

class ImageSet;
class ConstraintSet;
class BoundedConstraintSet;
class ConstrainedImageSet;


//! \ingroup GeometryModule ExactSetSubModule
//! \brief A set defined as the image of a box under a continuous function.
//! The set is described as \f$S=h(D) = \{ h(s) \mid s \in D\}\f$ where \f$D\f$ is the domain and \f$h\f$ the function.
class ImageSet
    : public LocatedSetInterface
    , public DrawableInterface
{
    Vector<Interval> _domain;
    RealVectorFunction _function;
  public:
    //! \brief Default constructor constructs the singleton in \f$\R^0\f$.
    ImageSet();
    //! \brief Construct the image of \a dom under the identity function.
    ImageSet(const Vector<Interval>& dom);
    //! \brief Construct the image of \a dom under the function \a fn.
    ImageSet(const Vector<Interval>& dom, const RealVectorFunction& fn);
    //! \brief The box used to define the set.
    const Vector<Interval>& domain() const { return this->_domain; }
    //! \brief The function used to define the set.
    const RealVectorFunction& function() const { return this->_function; }
    //! \brief Equality operator. Compares functions by referential equality.
    bool operator==(const ImageSet& ims) const {
     return this->_domain==ims._domain && this->_function.raw_pointer()==ims._function.raw_pointer(); }

    ImageSet* clone() const;
    uint dimension() const;
    tribool empty() const;
    tribool disjoint(const Box&) const;
    tribool overlaps(const Box&) const;
    tribool inside(const Box&) const;
    Box bounding_box() const;
    void draw(CanvasInterface&) const;
    std::ostream& write(std::ostream&) const;
};



//! \ingroup GeometryModule ExactSetSubModule
//! \brief A set defined as the preimage of a box (the \em codomain) under a continuous function.
//! The set is described as \f$S=g^{-1}(C) = \{ x \mid g(x)\in C\}\f$ where \f$C\f$ is the codomain and \f$g\f$ the function.
//!
//! A constraint set is not Drawable since it cannot be bounded. To plot the set, first intersect with a bounding box.
class ConstraintSet
    : public RegularSetInterface
{
    Vector<Interval> _codomain;
    RealVectorFunction _function;
  public:
    //! \brief Construct the preimage of \a C under \a g.
    ConstraintSet(const Vector<Interval>& C, const RealVectorFunction& g);
    //! \brief Construct from a polyhedron.
    ConstraintSet(const Polyhedron& p);
    //! \brief Construct from a list of constraints.
    ConstraintSet(const List<NonlinearConstraint>& c);
    //! \brief The codomain of the set.
    const Vector<Interval>& codomain() const { return this->_codomain; }
    //! \brief The function used to define the set.
    const RealVectorFunction& function() const { return this->_function; };
    //! \brief The \a i<sup>th</sup> constraint \f$g_i(x)\in c_i\f$.
    NonlinearConstraint constraint(uint i) const { return NonlinearConstraint(this->_function[i],this->_codomain[i]); }
    //! \brief The number of constraints.
    uint number_of_constraints() const { return this->_codomain.size(); };

    ConstraintSet* clone() const;
    uint dimension() const;
    tribool disjoint(const Box&) const;
    tribool overlaps(const Box&) const;
    tribool covers(const Box&) const;
    std::ostream& write(std::ostream&) const;
    //! \brief Intersect with a box to create a bounded set.
    friend BoundedConstraintSet intersection(const ConstraintSet& set, const Box& bound);
    //! \brief Compute the preimage of the set $S=g^{-1}(C)\f$ under \f$h\$.
    //! The resulting set is the constraint set \f$h^{-1}(S)=(g\circ h)^{-1}(C)\f$.
    friend ConstraintSet preimage(const RealVectorFunction& h, const ConstraintSet& s) {
     return ConstraintSet(s.codomain(),compose(s.function(),h)); }
};


//! \ingroup GeometryModule ExactSetSubModule
//! \brief A set defined as the intersection of a box with preimage of a box (the \em codomain) under a continuous function.
//! The set is described as \f$S=D\cap g^{-1}(C) = \{ x\in D \mid g(x)\in C\}\f$ where \f$D\f$ is the domain, \f$C\f$ is the codomain and \f$g\f$ the function.
//!
//! A constraint set is Drawable since it cannot be bounded. To plot the set, first intersect with a bounding box.
class BoundedConstraintSet
    : public SetInterface
    , public DrawableInterface
{
    Vector<Interval> _domain;
    RealVectorFunction _function;
    Vector<Interval> _codomain;
  public:
    //! \brief Construct the preimage of \a C under \a g.
    BoundedConstraintSet(const Vector<Interval>& D, const RealVectorFunction& g, const Vector<Interval>& C);
    //! \brief Construct the restriction of \a D under the constraints \a c.
    BoundedConstraintSet(const Vector<Interval>& D, const List<NonlinearConstraint>& c);
    //! \brief The domain of the set.
    const Vector<Interval>& domain() const { return this->_domain; }
    //! \brief The codomain of the set.
    const Vector<Interval>& codomain() const { return this->_codomain; }
    //! \brief The function used to define the set.
    const RealVectorFunction& function() const { return this->_function; };
    //! \brief The \a i<sup>th</sup> constraint \f$g_i(x)\in c_i\f$.
    NonlinearConstraint constraint(uint i) const { return NonlinearConstraint(this->_function[i],this->_codomain[i]); }
    //! \brief The number of constraints.
    uint number_of_constraints() const { return this->_codomain.size(); };

    BoundedConstraintSet* clone() const;
    uint dimension() const;
    tribool disjoint(const Box&) const;
    tribool overlaps(const Box&) const;
    tribool covers(const Box&) const;
    tribool inside(const Box&) const;
    Box bounding_box() const;
    std::ostream& write(std::ostream&) const;
    void draw(CanvasInterface&) const;
};


//! \ingroup GeometryModule ExactSetSubModule
//! \brief A set defined as the image of the intersection of a box \f$D\f$ and a constraint set \f$g^{-1}(C)\f$ under a function \f$f\f$.
//! In other words, \f$S=f(D\cap g^{-1}(C))\f$.
class ConstrainedImageSet
    : public LocatedSetInterface, public DrawableInterface
{
    Box _domain;
    RealVectorFunction _function;
    List< NonlinearConstraint > _constraints;
  public:
    //! \brief Construct the set with zero-dimensional parameterisation in zero dimensions with no constraints.
    ConstrainedImageSet() : _domain(), _function() { }
    //! \brief Construct the box \a dom.
    ConstrainedImageSet(const Vector<Interval>& dom) : _domain(dom), _function(RealVectorFunction::identity(dom.size())) { }
    //! \brief Construct the image of \a dom under \a fn.
    ConstrainedImageSet(const Vector<Interval>& dom, const RealVectorFunction& fn) : _domain(dom), _function(fn) {
     ARIADNE_ASSERT_MSG(dom.size()==fn.argument_size(),"dom="<<dom<<", fn="<<fn); }
    //! \brief Construct the image of \a dom under \a fn, using constraint \a c.
    ConstrainedImageSet(const Vector<Interval>& dom, const RealVectorFunction& fn, const NonlinearConstraint& c) : _domain(dom), _function(fn), _constraints(1u,c) {
     ARIADNE_ASSERT_MSG(dom.size()==fn.argument_size(),"dom="<<dom<<", fn="<<fn);
     ARIADNE_ASSERT_MSG(dom.size()==c.function().argument_size(),"dom="<<dom<<", c="<<c);
    }
    //! \brief Construct the image of \a dom under \a fn, using constraints \a c.
    ConstrainedImageSet(const Vector<Interval>& dom, const RealVectorFunction& fn, const List<NonlinearConstraint>& c) : _domain(dom), _function(fn), _constraints(c) {
     ARIADNE_ASSERT_MSG(dom.size()==fn.argument_size(),"dom="<<dom<<", fn="<<fn); }
    //! \brief Construct the image of \a dom under \a fn.
    ConstrainedImageSet(const List<Interval>& dom, const List<RealScalarFunction>& fn) : _domain(dom), _function(fn) {
     ARIADNE_ASSERT_MSG(_domain.size()==_function.argument_size(),"dom="<<dom<<", fn="<<fn); }
    //! \brief Convert from a bounded constraint set.
    ConstrainedImageSet(const BoundedConstraintSet& set);
    //! \brief Convert from an image set.
    ConstrainedImageSet(const ImageSet& set);
    //! \brief The domain of the set.
    const Vector<Interval>& domain() const { return this->_domain; }
    //! \brief The function used to define the set.
    const RealVectorFunction& function() const { return this->_function; };
    //! \brief The function used to define the set.
    const List<NonlinearConstraint>& constraints() const { return this->_constraints; };
    //! \brief The number of parameters used to define the set, which equals the dimension of \f$D\f$.
    uint number_of_parameters() const { return this->_domain.size(); };
    //! \brief The number of constraints.
    uint number_of_constraints() const { return this->_constraints.size(); };
    //! \brief The \a i<sup>th</sup> constraint.
    NonlinearConstraint const& constraint(uint i) const { return this->_constraints[i]; }

    //! \brief Apply the function \f$h\f$ to obtain the set \f$h\circ f(D\cap g^{-1}(C))\f$.
    void apply(const RealVectorFunction& h) {
     this->_function=compose(h,this->_function);
    }

    //! \brief Introduce a new constraint of the form \f$g(y)\in [c_l,c_u]\f$.
    void new_parameter_constraint(const NonlinearConstraint& c) {
     ARIADNE_ASSERT_MSG(c.function().argument_size()==this->domain().size(),*this<<", "<<c);
     this->_constraints.append(c); }

    //! \brief Introduce a new constraint of the form \f$g(y)\in [c_l,c_u]\f$.
    void new_space_constraint(const NonlinearConstraint& c) {
     ARIADNE_ASSERT_MSG(c.function().argument_size()==this->_function.result_size(),*this<<", "<<c);
     this->_constraints.append(NonlinearConstraint(compose(c.function(),_function),c.bounds())); }

    //! \brief Introduce a new constraint of the form \f$g(y)\in [c_l,c_u]\f$.
    void new_parameter_constraint(const RealScalarFunction& g, const Interval& C) {
     ARIADNE_ASSERT_MSG(g.argument_size()==this->domain().size(),*this<<", "<<g<<" in "<<C);
     this->_constraints.append(NonlinearConstraint(g,C)); }

    //! \brief Introduce a new constraint of the form \f$g(x)\in [c_l,c_u]\f$.
    void new_space_constraint(const RealScalarFunction& g, const Interval& C) {
     ARIADNE_ASSERT_MSG(g.argument_size()==this->_function.result_size(),*this<<", "<<g<<" in "<<C);
     this->_constraints.append(NonlinearConstraint(compose(g,_function),C)); }

    ConstrainedImageSet* clone() const { return new ConstrainedImageSet(*this); }
    uint dimension() const { return this->_function.result_size(); }
    tribool inside(const Box& bx) const { return subset(this->bounding_box(),bx); }

    //! \brief A coarse over-approximation to the set. Computed by taking the interval evaluation \f$h(D)\f$.
    Box bounding_box() const;
    //! \brief Construct an affine over-approximation
    AffineSet affine_over_approximation() const;
    //! \brief Construct an affine approximation, with undefined accuracy.
    AffineSet affine_approximation() const;
    //! \brief Split into two pieces by subdividing along a coordinate direction.
    Pair<ConstrainedImageSet,ConstrainedImageSet> split() const;
    //! \brief Split into two pieces by subdividing along the \a j<sup>th</sup> coordinate direction.
    Pair<ConstrainedImageSet,ConstrainedImageSet> split(uint j) const;

    //! \brief Test if the set is disjoint from a box.
    tribool disjoint(const Box&) const;
    //! \brief Test if the set overlaps (intersects the interior of) a box.
    tribool overlaps(const Box&) const;
    //! \brief Adjoin an outer approximation to a paving.
    void adjoin_outer_approximation_to(GridTreeSet& paving, int depth) const;

    //! \brief Test if the set satisfies the state constraint at all points.
    tribool satisfies(const NonlinearConstraint& c) const;

    //! \brief Draw to a canvas.
    void draw(CanvasInterface&) const;
    //! \brief Write to an output stream.
    std::ostream& write(std::ostream&) const;

  private:
    void affine_adjoin_outer_approximation_to(GridTreeSet& paving, int depth) const;
    void subdivision_adjoin_outer_approximation_to(GridTreeSet& paving, int depth) const;
    void constraint_adjoin_outer_approximation_to(GridTreeSet& paving, int depth) const;
};


} // namespace Ariadne

#endif
