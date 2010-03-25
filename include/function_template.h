/***************************************************************************
 *            function.cc
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

#include "function.h"
#include "real.h"
#include "polynomial.h"
#include "differential.h"
#include "taylor_model.h"
#include "propagator.h"

namespace Ariadne {



// A wrapper for classes with non-static _compute and _compute_approx methods
template<class F>
class ScalarFunctionTemplate
    : public ScalarFunctionInterface
{
  private:
    template<class R, class A> void _base_compute(R& r, const A& a) const {
        static_cast<const F*>(this)->_compute(r,a); }
  protected:
    ScalarFunctionTemplate() { }
  public:
    virtual Float evaluate(const Vector<Float>& x) const {
        Float r; _base_compute(r,x); return r; }
    virtual Interval evaluate(const Vector<Interval>& x) const {
        Interval r; _base_compute(r,x); return r; }

    virtual TaylorModel evaluate(const Vector<TaylorModel>& x) const {
        TaylorModel r(TaylorModel(x[0].argument_size(),x[0].accuracy_ptr()));
        _base_compute(r,x); return r; }

    virtual Differential<Float> evaluate(const Vector< Differential<Float> >& x) const {
        Differential<Float> r(Differential<Float>(x[0].argument_size(),x[0].degree()));
        _base_compute(r,x); return r; }
    virtual Differential<Interval> evaluate(const Vector< Differential<Interval> >& x) const {
        Differential<Interval> r(Differential<Interval>(x[0].argument_size(),x[0].degree()));
        _base_compute(r,x); return r; }

    virtual Propagator<Interval> evaluate(const Vector< Propagator<Interval> >& x) const {
        Propagator<Interval> r; _base_compute(r,x); return r; }

    virtual std::ostream& repr(std::ostream& os) const {
        return this->write(os); }
};


// A wrapper for classes with non-static _compute and _compute_approx methods
template<class F>
class VectorFunctionTemplate
    : public VectorFunctionInterface
{
  private:
    template<class R, class A> void _base_compute(R& r, const A& a) const {
        static_cast<const F*>(this)->_compute(r,a); }
  protected:
    VectorFunctionTemplate() { }
  public:
    virtual Vector<Float> evaluate(const Vector<Float>& x) const {
        Vector<Float> r(this->result_size()); _base_compute(r,x); return r; }
    virtual Vector<Interval> evaluate(const Vector<Interval>& x) const {
        Vector<Interval> r(this->result_size()); _base_compute(r,x); return r; }

    virtual Vector<TaylorModel> evaluate(const Vector<TaylorModel>& x) const {
        Vector<TaylorModel> r(this->result_size(),TaylorModel(x[0].argument_size(),x[0].accuracy_ptr()));
        _base_compute(r,x); return r; }

    virtual Vector< Differential<Float> > evaluate(const Vector< Differential<Float> >& x) const {
        Vector< Differential<Float> > r(this->result_size(),Differential<Float>(x[0].argument_size(),x[0].degree()));
        _base_compute(r,x); return r; }
    virtual Vector< Differential<Interval> > evaluate(const Vector< Differential<Interval> >& x) const {
        Vector< Differential<Interval> > r(this->result_size(),Differential<Interval>(x[0].argument_size(),x[0].degree()));
        _base_compute(r,x); return r; }

    virtual Vector< Propagator<Interval> > evaluate(const Vector< Propagator<Interval> >& x) const {
        Vector< Propagator<Interval> > r(this->result_size(),Propagator<Interval>()); _base_compute(r,x); return r; }

};


} // namespace Ariadne