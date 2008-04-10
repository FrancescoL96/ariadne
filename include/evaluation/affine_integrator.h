/***************************************************************************
 *            affine_integrator.h
 *
 *  Copyright  2006  Alberto Casagrande, Pieter Collins
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
 
/*! \file affine_integrator.h
 *  \brief Methods for integrating points and sets under a vector field.
 */

#ifndef ARIADNE_AFFINE_INTEGRATOR_H
#define ARIADNE_AFFINE_INTEGRATOR_H

#include "linear_algebra/declarations.h"
#include "geometry/declarations.h"
#include "system/declarations.h"
#include "evaluation/integrator_interface.h"

#include "system/affine_vector_field.h"

namespace Ariadne {
  namespace Evaluation {
   
    /*! \brief Compute an over-approximation to \f$\sum_{n=0}^{\infty} \frac{x^n}{(k+n)!}\f$. */
    template<class R> R gexp_up(const R& x, uint k);
  
    /*! \brief Compute an interval approximation to \f$\sum_{n=0}^{\infty} \frac{(tA)^n}{(k+n)!}b\f$. */
    template<class R> 
    LinearAlgebra::Vector< Numeric::Interval<R> > 
    gexp(const LinearAlgebra::Matrix< Numeric::Interval<R> >& A, 
         const LinearAlgebra::Vector< Numeric::Interval<R> >& b, 
         const Numeric::Interval<R>& t, const uint& k);
      
    /*! \brief Compute an interval approximation to \f$\sum_{n=0}^{\infty} \frac{(tA)^n}{(k+n)!}\f$. */
    template<class R> 
    LinearAlgebra::Matrix< Numeric::Interval<R> > 
    gexp(const LinearAlgebra::Matrix< Numeric::Interval<R> >& Ab, const Numeric::Interval<R>& t, const uint& k);
      
    template<class BS> class AffineIntegrator;
    
    /*!\ingroup Integrators
     * \brief An integrator based on using the exponential formula to integrate an affine vector field. 
     *  
     * The \f$C^1\f$-Affine algorithm is a Taylor method.
     */
    template<class R>
    class AffineIntegrator< Geometry::Zonotope<R> >
      : public IntegratorInterface< Geometry::Zonotope<R> >      
    {
      typedef Numeric::Interval<R> I;
     public:
      /*! \brief Constructor. */
      AffineIntegrator();

      /*! \brief Cloning operator. */
      virtual AffineIntegrator< Geometry::Zonotope<R> >* clone() const;

     public:

      /*! \brief Compute an integration time and a bounding box, given a bounding box for the intitial set, and a maximum allowable flow time. */
      virtual 
      std::pair< Numeric::Rational, Geometry::Box<R> >
      flow_bounds(const System::VectorField<R>& f, 
                  const Geometry::Box<R>& bx,
                  const Numeric::Rational& t) const; 

      /*! \brief Integrate a basic set for within a bounding set. */
      virtual Geometry::Point<I> flow_step(const System::VectorField<R>& vf,
                                           const Geometry::Point<I>& p,
                                           const Numeric::Rational& t,
                                           const Geometry::Box<R>& bb) const;
     
      /*! \brief Integrate a basic set for within a bounding set. */
      virtual LinearAlgebra::Matrix<I> flow_step_jacobian(const System::VectorField<R>& vf,
                                                          const Geometry::Point<I>& p,
                                                          const Numeric::Rational& r,
                                                          const Geometry::Box<R>& bb) const;
     

      virtual Geometry::Zonotope<R> 
      integration_step(const System::VectorField<R>& vector_field,
                       const Geometry::Zonotope<R>& initial_set,
                       const Numeric::Rational& step_size,
                       const Geometry::Box<R>& bounding_set) const;
      
      virtual Geometry::Zonotope<R> 
      reachability_step(const System::VectorField<R>& affine_vector_field,
                        const Geometry::Zonotope<R>& initial_set,
                        const Numeric::Rational& step_size,
                        const Geometry::Box<R>& bounding_set) const;

     public:
      /*! \brief Integrate \a initial point for time \a step_size. */
      Geometry::Point<I> 
      flow_step(const System::AffineVectorField<R>& affine_vector_field,
                const Geometry::Point<I>& initial_point,
                const Numeric::Rational& step_size) const;

      /*! \brief Comput the spacial Jacobian derivative at \a initial point for time \a step_size. */
      LinearAlgebra::Matrix<I> 
      flow_step_jacobian(const System::AffineVectorField<R>& vector_field,
                         const Geometry::Point<I>& affine_initial_point,
                         const Numeric::Rational& step_size) const;

      /*! \brief A \f$C^\infty\f$ algorithm for integrating forward a zonotope for a time up to time \a step_size. */
      Geometry::Zonotope<R> 
      integration_step(const System::AffineVectorField<R>& affine_vector_field,
                       const Geometry::Zonotope<R>& initial_set,
                       const Numeric::Rational& step_size) const;

      /*! \brief A \f$C^\infty\f$ algorithm for integrating forward a zonotope for a time up to time \a step_size. */
      Geometry::Zonotope<R> 
      reachability_step(const System::AffineVectorField<R>& affine_vector_field,
                        const Geometry::Zonotope<R>& initial_set,
                        const Numeric::Rational& step_size) const;

      /*! \brief Write to an output stream. */
      virtual std::ostream& write(std::ostream&) const;

     private:
      const System::AffineVectorField<R>* cast(const System::VectorField<R>*) const;
    };

     
  }
}

#endif /* ARIADNE_AFFINE_INTEGRATOR_H */
