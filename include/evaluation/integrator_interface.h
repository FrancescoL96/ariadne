/***************************************************************************
 *            integrator_interface.h
 *
 *  Copyright  2006-8  Alberto Casagrande, Pieter Collins
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
 
/*! \file integrator_interface.h
 *  \brief Class for computing the image of a basic set under a map.
 */

#ifndef ARIADNE_INTEGRATOR_INTERFACE_H
#define ARIADNE_INTEGRATOR_INTERFACE_H

#include "base/types.h"
#include "base/declarations.h"
#include "linear_algebra/declarations.h"
#include "geometry/declarations.h"
#include "system/declarations.h"

namespace Ariadne {
  namespace Evaluation {

    /*! \ingroup EvaluatorInterfaces \ingroup Integrators
     *  \brief Interface for computing a step of the evolution of an enclosure set under a vector field.
     */
    template<class ES>
    class IntegratorInterface
    {
      typedef Numeric::Rational T;
      typedef typename ES::real_type R;
      typedef Numeric::Interval<R> I;
     public:
      //@{ 
      //! \name Constructors and cloning operations.
      /*! \brief Virtual destructor. */
      virtual ~IntegratorInterface() { }
      /*! \brief Make a dynamically-allocated copy. */
      virtual IntegratorInterface<ES>* clone() const = 0;
      //@}


      //@{ 
      //! \name Methods for integrating a vector field over an enclosure set.

      /*! \brief Compute an integration time and a bounding box. */
      virtual 
      std::pair< Numeric::Rational, Geometry::Box<R> >
      flow_bounds(const System::VectorField<R>& vf, 
                  const Geometry::Box<R>& s,
                  const Numeric::Rational& t) const = 0; 
      
      /*! \brief Compute the time \a t flow of an enclosure set \a s under a vector field \a vf, assuming \a bb is a bounding box for the flow. */
      virtual 
      ES
      integration_step(const System::VectorField<R>& vf, 
                       const ES& s,
                       const Numeric::Rational& t, 
                       const Geometry::Box<R>& bb) const = 0; 
      
      /*! \brief Compute the time \a t flow tube around an enclosure set \a s under a vector field \a vf, assuming \a bb is a bounding box for the flow. */
      virtual 
      ES
      reachability_step(const System::VectorField<R>& vf, 
                        const ES& s,
                        const Numeric::Rational& t, 
                        const Geometry::Box<R>& bb) const = 0;

      /*! \brief Compute the evolution of a basic set \a s under the vector field \a vf for times in the range [t1,t2], assuming \a bb is a bounding box for the flow. */
      virtual 
      ES
      evolution_step(const System::VectorField<R>& vf, 
                     const ES& s,
                     const Numeric::Rational& t1, 
                     const Numeric::Rational& t2, 
                     const Geometry::Box<R>& bb) const;
          
      //@}

      //@{ 
      //! \name Input/output operators. */
      /*! \brief Write to an output stream. */
      virtual std::ostream& write(std::ostream&) const = 0;
      //@}
    };

    template<class ES> std::ostream& operator<<(std::ostream& os, const IntegratorInterface<ES>& i);

  }
}

#include "integrator_interface.inline.h"

#endif /* ARIADNE_INTEGRATOR_INTERFACE_H */
