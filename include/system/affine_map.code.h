/***************************************************************************
 *            affine_map.code.h
 *
 *  Copyright  2005-6  Alberto Casagrande, Pieter Collins
 *  casagrande@dimi.uniud.it  Pieter.Collins@cwi.nl
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
 

#include "affine_map.h"

#include "linear_algebra/vector.h"
#include "linear_algebra/matrix.h"

#include "geometry/point.h"

#include "system/exceptions.h"



namespace Ariadne {
  namespace System {

    template<class R>
    Geometry::Point<typename AffineMap<R>::F>
    AffineMap<R>::image(const Geometry::Point<F>& pt) const
    {
      ARIADNE_CHECK_ARGUMENT_DIMENSION(*this,pt,"Point AffineMap::image(Point pt)");
      LinearAlgebra::Vector<F> image(this->A()*LinearAlgebra::Vector<F>(pt.position_vector())+this->b());
      return Geometry::Point<F>(image);
    }
    
    
    template<class R>
    LinearAlgebra::Matrix<typename AffineMap<R>::F>
    AffineMap<R>::jacobian(const Geometry::Point<F>& pt) const
    {
      return this->_a;
    }
    
    template<class R>
    std::ostream& 
    AffineMap<R>::write(std::ostream& os) const
    {
      return os << "AffineMap( A=" << this->A()
                << ", b=" << this->b() << " )";
    }
    


  }
}
