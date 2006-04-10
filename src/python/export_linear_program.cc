/***************************************************************************
 *            python/export_linear_program.cc
 *
 *  Copyright  2006  Alberto Casagrande, Pieter Collins
 *  casagrande@dimi.uniud.it, Pieter.Collins@cwi.nl
 ****************************************************************************/

/*
 *  This program is free software; you can rediself_ns::stribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is diself_ns::stributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Library General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */


#include "linear_algebra/vector.h"
#include "linear_algebra/linear_program.h"

#include "python/typedefs.h"
using namespace Ariadne;

#include <boost/python.hpp>
using namespace boost::python;

void export_linear_program() {
  typedef const FMatrix& (FLinearProgram::*MatrixConst) () const;

  class_<FLinearProgram>("FLinearProgram",init<FMatrix,FVector,FVector>())
    .def(init<FMatrix>())
    .def("solve",&FLinearProgram::solve)
    .def("is_satisfiable",&FLinearProgram::is_satisfiable)
    .def("optimizing_point",&FLinearProgram::optimizing_point)
    .def("optimal_value",&FLinearProgram::optimal_value)
    .def("tableau",MatrixConst(&FLinearProgram::tableau),return_value_policy<copy_const_reference>())
    .def(self_ns::str(self))    // __self_ns::str__
  ;
}
