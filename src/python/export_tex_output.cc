/***************************************************************************
 *            python/export_tex_output.cc
 *
 *  Copyright  2007  Pieter Collins
 *  Pieter.Collins@cwi.nl
 ****************************************************************************/

/*
 *  This program is free software; you can redistribute it and/or modify
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


#include "python/python_float.h"

#include "numeric/integer.h"
#include "numeric/rational.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/matrix.h"
#include "geometry/rectangle.h"

#include "output/texstream.h"

using namespace Ariadne;
using namespace Ariadne::Numeric;
using namespace Ariadne::LinearAlgebra;
using namespace Ariadne::Geometry;
using namespace Ariadne::Output;
using namespace Ariadne::Python;

#include <boost/python.hpp>
using namespace boost::python;

template<class T> inline texfstream& write(texfstream& txs, const T& t) { return static_cast<texfstream&>(txs << t); }

void export_tex_output()
{

  class_<texfstream, boost::noncopyable>("TexFile",init<>())
    .def("open",(void(texfstream::*)(const char*))&texfstream::open)
    .def("open",(void(texfstream::*)(const char*,const char*))&texfstream::open)
    .def("close",(void(texfstream::*)())&texfstream::close)
    .def("write",&write< char >,return_internal_reference<1>())
    .def("write",&write< char* >,return_internal_reference<1>())
    .def("write",&write< Integer >,return_internal_reference<1>())
    .def("write",&write< Rational >,return_internal_reference<1>())
    .def("write",&write< Float >,return_internal_reference<1>())
    .def("write",&write< Interval<Float> >,return_internal_reference<1>())
    .def("write",&write< Vector<Float> >,return_internal_reference<1>())
    .def("write",&write< Matrix<Float> >,return_internal_reference<1>())
    .def("write",&write< Rectangle<Float> >,return_internal_reference<1>())
  ;
  
}
