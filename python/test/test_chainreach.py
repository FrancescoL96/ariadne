#!/usr/bin/python

##############################################################################
#            test_polyhedron.py
#
#  Copyright 2006  Pieter Collins <Pieter.Collins@cwi.nl>
##############################################################################

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

from ariadne.base import *
from ariadne.evaluation import *
from ariadne.geometry import *
from ariadne.linear_algebra import *
import sys


h=HenonMap(Dyadic(1.5),Dyadic(0.875))
gbb=Rectangle("[-4,4]x[-4,4]") # grid bounding box
g=FiniteGrid(gbb,512);
ir=Rectangle("[1.49,1.51]x[0.49,0.51]") # initial state
cb=Rectangle("[-4,4]x[-4,4]") # cutoff box
epsbb=Rectangle("[-4.1,4.1]x[-4.1,4.1]") # eps bounding box
i=RectangleListSet(ir)

cr=chainreach(h,i,g,cb)

eps=EpsPlot("cr.eps",epsbb)
eps.set_pen_colour("black")
eps.set_fill_colour("white")
eps.write(cb)
eps.set_line_style(0)
eps.set_fill_colour("green")
eps.write(cr)
eps.set_line_style(1)
eps.set_fill_colour("blue")
eps.write(ir)
eps.close()
sys.exit()
