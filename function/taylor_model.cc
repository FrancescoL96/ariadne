/***************************************************************************
 *            taylor_model.cc
 *
 *  Copyright 2008-15  Pieter Collins
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
 *  GNU Library General Public License for more detai1ls.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#include "numeric/numeric.h"
#include "function/taylor_model.tcc"

namespace Ariadne {

template class Series<ValidatedFloat64>;

template class TaylorModel<Validated,Float64>;
template int instantiate_transcendental<TaylorModel<Validated,Float64>>();
template int instantiate_ordered<TaylorModel<Validated,Float64>>();

template class TaylorModel<Approximate,Float64>;
template int instantiate_transcendental<TaylorModel<Approximate,Float64>>();

} // namespace Ariadne
