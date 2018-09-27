/***************************************************************************
 *            function_model.cpp
 *
 *  Copyright 2011--17  Pieter Collins
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "../function/function_model.hpp"
#include "../function/function_model.tpl.hpp"
#include "../function/taylor_model.hpp"

#include "../algebra/algebra.hpp"

#include "../function/formula.hpp"

namespace Ariadne {

template class FunctionModelFactoryInterface<ValidatedTag>;

template class FunctionModel<ValidatedTag,IntervalDomainType,IntervalDomainType>;
template class FunctionModel<ValidatedTag,IntervalDomainType,BoxDomainType>;

template class FunctionModel<ValidatedTag,BoxDomainType,IntervalDomainType>;
template class FunctionModel<ValidatedTag,BoxDomainType,BoxDomainType>;

} // namespace Ariadne
