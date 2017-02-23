/***************************************************************************
 *            hybrid_simulator.cpp
 *
 *  Copyright  2008-11  Pieter Collins
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

#include "function/functional.hpp"

#include "algebra/algebra.hpp"

#include "config.h"

#include "hybrid/hybrid_simulator.hpp"

#include "utility/array.hpp"
#include "utility/container.hpp"
#include "utility/tuple.hpp"
#include "utility/stlio.hpp"
#include "expression/valuation.hpp"
#include "expression/assignment.hpp"
#include "expression/space.hpp"

#include "function/function.hpp"
#include "function/formula.hpp"
#include "function/taylor_model.hpp"

#include "utility/logging.hpp"

#include "hybrid/hybrid_set.hpp"
#include "hybrid/hybrid_orbit.hpp"
#include "hybrid/hybrid_time.hpp"
#include "hybrid/hybrid_automaton_interface.hpp"


namespace Ariadne {

template<class T> class Orbit;

class DegenerateCrossingException { };

HybridSimulator::HybridSimulator()
    : _step_size(0.125)
{
}

Void HybridSimulator::set_step_size(double h)
{
    this->_step_size=h;
}


ExactPoint make_point(const HybridExactPoint& hpt, const RealSpace& spc) {
    if(hpt.space()==spc) { return hpt.point(); }
    Map<RealVariable,Float64Value> values=hpt.values();
    ExactPoint pt(spc.dimension());
    for(Nat i=0; i!=pt.size(); ++i) {
        pt[i]=values[spc.variable(i)];
    }
    return pt;
}

inline Float64Approximation evaluate(const EffectiveScalarFunction& f, const Vector<Float64Approximation>& x) { return f(x); }
inline Vector<Float64Approximation> evaluate(const EffectiveVectorFunction& f, const Vector<Float64Approximation>& x) { return f(x); }

Map<DiscreteEvent,EffectiveScalarFunction> guard_functions(const HybridAutomatonInterface& system, const DiscreteLocation& location) {
    Set<DiscreteEvent> events=system.events(location);
    Map<DiscreteEvent,EffectiveScalarFunction> guards;
    for(Set<DiscreteEvent>::ConstIterator iter=events.begin(); iter!=events.end(); ++iter) {
        guards.insert(*iter,system.guard_function(location,*iter));
    }
    return guards;
}

Orbit<HybridExactPoint>
HybridSimulator::orbit(const HybridAutomatonInterface& system, const HybridExactPoint& init_pt, const HybridTime& tmax) const
{
    Precision64 pr;
    HybridTime t(0.0,0);
    Float64Approximation h={this->_step_size,pr};

    DiscreteLocation location=init_pt.location();
    RealSpace space=system.continuous_state_space(location);
    ApproximatePoint point=make_point(init_pt,space);
    ApproximatePoint next_point;

    Orbit<HybridExactPoint> orbit(HybridExactPoint(location,space,cast_exact(point)));

    EffectiveVectorFunction dynamic=system.dynamic_function(location);
    Map<DiscreteEvent,EffectiveScalarFunction> guards=guard_functions(system,location);

    while(possibly(t<tmax)) {

        Bool enabled=false;
        DiscreteEvent event;
        for(Map<DiscreteEvent,EffectiveScalarFunction>::ConstIterator guard_iter=guards.begin(); guard_iter!=guards.end(); ++guard_iter) {
            if(probably(evaluate(guard_iter->second,point)>0)) {
                enabled=true;
                event=guard_iter->first;
                break;
            }
        }

        if(enabled) {
            DiscreteLocation target=system.target(location,event);
            EffectiveVectorFunction reset=system.reset_function(location,event);
            location=target;
            space=system.continuous_state_space(location);
            next_point=reset(point);

            dynamic=system.dynamic_function(location);
            guards=guard_functions(system,location);
            t._discrete_time+=1;
        } else {
            FloatApproximationVector k1,k2,k3,k4;
            ApproximatePoint pt1,pt2,pt3,pt4;

            ApproximatePoint const& pt=point;
            k1=evaluate(dynamic,pt);
            pt1=pt+h*k1;

            k2=evaluate(dynamic,pt1);
            pt2=pt1+(h/2)*k2;

            k3=evaluate(dynamic,pt2);
            pt3=pt1+(h/2)*k3;

            k4=evaluate(dynamic,pt3);

            next_point=pt+(h/6)*(k1+Float64Approximation(2.0)*(k2+k3)+k4);
            t._continuous_time += Real(Float64Value(h.raw()));
        }
        point=next_point;
        orbit.insert(t,HybridExactPoint(location,space,cast_exact(point)));
    }

    return orbit;

}



}  // namespace Ariadne
