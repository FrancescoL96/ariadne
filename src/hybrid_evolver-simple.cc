/***************************************************************************
 *            hybrid_evolver-constrained.cc
 *
 *  Copyright  2009  Pieter Collins
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

#include "numeric.h"
#include "vector.h"
#include "function.h"
#include "taylor_model.h"
#include "taylor_function.h"
#include "grid_set.h"
#include "hybrid_time.h"
#include "hybrid_automaton.h"
#include "hybrid_evolver-simple.h"
#include "orbit.h"

#include "integrator.h"
#include "solver.h"
#include <boost/concept_check.hpp>

namespace {

} // namespace

namespace Ariadne {

typedef Vector<Float> FloatVector;
typedef Vector<Interval> IntervalVector;

std::ostream& operator<<(std::ostream& os, const CrossingKind& crk) {
    switch(crk) {
        case DEGENERATE: os<<"degenerate"; break;
        case POSITIVE: os<<"positive"; break;
        case NEGATIVE: os<<"negative"; break;
        case INCREASING: os<<"increasing"; break;
        case DECREASING: os<<"decreasing"; break;
        case CONVEX: os<<"convex"; break;
        case CONCAVE: os<<"concave"; break;
        default: os << "unknown"; break;
    } return os;
}

std::ostream& operator<<(std::ostream& os, const StepKind& crk) {
    switch(crk) {
        case FULL_STEP: os<<"full"; break;
        case CREEP_STEP: os<<"creep"; break;
        case UNWIND_STEP: os<<"unwind"; break;
        case FINAL_STEP: os<<"final"; break;
        default: os << "unknown"; break;
    } return os;
}

std::ostream& operator<<(std::ostream& os, const TransitionData& transition) {
    return os << "kind="<<transition.event_kind<<", guard="<<transition.guard_function<<", "
                 "target="<<transition.target<<", reset="<<transition.reset_function;
}

std::ostream& operator<<(std::ostream& os, const TimingData& timing) {
    os << "step_kind="<<timing.step_kind<<", step_size="<<timing.step_size<<", "
       << "final_time="<<timing.final_time;
    if(timing.step_kind==CREEP_STEP) {
        os <<", evolution_time="<<timing.evolution_time;
    } else if(timing.step_kind==UNWIND_STEP) {
        os << ", finishing_time="<<timing.finishing_time;
    }
    return os;
}

bool is_blocking(EventKind evk) {
    switch(evk) {
        case INVARIANT: case PROGRESS: case URGENT: case IMPACT:
            return true;
        case PERMISSIVE:
            return false;
        default:
            ARIADNE_FAIL_MSG("EventKind "<<evk<<" not recognised by is_blocking(...) predicate.");
    }
}

bool is_activating(EventKind evk) {
    switch(evk) {
        case PERMISSIVE: case URGENT: case IMPACT:
            return true;
        case INVARIANT: case PROGRESS:
            return false;
        default:
            ARIADNE_FAIL_MSG("EventKind "<<evk<<" not recognised by is_activating(...) predicate.");
    }
}

Set<DiscreteEvent> blocking_events(const Map<DiscreteEvent,TransitionData>& transitions) {
    Set<DiscreteEvent> events;
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        if(is_blocking(transition_iter->second.event_kind)) {
            events.insert(transition_iter->first);
        }
    }
    return events;
}

Set<DiscreteEvent> activating_events(const Map<DiscreteEvent,TransitionData>& transitions) {
    Set<DiscreteEvent> events;
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        if(is_activating(transition_iter->second.event_kind)) {
            events.insert(transition_iter->first);
        }
    }
    return events;
}





Orbit<HybridEnclosure>
HybridEvolverBase::
orbit(const HybridAutomatonInterface& system,
      const HybridEnclosure& initial,
      const HybridTime& time,
      Semantics semantics) const
{
    ARIADNE_LOG(2,"\nHybridEvolverBase::orbit(...): verbosity="<<verbosity<<"\n");

    EvolutionData evolution_data;
    evolution_data.working_sets.push_back(HybridEnclosure(initial));
    while(!evolution_data.working_sets.empty()) {
        this->_upper_evolution_flow(evolution_data,system,time);
    }
    ARIADNE_ASSERT(evolution_data.working_sets.empty());
    ARIADNE_ASSERT(evolution_data.starting_sets.empty());

    Orbit<HybridEnclosure> orbit(initial);
    orbit.adjoin_intermediate(ListSet<HybridEnclosure>(evolution_data.intermediate_sets));
    orbit.adjoin_reach(evolution_data.reach_sets);
    orbit.adjoin_final(evolution_data.evolve_sets);
    return orbit;
}


HybridEvolverBase::HybridEvolverBase()
    : _parameters(new EvolutionParametersType())
{ }

HybridEvolverBase::HybridEvolverBase(const EvolutionParametersType& parameters)
    : _parameters(new EvolutionParametersType(parameters))
{ }

void
HybridEvolverBase::
_evolution(ListSet<HybridEnclosure>& final,
           ListSet<HybridEnclosure>& reachable,
           ListSet<HybridEnclosure>& intermediate,
           HybridAutomatonInterface const& system,
           HybridEnclosure const& initial_set,
           HybridTime const& maximum_time,
           Semantics semantics,
           bool reach) const
{
    EvolutionData evolution_data;

    evolution_data.working_sets.push_back(HybridEnclosure(initial_set));

    while(!evolution_data.working_sets.empty()) {
        this->_upper_evolution_flow(evolution_data,system,maximum_time);
    }

    final=evolution_data.evolve_sets;
    reachable=evolution_data.reach_sets;
    intermediate=evolution_data.intermediate_sets;
}

void
HybridEvolverBase::
_log_summary(uint ws, uint rs, HybridEnclosure const& starting_set) const
{
    Box starting_bounding_box=starting_set.space_bounding_box();
    Interval starting_time_range=starting_set.time_range();
    ARIADNE_LOG(1,"\r"
            <<"#w="<<std::setw(4)<<std::left<<ws+1u
            <<"#r="<<std::setw(4)<<std::left<<rs
            <<"#e="<<std::setw(3)<<std::left<<starting_set.previous_events().size()
            <<" t=["<<std::setw(7)<<std::left<<std::fixed<<starting_time_range.lower()
            <<","<<std::setw(7)<<std::left<<std::fixed<<starting_time_range.upper()<<"]"
            <<" #p="<<std::setw(2)<<std::left<<starting_set.number_of_parameters()
            <<" #c="<<std::setw(2)<<std::left<<starting_set.number_of_constraints()
            <<" r="<<std::setw(7)<<starting_bounding_box.radius()
            <<" c="<<starting_bounding_box.centre()
            <<" l="<<std::left<<starting_set.location()
            <<" e="<<starting_set.previous_events()
            <<"                      \n");
}

Map<DiscreteEvent,TransitionData>
HybridEvolverBase::
_extract_transitions(DiscreteLocation const& location,
                     HybridAutomatonInterface const& system) const
{
    Map<DiscreteEvent,TransitionData> transitions;
    Set<DiscreteEvent> events = system.urgent_events(location);
    for(Set<DiscreteEvent>::const_iterator event_iter=events.begin();
        event_iter!=events.end(); ++event_iter)
    {
        DiscreteLocation target=system.target(location,*event_iter);
        EventKind event_kind=system.event_kind(location,*event_iter);
        ScalarFunction guard_function=system.guard_function(location,*event_iter);
        VectorFunction reset_function=system.reset_function(location,*event_iter);
        TransitionData transition_data={event_kind,guard_function,target,reset_function};
        transitions.insert(*event_iter,transition_data);
    }
    return transitions;
}

void
HybridEvolverBase::
_process_initial_events(EvolutionData& evolution_data,
                        HybridEnclosure const& initial_set,
                        Map<DiscreteEvent,TransitionData> const& transitions) const
{
    ARIADNE_LOG(2,"HybridEvolverBase::_process_initial_events(...)\n");
    ARIADNE_ASSERT(evolution_data.starting_sets.empty());
    HybridEnclosure invariant_set=initial_set;

    // Apply restrictions due to invariants
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        DiscreteEvent event=transition_iter->first;
        TransitionData const & transition=transition_iter->second;
        if(transition.event_kind==INVARIANT) {
            if (possibly(initial_set.satisfies(transition.guard_function>=0))) {
                invariant_set.new_invariant(event,transition.guard_function);
            }
        }
    }

    // Set the flowable set, storing the invariant set as a base for jumps
    HybridEnclosure flowable_set = invariant_set;

    // Compute possibly initially active events
    Set<DiscreteEvent> events=transitions.keys();
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        DiscreteEvent event=transition_iter->first;
        TransitionData const & transition=transition_iter->second;
        if(transition.event_kind!=INVARIANT) {
            if(possibly(initial_set.satisfies(transition.guard_function>=0))) {
                if(transition.event_kind!=PERMISSIVE) {
                    HybridEnclosure immediate_jump_set=invariant_set;
                    immediate_jump_set.new_activation(event,transition.guard_function);
                    if(!definitely(immediate_jump_set.empty())) {
                        immediate_jump_set.apply_reset(event,transition.target,transition.reset_function);
                        ARIADNE_LOG(4,"immediate_jump_set="<<immediate_jump_set<<"\n");
                        evolution_data.intermediate_sets.append(immediate_jump_set);
                        evolution_data.working_sets.append(immediate_jump_set);
                    }
                }
                if(transition_iter->second.event_kind!=PROGRESS) {
                    flowable_set.new_invariant(event,transition.guard_function);
                }
            }
        }
    }
    evolution_data.starting_sets.append(flowable_set);
}

VectorIntervalFunction
HybridEvolverBase::
_compute_flow(VectorFunction dynamic,
              Box const& initial_box,
              const Float& maximum_step_size) const
{
    // Compute flow and actual time step size used
    TaylorIntegrator integrator(32,this->parameters().flow_accuracy);
    VectorIntervalFunction flow_model=integrator.flow(dynamic,initial_box,maximum_step_size);
    ARIADNE_LOG(6,"twosided_flow_model="<<flow_model<<"\n");
    IntervalVector flow_domain=flow_model.domain();
    Float step_size=flow_domain[flow_domain.size()-1u].upper();
    flow_domain[flow_domain.size()-1u]=Interval(0,step_size);
    flow_model=restrict(flow_model,flow_domain);
    ARIADNE_LOG(6,"flow_model="<<flow_model<<"\n");
    return flow_model;
}

Set<DiscreteEvent>
HybridEvolverBase::
_compute_active_events(VectorFunction const& dynamic,
                       Map<DiscreteEvent,ScalarFunction> const& guards,
                       VectorIntervalFunction const& flow,
                       HybridEnclosure const& starting_set) const
{
    Set<DiscreteEvent> events=guards.keys();
    Set<DiscreteEvent> active_events;
    HybridEnclosure reach_set=starting_set;
    IntervalVector flow_bounds=flow.range();
    reach_set.apply_flow(flow,flow.domain()[flow.domain().size()-1].upper());
    for(Set<DiscreteEvent>::iterator event_iter=events.begin(); event_iter!=events.end(); ++event_iter) {
        HybridEnclosure test_set=reach_set;
        test_set.new_activation(*event_iter,guards[*event_iter]);
        if(!definitely(test_set.empty())) {
            // Test direction of guard increase
            ScalarFunction flow_derivative = lie_derivative(guards[*event_iter],dynamic);
            Interval flow_derivative_range = flow_derivative(flow_bounds);
            if(flow_derivative_range.upper()>0.0) {
                // Compute crossing time
                active_events.insert(*event_iter);
            }
        }
    }
    return active_events;
}


Map<DiscreteEvent,CrossingData>
HybridEvolverBase::
_compute_crossings(Set<DiscreteEvent> const& active_events,
                   VectorFunction const& dynamic,
                   Map<DiscreteEvent,ScalarFunction> const& guards,
                   VectorIntervalFunction const& flow,
                   HybridEnclosure const& initial_set) const
{
    Map<DiscreteEvent,CrossingData> crossing_data;
    Box flow_bounds=flow.range();
    for(Set<DiscreteEvent>::const_iterator event_iter=active_events.begin();
        event_iter!=active_events.end(); ++event_iter)
    {
        const DiscreteEvent event=*event_iter;
        ScalarFunction const& guard=guards[event];
        ScalarFunction derivative=lie_derivative(guard,dynamic);
        Interval derivative_range=derivative.evaluate(flow_bounds);
        if(derivative_range.lower()>0.0) {
            ScalarIntervalFunction crossing_time;
            try {
                crossing_time=implicit(compose(guard,flow));
                crossing_data[event]=CrossingData(INCREASING,crossing_time);
            }
            catch(const ImplicitFunctionException& e) {
                ARIADNE_LOG(0,"Error in computing crossing time for event "<<*event_iter<<":\n  "<<e.what()<<"\n");
                crossing_data[event]=CrossingData(CONVEX);
            }
        } else if(derivative_range.upper()<0.0) {
            crossing_data[event]=CrossingData(DECREASING);
        } else {
            ScalarFunction second_derivative=lie_derivative(derivative,dynamic);
            Interval second_derivative_range=second_derivative.evaluate(flow_bounds);
            if(second_derivative_range.lower()>0.0) {
                crossing_data[event]=CrossingData(CONVEX);
            } else if(second_derivative_range.upper()<0.0) {
                try {
                    ScalarIntervalFunction critical_time=implicit(compose(derivative,flow));
                    crossing_data[event]=CrossingData(CONCAVE,critical_time);
                }
                catch(const ImplicitFunctionException& e) {
                    ARIADNE_LOG(0,"Error in computing crossing time for event "<<*event_iter<<":\n  "<<e.what()<<"\n");
                    crossing_data[event]=CrossingData(DEGENERATE);
                }
                    crossing_data[event]=CONCAVE;
            } else {
                crossing_data[event]=CrossingData(DEGENERATE);
            }
        }
    }
    return crossing_data;
}




TimingData
HybridEvolverBase::
_compute_timing(Set<DiscreteEvent> const& active_events,
                Real final_time,
                VectorIntervalFunction const& flow,
                Map<DiscreteEvent,CrossingData> const& crossings,
                HybridEnclosure const& initial_set) const
{
    TimingData result;
    result.step_size=flow.domain()[flow.domain().size()-1].upper();
    result.final_time=final_time;
    ScalarIntervalFunction remaining_time=result.final_time-initial_set.time_function();
    Interval remaining_time_range=remaining_time.range();
    // NOTE: The time function may be negative or greater than the final time
    // over part of the parameter domain.
    if(remaining_time_range.upper()<=result.step_size) {
        result.step_kind=FINAL_STEP;
        result.finishing_time=ScalarIntervalFunction::constant(initial_set.space_bounding_box(),final_time);
    } else if(remaining_time_range.lower()<=result.step_size) {
        result.step_kind=UNWIND_STEP;
        if(remaining_time_range.width()<result.step_size) {
            Float constant_finishing_time=result.final_time-remaining_time_range.lower()+result.step_size;
            result.finishing_time=ScalarIntervalFunction::constant(initial_set.space_bounding_box(),constant_finishing_time);
        } else {
            // FIXME: The finishing time may need to be adjusted
            result.finishing_time=0.5*(result.step_size+initial_set.time_function());
        }
    } else {
        result.step_kind=FULL_STEP;
        result.evolution_time=ScalarIntervalFunction::constant(initial_set.space_bounding_box(),result.step_size);
    }
    return result;
}



HybridEnclosure
HybridEvolverBase::
_compute_transverse_jump_set(HybridEnclosure const& starting_set,
                             DiscreteEvent const& event,
                             VectorIntervalFunction const& flow,
                             ScalarIntervalFunction const& evolution_time,
                             Float const& final_time,
                             Set<DiscreteEvent> const& active_events,
                             Map<DiscreteEvent,TransitionData> const& transitions,
                             Map<DiscreteEvent,CrossingData> const& crossing_data) const
{
    Map<DiscreteEvent,CrossingKind> crossing_kinds;
    Map<DiscreteEvent,ScalarIntervalFunction> crossing_times;
    HybridEnclosure jump_set=starting_set;
    switch(transitions[event].event_kind) {
        case permissive:
            jump_set.apply_flow(flow,evolution_time);
            jump_set.new_activation(event,transitions[event].guard_function);
            break;
        case urgent: case impact:
            jump_set.apply_flow_step(flow,crossing_times[event]);
            break;
        default:
            ARIADNE_FAIL_MSG("Invalid event kind "<<transitions[event].event_kind<<" for transition.");
    }

    // Apply blocking conditions for other active events
    for(Set<DiscreteEvent>::const_iterator other_event_iter=active_events.begin();
        other_event_iter!=active_events.end(); ++other_event_iter)
    {
        DiscreteEvent other_event=*other_event_iter;
        if(other_event!=event) {
            switch(crossing_kinds[*other_event_iter]) {
                case DEGENERATE:
                case CONCAVE:
                    ARIADNE_FAIL_MSG("Case of degenerate or concave other crossing not handled."); break;
                    // follow-through
                case INCREASING: case CONVEX:
                    jump_set.new_invariant(other_event,transitions[other_event].guard_function);
                    break;
                case NEGATIVE: case DECREASING:
                    // no additional constraints needed
                case POSITIVE:
                    // set should be empty
                    ARIADNE_FAIL_MSG("Case of no crossing due to positive guard not handled.");
                    break;
            }
        }
    }

    jump_set.apply_reset(event,transitions[event].target,transitions[event].reset_function);
    return jump_set;
}

// Apply time step without using any crossing functions
// Currently, do not handle CONCAVE and DEGENERATE constraints
void TransverseSimpleHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing_data,
                 Map<DiscreteEvent,CrossingData> const& crossings,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    ARIADNE_LOG(2,"TransverseSimpleHybridEvolver::_apply_time_step(...)\n");
    ARIADNE_LOG(4,"starting_time="<<starting_set.time_function().polynomial());
    //ARIADNE_LOG(4,"evolution_time="<<evolution_time.range()<<" final_time="<<final_time<<"\n");

    Set<DiscreteEvent> blocking_events=Ariadne::blocking_events(transitions);
    Set<DiscreteEvent> activating_events=Ariadne::activating_events(transitions);

    // Compute reach set
    HybridEnclosure reach_set=starting_set;
    reach_set.apply_flow(flow,timing_data.evolution_time);
    HybridEnclosure evolve_set=starting_set;
    evolve_set.apply_flow_step(flow,timing_data.evolution_time);

    // Compute reach and evolve sets
    for(Set<DiscreteEvent>::const_iterator event_iter=blocking_events.begin();
        event_iter!=blocking_events.end(); ++event_iter)
    {
        DiscreteEvent event=*event_iter;
        CrossingKind  crossing_kind = crossings[event].crossing_kind;
        TransitionData const & transition = transitions[event];
        switch(crossing_kind) {
            case INCREASING: case CONVEX:
                reach_set.new_invariant(event,transition.guard_function);
                evolve_set.new_invariant(event,transition.guard_function);
                break;
            case DECREASING:
                break;
            case DEGENERATE: case CONCAVE:
                //ARIADNE_FAIL_MESSAGE("Crossing kind "<<crossing_kind<<" not handled by HybridEvolverBase");
                break;
            default:
                //ARIADNE_FAIL_MESSAGE("Crossing kind "<<crossing_kind<<" not recognised by HybridEvolverBase");
                break;
        }
    }

    // Compute jump sets
    for(Set<DiscreteEvent>::const_iterator event_iter=activating_events.begin();
        event_iter!=activating_events.end(); ++event_iter)
    {
        DiscreteEvent event=*event_iter;
        TransitionData const & transition = transitions[event];
        HybridEnclosure jump_set=reach_set;
        switch(transitions[event].event_kind) {
            case PERMISSIVE:
                jump_set.new_activation(event,transition.guard_function);
                break;
            case URGENT: case IMPACT:
                jump_set.new_guard(event,transition.guard_function);
                jump_set.new_invariant(event,lie_derivative(transition.guard_function,dynamic));
                break;
            default:
                break;
        }
        if(!definitely(jump_set.empty())) {
            jump_set.apply_reset(event,transition.target,transition.reset_function);
            evolution_data.working_sets.append(jump_set);
        }
    }

    //FIXME: Sort this out...
    Real final_time = timing_data.final_time;

    HybridEnclosure final_set=reach_set;
    final_set.set_time(final_time);

    reach_set.bound_time(final_time);

    if(evolve_set.time_function().range().upper() > Float(final_time)) {
        evolve_set.bound_time(final_time);
    }

    if(!definitely(final_set.empty())) {
        evolution_data.evolve_sets.append(final_set);
    }

    evolution_data.starting_sets.append(evolve_set);
    evolution_data.reach_sets.append(reach_set);
}


void SimpleHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing_data,
                 Map<DiscreteEvent,CrossingData> const& crossing_data,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    ARIADNE_LOG(2,"SimpleHybridEvolver::_apply_time_step(...)\n");
    ARIADNE_LOG(4,"starting_time="<<starting_set.time_function().polynomial()<<"\n");
    //ARIADNE_LOG(4,"evolution_time="<<evolution_time.range()<<" final_time="<<final_time<<"\n");

    Float step_size=timing_data.step_size;
    ScalarIntervalFunction evolution_time=timing_data.evolution_time;
    Float final_time=timing_data.final_time;

    HybridEnclosure reach_set=starting_set;
    reach_set.apply_flow(flow,evolution_time);
    HybridEnclosure evolve_set=starting_set;
    evolve_set.apply_flow_step(flow,evolution_time);

    ScalarIntervalFunction critical_function;

    if(reach_set.time_function().range().upper() > Float(final_time)) {
        reach_set.bound_time(final_time);
    }

    Set<DiscreteEvent> transition_events;
    Set<DiscreteEvent> blocking_events;
    for(Set<DiscreteEvent>::const_iterator event_iter=blocking_events.begin();
        event_iter!=blocking_events.end(); ++event_iter)
    {
        switch(transitions[*event_iter].event_kind) {
            case PERMISSIVE:
                transition_events.insert(*event_iter);
                break;
            case URGENT: case IMPACT:
                transition_events.insert(*event_iter);
            case INVARIANT: case PROGRESS:
                blocking_events.insert(*event_iter);
        }
    }
    ARIADNE_LOG(6,"transition_events="<<transition_events<<"\n");
    ARIADNE_LOG(6,"blocking_events="<<blocking_events<<"\n");

    // Compute jump sets
    for(Set<DiscreteEvent>::const_iterator event_iter=transition_events.begin();
        event_iter!=transition_events.end(); ++event_iter)
    {
        DiscreteEvent event=*event_iter;
        TransitionData const & transition = transitions[event];
        HybridEnclosure jump_set=starting_set;
        switch(transitions[event].event_kind) {
            case PERMISSIVE:
                jump_set.apply_flow(flow,evolution_time);
                jump_set.new_activation(event,transition.guard_function);
                break;
            case URGENT: case IMPACT:
                switch(crossing_data[event].crossing_kind) {
                    case INCREASING:
                        jump_set.apply_flow_step(flow,crossing_data[event].crossing_time); break;
                    case CONVEX:
                        jump_set.apply_flow(flow,step_size);
                        jump_set.new_guard(event,transition.guard_function);
                        jump_set.new_invariant(event,lie_derivative(transition.guard_function,dynamic));
                        break;
                    default:
                        break;
                }
                break;
            default:
                ARIADNE_FAIL_MSG("Invalid event kind "<<transitions[event].event_kind<<" for transition.");
        }

        // Apply blocking conditions for other active events
        for(Set<DiscreteEvent>::const_iterator other_event_iter=blocking_events.begin();
            other_event_iter!=blocking_events.end(); ++other_event_iter)
        {
            DiscreteEvent other_event=*other_event_iter;
            if(other_event!=event) {
                switch(crossing_data[*other_event_iter].crossing_kind) {
                        case DEGENERATE:
                        case CONCAVE:
                        ARIADNE_FAIL_MSG("Case of concave or degenerate crossing not handled.");
                        // follow-through
                    case INCREASING: case CONVEX:
                        jump_set.new_invariant(other_event,transitions[other_event].guard_function);
                        break;
                    case NEGATIVE: case DECREASING:
                        // no additional constraints needed
                    case POSITIVE:
                        // set should be empty
                        ARIADNE_FAIL_MSG("Case of no crossing due to positive guard not handled.");
                        break;
                }
            }
        }

        if(jump_set.time_function().range().upper() > Float(final_time)) {
            jump_set.bound_time(final_time);
        }
        if(!definitely(jump_set.empty())) {
            jump_set.apply_reset(event,transitions[event].target,transitions[event].reset_function);
            evolution_data.working_sets.append(jump_set);
        }
    }

    for(Set<DiscreteEvent>::const_iterator event_iter=blocking_events.begin();
        event_iter!=blocking_events.end(); ++event_iter)
    {
        DiscreteEvent event=*event_iter;
        switch(crossing_data[*event_iter].crossing_kind) {
            case DEGENERATE:
                ARIADNE_FAIL_MSG("Case of degenerate crossing not handled.");
                break;
            case CONCAVE: {
                ARIADNE_LOG(2,"Concave crossing\n");
                reach_set.new_invariant(event,critical_function.function());
                evolve_set.new_invariant(event,critical_function.function());
                break;
            }
            case INCREASING: case CONVEX:
                // Only need constraints at final time
                reach_set.new_invariant(event,transitions[event].guard_function);
                evolve_set.new_invariant(event,transitions[event].guard_function);
                break;
            case NEGATIVE: case DECREASING:
                // no additional constraints needed
                break;
            case POSITIVE:
                // set should be empty
                ARIADNE_FAIL_MSG("Case of no crossing due to positive guard not handled.");
                break;
        }
    }

    HybridEnclosure final_set=reach_set;
    final_set.set_time(final_time);
    if(!definitely(final_set.empty())) {
        evolution_data.evolve_sets.append(final_set);
    }
    ARIADNE_LOG(6,"final_set="<<final_set<<"\n");

    reach_set.bound_time(final_time);
    evolution_data.reach_sets.append(reach_set);
    ARIADNE_LOG(6,"reach_set="<<reach_set<<"\n");


    if(evolve_set.time_function().range().upper() > Float(final_time)) {
        evolve_set.bound_time(final_time);
    }
    ARIADNE_LOG(6,"evolve_set="<<evolve_set<<"\n");

}


/*
HybridEnclosure
HybridEvolverBase::
_apply_final_time_step(HybridEnclosure const& starting_set,
                       Float const& final_time,
                       VectorTaylorFunction const& flow,
                       Set<DiscreteEvent> const& active_events,
                       Map<DiscreteEvent,ScalarFunction> const& guards) const
{
    // FIXME: Check for remaining time outside of flow time domain
    DiscreteEvent final_time_event("final_time");
    ScalarIntervalFunction final_time_function=ScalarIntervalFunction::constant(starting_set.parameter_domain(),final_time);

    HybridEnclosure final_set=starting_set;
    final_set.apply_flow_and_set_time(flow,final_time_function);

    for(Set<DiscreteEvent>::const_iterator event_iter=active_events.begin();
        event_iter!=active_events.end(); ++event_iter)
    {
        DiscreteEvent event=*event_iter;
        final_set.new_invariant(event,guards[event]);
    }

    return final_set;
}
*/


void
HybridEvolverBase::
_upper_evolution_flow(EvolutionData& evolution_data,
                      HybridAutomatonInterface const& system,
                      HybridTime const& maximum_hybrid_time) const
{
    ARIADNE_LOG(3,"HybridEvolverBase::_upper_evolution_flow\n");

    typedef Map<DiscreteEvent,ScalarFunction>::const_iterator constraint_iterator;
    typedef Set<DiscreteEvent>::const_iterator event_iterator;

    const Float final_time=maximum_hybrid_time.continuous_time();
    const uint maximum_steps=maximum_hybrid_time.discrete_time();

    // Routine check for emptiness
    if(evolution_data.working_sets.empty()) { return; }

    // Get the starting set for this round of evolution
    HybridEnclosure initial_set=evolution_data.working_sets.back(); evolution_data.working_sets.pop_back();
    ARIADNE_LOG(2,"initial_set="<<initial_set<<"\n\n");

    // Test if maximum number of steps has been reached
    if(initial_set.previous_events().size()>maximum_steps) {
        evolution_data.evolve_sets.append(initial_set); return;
    }

    if(initial_set.time_range().lower()>=final_time) {
        ARIADNE_WARN("starting_set.time_range()="<<initial_set.time_range()<<" which exceeds final time="<<final_time<<"\n");
        return;
    }

    // Extract starting location
    const DiscreteLocation location=initial_set.location();

    // Cache dynamic and constraint functions
    VectorFunction dynamic=system.dynamic_function(location);
    Map<DiscreteEvent,TransitionData> transitions = this->_extract_transitions(location,system);
    Set<DiscreteEvent> events = transitions.keys();

    ARIADNE_LOG(4,"\ndynamic="<<dynamic<<"\n");
    ARIADNE_LOG(4,"transitions="<<transitions<<"\n\n");

    // Process the initially active events; cut out active points to leave initial flowable set.
    this->_process_initial_events(evolution_data, initial_set,transitions);
    ARIADNE_ASSERT(evolution_data.starting_sets.size()==1);

    while(!evolution_data.starting_sets.empty()) {
        this->_upper_evolution_step(evolution_data,dynamic,transitions,final_time);
    }
}

void
HybridEvolverBase::
_upper_evolution_step(EvolutionData& evolution_data,
                      VectorFunction const& dynamic,
                      Map<DiscreteEvent,TransitionData> const& transitions,
                      Real const& final_time) const
{
    ARIADNE_LOG(3,"HybridEvolverBase::_upper_evolution_step\n");
    HybridEnclosure starting_set=evolution_data.starting_sets.back(); evolution_data.starting_sets.pop_back();

    ARIADNE_LOG(2,"starting_set="<<starting_set<<"\n");
    if(definitely(starting_set.empty())) {
        ARIADNE_LOG(4,"Empty starting_set "<<starting_set<<"\n");
        return;
    }

    if(starting_set.time_range().lower()>=static_cast<Float>(final_time)) {
        ARIADNE_WARN("starting_set.time_range()="<<starting_set.time_range()<<" which exceeds final time="<<final_time<<"\n");
        return;
    }

    if(verbosity==1) { _log_summary(evolution_data.working_sets.size(),evolution_data.reach_sets.size(),starting_set); }

    Map<DiscreteEvent,ScalarFunction> guard_functions;
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        guard_functions.insert(transition_iter->first,transition_iter->second.guard_function);
    }
    ARIADNE_LOG(4,"guards="<<guard_functions<<"\n");

    // Compute the bounding box of the enclosure
    const Box starting_bounding_box=starting_set.space_bounding_box();

    // Compute flow and actual time step size used
    const FlowFunctionPatch flow_model=this->_compute_flow(dynamic,starting_bounding_box,this->parameters().maximum_step_size);
    ARIADNE_LOG(4,"flow_model.domain()="<<flow_model.domain()<<" flow_model.range()="<<flow_model.range()<<"\n");

    // Compute possibly active urgent events with increasing guards, and crossing times
    Set<DiscreteEvent> active_events =
        this->_compute_active_events(dynamic,guard_functions,flow_model,starting_set);

    // Compute the kind of crossing (increasing, convex, etc);
    Map<DiscreteEvent,CrossingData> crossing_data =
        this->_compute_crossings(active_events,dynamic,guard_functions,flow_model,starting_set);

    // Compute end conditions for flow
    TimingData timing_data = this->_compute_timing(active_events,Real(final_time),flow_model,crossing_data,starting_set);

    // Apply the time step
    HybridEnclosure reach_set, evolve_set;
    this->_apply_time_step(evolution_data,starting_set,flow_model,timing_data,crossing_data,dynamic,transitions);
}




DiscreteEvent step_time_event("step_time");

void
VerySimpleHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing,
                 Map<DiscreteEvent,CrossingData> const& crossings,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    // Apply time step without computing transition crossing times
    ARIADNE_LOG(3,"VerySimpleHybridEvolver::_apply_time_step(...)\n");
    ARIADNE_ASSERT(crossings.size()<=1);

    HybridEnclosure reach_set=starting_set;
    HybridEnclosure evolve_set=starting_set;
    reach_set.apply_flow(flow,timing.step_size);
    evolve_set.apply_flow_step(flow,timing.step_size);
    if(!crossings.empty()) {
        DiscreteEvent event=crossings.begin()->first;
        HybridEnclosure jump_set=reach_set;
        jump_set.new_guard(event,transitions[event].guard_function);
        jump_set.apply_reset(event,transitions[event].target,transitions[event].reset_function);
        if(jump_set.time_range().upper()>timing.final_time) {
            jump_set.bound_time(timing.final_time);
        }
        if(!definitely(jump_set.empty())) {
            evolution_data.intermediate_sets.append(jump_set);
            evolution_data.working_sets.append(jump_set);
            ARIADNE_LOG(4,"  jump_set="<<jump_set<<"\n");
        }
        reach_set.new_invariant(event,transitions[event].guard_function);
        evolve_set.new_invariant(event,transitions[event].guard_function);
    }
    if(reach_set.time_function().range().upper()>=timing.final_time) {
        HybridEnclosure final_set=reach_set;
        final_set.set_time(timing.final_time);
        if(!definitely(final_set.empty())) {
            evolution_data.evolve_sets.append(final_set);
            ARIADNE_LOG(4,"  final_set="<<final_set<<"\n");
        }
        reach_set.bound_time(timing.final_time);
        evolve_set.bound_time(timing.final_time);
    }
    evolution_data.reach_sets.append(reach_set);
    ARIADNE_LOG(4,"  reach_set="<<reach_set<<"\n");
    if(!definitely(evolve_set.empty())) {
        evolution_data.intermediate_sets.append(evolve_set);
        evolution_data.starting_sets.append(evolve_set);
        ARIADNE_LOG(4,"  evolve_set="<<evolve_set<<"\n");
    }
    //{ char c; cin.get(c); }
}

void
DeterministicTransverseHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing,
                 Map<DiscreteEvent,CrossingData> const& crossings,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    ARIADNE_LOG(4,"\n");
    ARIADNE_LOG(3,"DeterministicTransverseHybridEvolver::_apply_time_step(...)\n");
    ARIADNE_LOG(4,timing<<"\n");
    ARIADNE_LOG(4,"starting_set="<<starting_set<<"\n");

    Set<DiscreteEvent> events=crossings.keys();
    typedef Set<DiscreteEvent>::const_iterator EventIterator;

    for(EventIterator event_iter=events.begin(); event_iter!=events.end(); ++event_iter) {
        const DiscreteEvent event=*event_iter;
        ARIADNE_LOG(4,"event="<<event<<", crossing_time="<<polynomial(crossings[event].crossing_time)<<"\n");
        HybridEnclosure jump_set=starting_set;
        const ScalarIntervalFunction& crossing_time=crossings[event].crossing_time;
        jump_set.new_invariant(event,-crossing_time);  // Ensure crossing time is positive
        for(EventIterator other_event_iter=events.begin(); other_event_iter!=events.end(); ++other_event_iter) {
            const DiscreteEvent other_event=*other_event_iter;
            if(other_event!=event) {
                jump_set.new_invariant(other_event,transitions[other_event].guard_function);
            }
        }
        ARIADNE_LOG(4,"  active_set="<<jump_set<<"\n");
        switch(timing.step_kind) {
            case FULL_STEP:
                jump_set.new_invariant(event,(crossing_time-timing.step_size).function());
                jump_set.apply_flow_step(flow,crossing_time);
                if(jump_set.time_function().range().upper()>timing.final_time) { jump_set.bound_time(timing.final_time); }
                break;
            case CREEP_STEP:
                jump_set.new_invariant(event,crossing_time-timing.evolution_time);
                jump_set.apply_flow_step(flow,crossing_time);
                if(jump_set.time_function().range().upper()>timing.final_time) { jump_set.bound_time(timing.final_time); }
                break;
            case UNWIND_STEP:
                jump_set.apply_flow_step(flow,crossing_time);
                jump_set.bound_time(timing.finishing_time);
                break;
            case FINAL_STEP:
                jump_set.apply_flow_step(flow,crossing_time);
                jump_set.bound_time(timing.final_time);
                break;
        }

        ARIADNE_LOG(4,"  active_set="<<jump_set<<"\n");
        jump_set.apply_reset(event,transitions[event].target,transitions[event].reset_function);
        ARIADNE_LOG(4,"  jump_set="<<jump_set<<"\n");
        if(!definitely(jump_set.empty())) {
            evolution_data.working_sets.append(jump_set);
            evolution_data.intermediate_sets.append(jump_set);
        }
    }

    HybridEnclosure evolve_set=starting_set;
    HybridEnclosure reach_set=starting_set;
    switch(timing.step_kind) {
        case FULL_STEP:
            evolve_set.apply_flow_step(flow,timing.step_size);
            reach_set.apply_flow(flow,timing.step_size);
            break;
        case CREEP_STEP:
            evolve_set.apply_flow_step(flow,timing.evolution_time);
            reach_set.apply_flow(flow,timing.evolution_time);
            break;
        case UNWIND_STEP:
            evolve_set.apply_flow_and_set_time(flow,timing.finishing_time);
            reach_set.apply_flow_and_bound_time(flow,timing.finishing_time);
            break;
        case FINAL_STEP:
            evolve_set.apply_flow_and_set_time(flow,timing.final_time);
            reach_set.apply_flow_and_bound_time(flow,timing.final_time);
            break;
        default:
            ARIADNE_FAIL_MSG("DeterministicTransverseHybridEvolver cannot handle flow step kind "<<timing.step_kind<<"\n");
    }
    HybridEnclosure final_set;

    for(EventIterator event_iter=events.begin(); event_iter!=events.end(); ++event_iter) {
        const DiscreteEvent event=*event_iter;
        evolve_set.new_invariant(event,transitions[event].guard_function);
        reach_set.new_invariant(event,transitions[event].guard_function);
    }

    if(timing.step_kind!=FINAL_STEP) {
        if(reach_set.time_function().range().upper()>timing.final_time) {
            HybridEnclosure final_set=reach_set;
            final_set.set_time(timing.final_time);
            if(!definitely(final_set.empty())) {
                evolution_data.evolve_sets.append(final_set);
            }
            ARIADNE_LOG(4,"  final_set="<<final_set<<"\n");
            reach_set.bound_time(timing.final_time);
            evolve_set.bound_time(timing.final_time);
        }
    }

    evolution_data.reach_sets.append(reach_set);
    ARIADNE_LOG(4,"  reach_set="<<reach_set<<"\n");

    switch(timing.step_kind) {
        case FINAL_STEP:
            // This is definitely the final step, so the evolve set is the final set
            ARIADNE_LOG(4,"  final_set="<<evolve_set<<"\n");
            if(!definitely(evolve_set.empty())) {
                evolution_data.evolve_sets.append(evolve_set);
            }
            break;
        case FULL_STEP: case CREEP_STEP: case UNWIND_STEP:
            ARIADNE_LOG(4,"  evolve_set="<<evolve_set<<"\n");
            if(!definitely(evolve_set.empty()) && evolve_set.time_function().range().lower()<timing.final_time) {
                evolution_data.starting_sets.append(evolve_set);
                evolution_data.intermediate_sets.append(evolve_set);
            } else {
                evolution_data.evolve_sets.append(evolve_set);
            }
    }
    ARIADNE_LOG(4,"\n");
}

ScalarIntervalFunction
DeterministicHybridEvolver::
_evolution_time(ScalarIntervalFunction const& maximum_evolution_time,
                Map<DiscreteEvent,ScalarIntervalFunction> const& crossing_times) const
{
    // Compute the evolution time for the current step given a maximum time compatible
    // with the flow, and crossing time functions for the transverse events
    Float step_size=maximum_evolution_time.range().upper();
    ScalarIntervalFunction evolution_time=maximum_evolution_time;
    for(Map<DiscreteEvent,ScalarIntervalFunction>::const_iterator time_iter=crossing_times.begin();
        time_iter!=crossing_times.end(); ++time_iter)
    {
        const ScalarIntervalFunction& crossing_time=time_iter->second;
        Float maximum_crossing_time=crossing_time.range().upper();
        maximum_crossing_time=max(maximum_crossing_time,2*step_size);
        ScalarTaylorFunction scaled_crossing_time=crossing_time/maximum_crossing_time;
        ScalarTaylorFunction creep_factor=scaled_crossing_time*(2-scaled_crossing_time);
        evolution_time=evolution_time*scaled_crossing_time;
    }
    return evolution_time;
}

void
DeterministicHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing_data,
                 Map<DiscreteEvent,CrossingData> const& crossings,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    ARIADNE_NOT_IMPLEMENTED;
}

/*
virtual void
TransverseHybridEvolver::
_apply_blocking(HybridEnclosure& set,
                const CrossingData& crossing,
                const TransitionData& transition,
                DiscreteEvent event)
{
    if(is_blocking(transition.event_kind)) {
        switch(crossing.kind) {
            case INCREASING: case CONVEX:
                set.new_invariant(transition.guard_function; break;
            case NEGATIVE: case DECREASING:
                break;
            case CONCAVE: case DEGENERATE:
                ARIADNE_FAIL_MESSAGE("TransverseHybridEvolver cannot handle "<<crossing.kind<<" crossing.";
            default:
                ARIADNE_FAIL_MSG("CrossingKind "<<crossing.kind<<" not recognised by TransverseHybridEvolver.");
        }
    }
}
*/

void
TransverseHybridEvolver::
_apply_time_step(EvolutionData& evolution_data,
                 HybridEnclosure const& starting_set,
                 VectorIntervalFunction const& flow,
                 TimingData const& timing_data,
                 Map<DiscreteEvent,CrossingData> const& crossings,
                 VectorFunction const& dynamic,
                 Map<DiscreteEvent,TransitionData> const& transitions) const
{
    HybridEnclosure reach_set=starting_set;
    for(Map<DiscreteEvent,TransitionData>::const_iterator transition_iter=transitions.begin();
        transition_iter!=transitions.end(); ++transition_iter)
    {
        DiscreteEvent event = transition_iter->first;
        CrossingData const& crossing = crossings[event];
        TransitionData const& transition = transition_iter->second;
        CrossingKind crossing_kind=crossing.crossing_kind;
        std::cerr<<crossing_kind;
        if(is_blocking(transition.event_kind)) {
            switch(crossing.crossing_kind) {
                case INCREASING: case CONVEX:
                    reach_set.new_invariant(event,transition.guard_function); break;
                case NEGATIVE: case DECREASING:
                    break;
                case CONCAVE: case DEGENERATE:
                    ARIADNE_FAIL_MSG("TransverseHybridEvolver cannot handle "<<crossing_kind<<" crossing.");
                default:
                    ARIADNE_FAIL_MSG("CrossingKind "<<crossing_kind<<" not recognised by TransverseHybridEvolver.");
            }
        }
    }


}



} // namespace Ariadne
