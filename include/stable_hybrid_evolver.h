/***************************************************************************
 *            stable_hybrid_evolver.h
 *
 *  Copyright  2007-9  Alberto Casagrande, Pieter Collins, Davide Bresolin
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
 
/*! \file stable_hybrid_evolver.h
 *  \brief The most stable evolver for hybrid systems. 
 */

#ifndef ARIADNE_STABLE_HYBRID_EVOLVER_H
#define ARIADNE_STABLE_HYBRID_EVOLVER_H

#include <string>
#include <vector>
#include <list>
#include <iostream>

#include <boost/smart_ptr.hpp>

#include "tuple.h"

#include "hybrid_set.h"

#include "hybrid_automaton.h"
#include "evolver_interface.h"
#include "evolver_base.h"
#include "evolution_parameters.h"

#include "logging.h"

namespace Ariadne {  

class TaylorModel;
class TaylorExpression;
class TaylorFunction;
class TaylorSet;
typedef std::pair<DiscreteState,TaylorSet> HybridTaylorSet;

template<class MDL> class CalculusInterface;

class EvolutionParameters;
class EvolutionProfiler;
template<class ES> class Orbit;

class HybridTime;
class HybridAutomaton;





/*! \brief A class for computing the evolution of a hybrid system. 
 *
 * The actual evolution steps are performed by the HybridEvolver class.
 */
class StableHybridEvolver
    : public EvolverBase< HybridAutomaton, HybridTaylorSet>
    , public Loggable
{
    typedef TaylorModel ModelType;
    typedef FunctionInterface FunctionType;
    typedef Vector<Interval> BoxType;
    typedef TaylorFunction FunctionModelType;
    typedef TaylorFunction MapModelType;
    typedef TaylorFunction FlowModelType;
    typedef TaylorExpression ConstraintModelType;
    typedef TaylorModel TimeModelType;
    typedef TaylorSet SetModelType;
    typedef TaylorSet TimedSetModelType;
  public:
    typedef ContinuousEvolutionParameters EvolutionParametersType;
    typedef HybridAutomaton::TimeType TimeType;
    typedef int IntegerType;
    typedef Float RealType;
    typedef std::vector<DiscreteEvent> EventListType;
    typedef HybridAutomaton SystemType;
    typedef TaylorSet ContinuousEnclosureType;
    typedef pair<DiscreteState,TaylorSet> HybridEnclosureType;
    typedef HybridEnclosureType EnclosureType;
    typedef pair< Interval , EnclosureType > TimedEnclosureType;
    typedef Orbit<EnclosureType> OrbitType;
    typedef ListSet<EnclosureType> EnclosureListType;
    typedef std::vector<TimedEnclosureType> TimedEnclosureListType;
    typedef Float ContinuousTimeType;
  public:
    
    //! \brief Default constructor.
    StableHybridEvolver();
  
    //! \brief Construct from parameters using a default integrator.
    StableHybridEvolver(const EvolutionParametersType& parameters);
  
    /*! \brief Make a dynamically-allocated copy. */
    StableHybridEvolver* clone() const { return new StableHybridEvolver(*this); }

    //@{
    //! \name Parameters controlling the evolution.
    //! \brief A reference to the parameters controlling the evolution.
    EvolutionParametersType& parameters() { return *this->_parameters; }
    const EvolutionParametersType& parameters() const { return *this->_parameters; }

    //@}
  

    //@{
    //! \name Evolution using abstract sets.
    //! \brief Compute an approximation to the orbit set using the given semantics. 
    Orbit<EnclosureType> orbit(const SystemType& system, const EnclosureType& initial_set, const TimeType& time, Semantics semantics) const;


    //! \brief Compute an approximation to the evolution set using the given semantics. 
    EnclosureListType evolve(const SystemType& system, const EnclosureType& initial_set, const TimeType& time, Semantics semantics) const {
        EnclosureListType final; EnclosureListType reachable; EnclosureListType intermediate; 
        this->_evolution(final,reachable,intermediate,system,initial_set,time,semantics,false); 
        return final; }

    //! \brief Compute an approximation to the evolution set under the given semantics. 
    EnclosureListType reach(const SystemType& system, const EnclosureType& initial_set, const TimeType& time, Semantics semantics) const {
        EnclosureListType final; EnclosureListType reachable; EnclosureListType intermediate; 
        this->_evolution(final,reachable,intermediate,system,initial_set,time,semantics,true); 
        return reachable; }

    TimedEnclosureListType timed_evolution(const SystemType& system, const EnclosureType& initial, 
        const TimeType& time, Semantics semantics, bool reach) const;

  protected:
    virtual void _evolution(EnclosureListType& final, EnclosureListType& reachable, EnclosureListType& intermediate, 
                            const SystemType& system, const EnclosureType& initial, const TimeType& time, 
                            Semantics semantics, bool reach) const;

    typedef tuple<DiscreteState, EventListType, SetModelType, TimeModelType> HybridTimedSetType;
    virtual void _evolution_step(std::vector< HybridTimedSetType >& working_sets, 
                                 EnclosureListType& final, EnclosureListType& reachable, EnclosureListType& intermediate,  
                                 const SystemType& system, const HybridTimedSetType& current_set, const TimeType& time, 
                                 Semantics semantics, bool reach) const;

  private:
    boost::shared_ptr< EvolutionParametersType > _parameters;
    boost::shared_ptr< CalculusInterface<ModelType> > _toolbox;
    //boost::shared_ptr< EvolutionProfiler >  _profiler;
};


  
} // namespace Ariadne

#endif // ARIADNE_STABLE_HYBRID_EVOLVER_H