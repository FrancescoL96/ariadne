/***************************************************************************
 *            set_based_hybrid_evolver.code.h
 *
 *  Copyright  2004-7  Alberto Casagrande,  Pieter Collins
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
 
#include "geometry/rectangle_expression.h"
#include "geometry/set_interface.h"
#include "geometry/hybrid_set.h"
#include "system/set_based_hybrid_automaton.h"
#include "evaluation/evolution_parameters.h"
#include "evaluation/map_evolver.h"
#include "evaluation/vector_field_evolver.h"

#include "evaluation/applicator.h"
#include "evaluation/lohner_integrator.h"

#include "output/epsstream.h"
#include "output/logging.h"

#include "set_based_hybrid_evolver.h"

namespace Ariadne {
 
namespace Evaluation { static int& verbosity = hybrid_evolver_verbosity; }

namespace Evaluation {

  template<class SetInterface> 
  class HybridTimedSet 
  {
   public:
    HybridTimedSet(const time_type& t, const Geometry::DiscreteState& id, const SetInterface& s)
      : _time(t), _discrete_state(id), _continuous_state_set(s) { }
    const time_type& time() const { return _time; }
    const Geometry::DiscreteState& discrete_state() const { return _discrete_state; }
    const SetInterface& continuous_state_set() const { return _continuous_state_set; } 
    
    bool operator==(const HybridTimedSet& other) const { 
      return this->_time == other._time 
        && this->_discrete_state==other._discrete_state 
        && this->_continuous_state_set==other._continuous_state_set; }
    bool operator!=(const HybridTimedSet& other) const { return !(*this==other); }
    bool operator<=(const HybridTimedSet& other) const { return this->_time <= other._time; }
   private:
    time_type _time;
    Geometry::DiscreteState _discrete_state;
    SetInterface _continuous_state_set;
  };

}


template<class R>
Evaluation::SetBasedHybridEvolver<R>::~SetBasedHybridEvolver()
{
  delete this->_applicator;
  delete this->_integrator;
}

template<class R>
Evaluation::SetBasedHybridEvolver<R>::SetBasedHybridEvolver(const EvolutionParameters<R>& p)
  : _applicator(new MapEvolver<R>(p)), _integrator(new VectorFieldEvolver<R>(p))
{
}

template<class R>
Evaluation::SetBasedHybridEvolver<R>::SetBasedHybridEvolver(const MapEvolver<R>& a, const VectorFieldEvolver<R>& i)
  : _applicator(new MapEvolver<R>(a)), _integrator(new VectorFieldEvolver<R>(i))
{
}

template<class R>
Evaluation::SetBasedHybridEvolver<R>::SetBasedHybridEvolver(const SetBasedHybridEvolver<R>& e)
  : _applicator(e._applicator->clone()), _integrator(e._integrator->clone())
{
}





template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::discrete_step(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                    const Geometry::HybridSet<R>& initial_set)
{
  ARIADNE_LOG(2,"HybridSet SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton automaton, HybridSet initial_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n");
  if(automaton.locations()!=initial_set.locations()) {
    throw std::runtime_error("SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton,HybridSet): initial_set locations do not match hybrid_automaton modes");
  }
  ARIADNE_CHECK_BOUNDED(initial_set,"SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton automaton, HybridSet initial_set)");

  typedef Geometry::HybridSpace::const_iterator locations_iterator;

  Geometry::HybridSpace locations=automaton.locations();
  R grid_separation=this->_applicator->parameters().grid_length();

  Geometry::HybridGridMaskSet<R> grid_initial_set;
  for(locations_iterator loc_iter=locations.begin(); loc_iter!=locations.end(); ++loc_iter) {
    Geometry::DiscreteState id=loc_iter->discrete_state();
    dimension_type dim=loc_iter->dimension();
    Geometry::Grid<R> grid(LinearAlgebra::Vector<R>(dim,grid_separation));
    Geometry::Box<R> bounding_box=initial_set[id].bounding_box();
    Geometry::FiniteGrid<R> finite_grid(grid,bounding_box);
    grid_initial_set.new_location(id,finite_grid);
    grid_initial_set[id].adjoin_outer_approximation(initial_set[id]);
  }
  Geometry::HybridGridMaskSet<R> grid_result=discrete_step(automaton,grid_initial_set);
  Geometry::HybridDenotableSet< Geometry::GridMaskSet<R> >& grid_base_result(grid_result);
  Geometry::HybridSet<R> result(grid_base_result);
  return result;
}


template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::continuous_chainreach(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                            const Geometry::HybridSet<R>& initial_set)
{
  throw NotImplemented(__PRETTY_FUNCTION__);
}



template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::lower_reach(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                  const Geometry::HybridSet<R>& initial_set)
{
  ARIADNE_LOG(2,"HybridSet SetBasedHybridEvolver::lower_reach(SetBasedHybridAutomaton automaton, HybridSet initial_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n");

  typedef Geometry::HybridSpace::const_iterator locations_iterator;

  Geometry::HybridSpace locations=automaton.locations();
  R grid_separation=this->_applicator->parameters().grid_length();

  Geometry::HybridGridCellListSet<R> grid_initial_set;
  for(locations_iterator loc_iter=locations.begin(); loc_iter!=locations.end(); ++loc_iter) {
    Geometry::DiscreteState id=loc_iter->discrete_state();
    dimension_type dim=loc_iter->dimension();
    Geometry::Grid<R> grid(LinearAlgebra::Vector<R>(dim,grid_separation));
    grid_initial_set.new_location(id,grid);
    grid_initial_set[id].adjoin_outer_approximation(initial_set[id]);
  }
  Geometry::HybridGridCellListSet<R> grid_result=lower_reach(automaton,grid_initial_set);
  Geometry::HybridDenotableSet< Geometry::GridCellListSet<R> >& grid_base_result(grid_result);
  Geometry::HybridSet<R> result(grid_base_result);
  return result;
}



template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::chainreach(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                 const Geometry::HybridSet<R>& initial_set)
{
  throw NotImplemented(__PRETTY_FUNCTION__);
}





template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::continuous_chainreach(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                            const Geometry::HybridSet<R>& initial_set,
                                                            const Geometry::HybridSet<R>& bounding_set)
{
  ARIADNE_LOG(2,"HybridSet SetBasedHybridEvolver::continuous_chainreach(SetBasedHybridAutomaton automaton, HybridSet initial_set, HybridSet bounding_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n"<<"bounding_set="<<bounding_set<<"\n");
  ARIADNE_CHECK_BOUNDED(initial_set,"SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton automaton, HybridSet initial_set)");

  typedef Geometry::HybridSpace::const_iterator locations_iterator;

  Geometry::HybridSpace locations=automaton.locations();
  R grid_separation=this->_applicator->parameters().grid_length();

  Geometry::HybridGridMaskSet<R> grid_initial_set;
  Geometry::HybridGridMaskSet<R> grid_bounding_set;
  for(locations_iterator loc_iter=locations.begin(); loc_iter!=locations.end(); ++loc_iter) {
    Geometry::DiscreteState id=loc_iter->discrete_state();
    dimension_type dim=loc_iter->dimension();
    Geometry::Grid<R> grid(LinearAlgebra::Vector<R>(dim,grid_separation));
    Geometry::Box<R> bounding_box=initial_set[id].bounding_box();
    Geometry::FiniteGrid<R> finite_grid(grid,bounding_box);
    grid_initial_set.new_location(id,finite_grid);
    grid_initial_set[id].adjoin_outer_approximation(initial_set[id]);
    grid_bounding_set.new_location(id,finite_grid);
    grid_bounding_set[id].adjoin_outer_approximation(bounding_set[id]);
  }
  Geometry::HybridGridMaskSet<R> grid_result=continuous_chainreach(automaton,grid_initial_set,grid_bounding_set);
  Geometry::HybridDenotableSet< Geometry::GridMaskSet<R> >& grid_base_result(grid_result);
  Geometry::HybridSet<R> result(grid_base_result);
  return result;
}



template<class R>
Geometry::HybridSet<R> 
Evaluation::SetBasedHybridEvolver<R>::chainreach(const System::SetBasedHybridAutomaton<R>& automaton, 
                                                 const Geometry::HybridSet<R>& initial_set, 
                                                 const Geometry::HybridSet<R>& bounding_set)
{
  using namespace Geometry;
  ARIADNE_LOG(1,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridSet,HybridSet)"<<std::endl);
  ARIADNE_LOG(2,"  initial_set="<<initial_set<<"\n  bounding_set="<<bounding_set<<std::endl);

  ARIADNE_CHECK_SAME_LOCATIONS(automaton,initial_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridSet,HybridSet)");
  ARIADNE_CHECK_SAME_LOCATIONS(automaton,bounding_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridSet,HybridSet)");
  ARIADNE_CHECK_BOUNDED(bounding_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridSet,HybridSet): bounding_set");
  ARIADNE_LOG(5,"Checked input"<<std::endl);
  
  HybridGridMaskSet<R> grid_bounding_set;
  HybridGridMaskSet<R> grid_initial_set;
  
  for(typename Geometry::HybridSet<R>::locations_const_iterator bs_iter=bounding_set.locations_begin();
      bs_iter!=bounding_set.locations_end(); ++bs_iter)
  {
    Geometry::DiscreteState id=bs_iter->first;
    Grid<R> grid(bs_iter->second->dimension(),this->_applicator->parameters().grid_length());
    FiniteGrid<R> fgrid(grid,bs_iter->second->bounding_box());
    ARIADNE_LOG(5,"Made grid"<<std::endl);
    grid_bounding_set.new_location(id,fgrid);
    grid_initial_set.new_location(id,fgrid);
    grid_bounding_set[id].adjoin_outer_approximation(*bs_iter->second);
    grid_bounding_set[id].restrict_outer_approximation(automaton.mode(id).invariant());
    grid_initial_set[id].adjoin_outer_approximation(*bs_iter->second);
    grid_initial_set[id].restrict_outer_approximation(initial_set[id]);
  }
  ARIADNE_LOG(5,"Made cells"<<std::endl);


  ARIADNE_LOG(2,"  grid_initial_set="<<grid_initial_set<<"\n  grid_bounding_set="<<grid_bounding_set<<std::endl);
  HybridGridMaskSet<R> grid_chainreach_set=this->chainreach(automaton,grid_initial_set,grid_bounding_set);
  ARIADNE_LOG(2,"  grid_chainreach_set="<<grid_chainreach_set<<std::endl);
  HybridSet<R> chainreach_set(grid_chainreach_set);
  ARIADNE_LOG(2,"  chainreach_set="<<chainreach_set<<std::endl);
  return chainreach_set;
}




template<class R>
Geometry::HybridListSet< Geometry::Box<R> > 
Evaluation::SetBasedHybridEvolver<R>::_discrete_step(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                     const Geometry::HybridListSet< Geometry::Box<R> >& initial_set)
{
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_transition_const_iterator discrete_transition_const_iterator;

  Geometry::HybridListSet< Geometry::Box<R> > result_set(initial_set.space());

  for(discrete_transition_const_iterator dt_iter=hybrid_automaton.transitions().begin();
      dt_iter!=hybrid_automaton.transitions().end(); ++dt_iter)
  {
    const System::SetBasedDiscreteTransition<R>& dt = *dt_iter;
    const Geometry::ListSet< Geometry::Box<R> >& initial=initial_set[dt.source().discrete_state()];
    Geometry::ListSet< Geometry::Box<R> >& destination=result_set[dt.destination().discrete_state()];
    Geometry::ListSet< Geometry::Box<R> > active=Geometry::inner_intersection(initial,dt.activation());
    Geometry::ListSet< Geometry::Box<R> > image=this->_applicator->image(dt.reset(),active);
    ARIADNE_LOG(4,", "<<image.size()<<" boxes in image\n");
    destination.adjoin(image);
  }
  return result_set;
}


template<class R>
Geometry::HybridListSet< Geometry::Box<R> > 
Evaluation::SetBasedHybridEvolver<R>::_continuous_reach(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                        const Geometry::HybridListSet< Geometry::Box<R> >& initial_set,
                                                        const Numeric::Rational& maximum_time)
{
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_mode_const_iterator discrete_mode_const_iterator;

  Geometry::HybridListSet< Geometry::Box<R> > result_set(initial_set.space());

  for(discrete_mode_const_iterator dm_iter=hybrid_automaton.modes().begin();
      dm_iter!=hybrid_automaton.modes().end(); ++dm_iter)
  {
    const System::SetBasedDiscreteMode<R>& dm = *dm_iter;
    const Geometry::ListSet< Geometry::Box<R> >& initial=initial_set[dm.discrete_state()];
    Geometry::ListSet< Geometry::Box<R> >& reach=result_set[dm.discrete_state()];
    ARIADNE_LOG(4,"continuous_reach in mode "<<dm.discrete_state()<<":\n");
    reach=this->_integrator->lower_reach(dm.dynamic(),initial,dm.invariant(),maximum_time);
    ARIADNE_LOG(4,", "<<reach.size()<<" boxes in reached set\n");
  }
  return result_set;
}




template<class R>
Geometry::HybridGridMaskSet<R> 
Evaluation::SetBasedHybridEvolver<R>::_discrete_step(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                     const Geometry::HybridGridMaskSet<R>& initial_set, 
                                                     const Geometry::HybridGridMaskSet<R>& domain_set)
{
  Geometry::HybridGridMaskSet<R> result_set(initial_set);
  result_set.clear();
  
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_transition_const_iterator discrete_transition_const_iterator;
  
  for(discrete_transition_const_iterator dt_iter=hybrid_automaton.transitions().begin();
      dt_iter!=hybrid_automaton.transitions().end(); ++dt_iter)
  {
    const System::SetBasedDiscreteTransition<R>& dt = *dt_iter;
    const Geometry::GridMaskSet<R>& initial=initial_set[dt.source().discrete_state()];
    Geometry::GridMaskSet<R>& destination=result_set[dt.destination().discrete_state()];
    Geometry::GridCellListSet<R> active=initial;
    active.restrict_outer_approximation(dt.activation());
    ARIADNE_LOG(4,"discrete_step of transition "<<dt.discrete_event()<<" from mode "<<dt.source().discrete_state()<<" to mode "<<dt.destination().discrete_state()<<":\n");
    ARIADNE_LOG(4,"  "<<active.size()<<" activated cells");
    Geometry::GridCellListSet<R> image=this->_applicator->image(dt.reset(),active,destination.grid());
    image.unique_sort();
    ARIADNE_LOG(4,", "<<image.size()<<" cells in image\n");
    destination.adjoin(image);
  }
  return result_set;
}


template<class R>
Geometry::HybridGridMaskSet<R> 
Evaluation::SetBasedHybridEvolver<R>::_continuous_chainreach(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                             const Geometry::HybridGridMaskSet<R>& initial_set,
                                                             const Geometry::HybridGridMaskSet<R>& domain_set)
{
  Geometry::HybridGridMaskSet<R> result_set(initial_set);
  
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_mode_const_iterator discrete_mode_const_iterator;
  result_set.clear();
  
  for(discrete_mode_const_iterator dm_iter=hybrid_automaton.modes().begin();
      dm_iter!=hybrid_automaton.modes().end(); ++dm_iter)
  {
    const System::SetBasedDiscreteMode<R>& dm = *dm_iter;
    const System::VectorFieldInterface<R>& vf = dm.dynamic();
    const Geometry::GridMaskSet<R>& domain=domain_set[dm.discrete_state()];
    const Geometry::GridMaskSet<R>& initial=initial_set[dm.discrete_state()];
    Geometry::GridMaskSet<R> start=regular_intersection(initial,domain);
    ARIADNE_LOG(4,"continuous_chainreach in mode "<<dm.discrete_state()<<":\n  "<<start.size()<<" initial cells,");
    result_set[dm.discrete_state()].adjoin(this->_integrator->chainreach(vf,start,domain));
    result_set[dm.discrete_state()].restrict(domain);
    ARIADNE_LOG(4," reached "<<result_set[dm.discrete_state()].size()<<" cells\n");
  }
  return result_set;
}


template<class R>
Geometry::HybridGridMaskSet<R> 
Evaluation::SetBasedHybridEvolver<R>::discrete_step(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                    const Geometry::HybridGridMaskSet<R>& initial_set)
{
  ARIADNE_LOG(2,"HybridGridMaskSet SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton automaton, HybridGridMaskSet initial_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,initial_set,"SetBasedHybridEvolver::discrete_step(SetBasedHybridAutomaton,HybridGridMaskSet)");

  Geometry::HybridGridMaskSet<R> result_set(initial_set);
  result_set.clear();
  
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_transition_const_iterator discrete_transition_const_iterator;
  
  for(discrete_transition_const_iterator dt_iter=hybrid_automaton.transitions().begin();
      dt_iter!=hybrid_automaton.transitions().end(); ++dt_iter)
  {
    ARIADNE_LOG(4,"  transition="<<*dt_iter<<"\n");
    const System::SetBasedDiscreteTransition<R>& dt = *dt_iter;
    const Geometry::GridMaskSet<R>& source_set=initial_set[dt.source().discrete_state()];
    Geometry::GridMaskSet<R>& destination_set=result_set[dt.destination().discrete_state()];
    Geometry::GridMaskSet<R> active_cells=source_set;
    active_cells.restrict_outer_approximation(dt.activation());
    ARIADNE_LOG(4,"discrete_step of transition "<<dt.discrete_event()<<" from mode "<<dt.source().discrete_state()<<" to mode "<<dt.destination().discrete_state()<<":\n")
    ARIADNE_LOG(4,"  "<<active_cells.size()<<" activated cells");
    const Geometry::GridCellListSet<R> image_cells=this->_applicator->image(dt.reset(),active_cells,destination_set.grid());
    ARIADNE_LOG(4,", "<<image_cells.size()<<" cells in image\n");
    destination_set.adjoin(image_cells);

  }
  
  return result_set;
}


template<class R>
Geometry::HybridGridMaskSet<R> 
Evaluation::SetBasedHybridEvolver<R>::continuous_chainreach(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                            const Geometry::HybridGridMaskSet<R>& initial_set,
                                                            const Geometry::HybridGridMaskSet<R>& bounding_set)
{
  ARIADNE_LOG(2,"HybridGridMaskSet SetBasedHybridEvolver::continuous_chainreach(SetBasedHybridAutomaton automaton, HybridGridMaskSet initial_set, HybridGridMaskSet bounding_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n"<<"bounding_set="<<bounding_set<<"\n");
  ARIADNE_LOG(3,"checking input... ");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,initial_set,"SetBasedHybridEvolver::continuous_chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet)");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,bounding_set,"SetBasedHybridEvolver::continuous_chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet)");
  
  ARIADNE_CHECK_BOUNDED(bounding_set,"SetBasedHybridEvolver::continuous_chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet): bounding_set");
  
  ARIADNE_LOG(3,"successful\n");

  typedef typename System::SetBasedHybridAutomaton<R>::discrete_mode_const_iterator discrete_mode_const_iterator;
  
  Geometry::HybridGridMaskSet<R> domain_set(bounding_set);
  for(discrete_mode_const_iterator dm_iter=hybrid_automaton.modes().begin();
      dm_iter!=hybrid_automaton.modes().end(); ++dm_iter)
  {
    ARIADNE_LOG(3,"computing domain set in mode "<<dm_iter->discrete_state()<<"... invariant="<<dm_iter->invariant()<<"\n");
    domain_set[dm_iter->discrete_state()].restrict_outer_approximation(dm_iter->invariant());
  }
  ARIADNE_LOG(3,"domain_set="<<domain_set<<"\n");

  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n"<<"domain_set="<<domain_set<<"\n");
  return _continuous_chainreach(hybrid_automaton,initial_set,domain_set);
}



template<class R>
Geometry::HybridGridCellListSet<R> 
Evaluation::SetBasedHybridEvolver<R>::lower_reach(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                  const Geometry::HybridGridCellListSet<R>& initial_set)
{
  using namespace Geometry;
  ARIADNE_LOG(2,"HybridGridCellListSet SetBasedHybridEvolver::lower_reach(SetBasedHybridAutomaton automaton, HybridGridCellListSet initial_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,initial_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet)");

  typedef typename System::SetBasedHybridAutomaton<R>::discrete_transition_const_iterator discrete_transition_const_iterator;
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_mode_const_iterator discrete_mode_const_iterator;

  const Geometry::HybridGrid<R>& grid=initial_set.grid();
  Geometry::HybridListSet< Box<R> > reach_set(initial_set.space());
  Geometry::HybridListSet< Box<R> > integrated_set(initial_set.space());
  Geometry::HybridListSet< Box<R> > found_set(initial_set.space());
  found_set.adjoin(initial_set);

  ARIADNE_LOG(4,found_set.size()<<" cells in initial set\n");
  uint step=0;
  // FIXME: Use correct size;
  uint maximum_steps=16;
  Numeric::Rational maximum_integration_time=16;
  while(step<maximum_steps && !!found_set.empty()) {
    integrated_set=this->_continuous_reach(hybrid_automaton,found_set,maximum_integration_time);
    ARIADNE_LOG(4,"\nintegration found "<<integrated_set.size()<<" boxes by continuous evolution, \n"<<std::endl);
    found_set=this->_discrete_step(hybrid_automaton,integrated_set);
    ARIADNE_LOG(4,"\nreset found "<<found_set.size()<<" boxes by discrete step, \n"<<std::endl);
    reach_set.adjoin(found_set);
    if(verbosity>=4) {
      std::stringstream filename;
      filename << "hybrid_lower_reach-"<<step<<".eps";
      Output::epsfstream eps;
    }
  }

  return Geometry::outer_approximation(reach_set,grid);
}


template<class R>
Geometry::HybridGridMaskSet<R> 
Evaluation::SetBasedHybridEvolver<R>::chainreach(const System::SetBasedHybridAutomaton<R>& hybrid_automaton, 
                                                 const Geometry::HybridGridMaskSet<R>& initial_set, 
                                                 const Geometry::HybridGridMaskSet<R>& bounding_set)
{
  using namespace Geometry;
  ARIADNE_LOG(2,"HybridGridMaskSet SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton automaton, HybridGridMaskSet initial_set, HybridGridMaskSet bounding_set)\n");
  ARIADNE_LOG(3,"initial_set="<<initial_set<<"\n"<<"bounding_set="<<bounding_set<<"\n");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,initial_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet)");
  ARIADNE_CHECK_SAME_LOCATIONS(hybrid_automaton,bounding_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet)");
  
  ARIADNE_CHECK_BOUNDED(bounding_set,"SetBasedHybridEvolver::chainreach(SetBasedHybridAutomaton,HybridGridMaskSet,HybridGridMaskSet): bounding_set");

  typedef typename System::SetBasedHybridAutomaton<R>::discrete_transition_const_iterator discrete_transition_const_iterator;
  typedef typename System::SetBasedHybridAutomaton<R>::discrete_mode_const_iterator discrete_mode_const_iterator;
  
  // Compute restricted invariant domains as GridMaskSets
  Geometry::HybridGridMaskSet<R> domain_set(bounding_set);
  for(discrete_mode_const_iterator dm_iter=hybrid_automaton.modes().begin();
      dm_iter!=hybrid_automaton.modes().end(); ++dm_iter)
  {
    domain_set[dm_iter->discrete_state()].restrict_outer_approximation(dm_iter->invariant());
  }

  HybridGridMaskSet<R> result_set=domain_set;
  HybridGridMaskSet<R> integrated_set=domain_set;
  integrated_set.clear();
  result_set.restrict(initial_set);
  ARIADNE_LOG(4,result_set.size()<<" cells in initial set\n");
  HybridGridMaskSet<R> found_set=result_set;
  uint step=0;
  while(!found_set.empty()) {
    integrated_set=this->_continuous_chainreach(hybrid_automaton,found_set,domain_set);
    ARIADNE_LOG(4,"\nchainreach found "<<integrated_set.size()<<" cell by continuous evolution, \n"<<std::endl);
    if(verbosity>=4) {
      std::stringstream filename;
      filename << "hybrid_chainreach-"<<step<<".eps";
      const GridMaskSet<R>& dom=domain_set.locations_begin()->second;
      const GridMaskSet<R>& res=result_set.locations_begin()->second;
      const GridMaskSet<R>& fnd=found_set.locations_begin()->second;
      Output::epsfstream eps;
      eps.open(filename.str().c_str(),dom.bounding_box());
      eps << fill_colour(Output::green) << dom.extent();
      eps << fill_colour(Output::green) << res;
      eps << fill_colour(Output::red) << fnd;
      eps.close();
    }
    found_set=this->_discrete_step(hybrid_automaton,integrated_set,domain_set);
    ARIADNE_LOG(4,"\nchainreach found "<<found_set.size()<<" cells by discrete step");
    found_set.remove(result_set);
    ARIADNE_LOG(4," of which "<<found_set.size()<<" are new; ");
    result_set.adjoin(integrated_set);
    result_set.adjoin(found_set);
    ARIADNE_LOG(4,"reached "<<result_set.size()<< " cells in total.\n" << std::endl);
    ++step;
  }
  return result_set;
}


}
