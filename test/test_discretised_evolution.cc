/***************************************************************************
 *      test_discretised_evolution.cc
 *
 *  Copyright  2006-8  Pieter Collins
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

#include <fstream>

#include "tuple.h"
#include "vector.h"
#include "matrix.h"
#include "function.h"
#include "taylor_set.h"
#include "zonotope.h"
#include "list_set.h"
#include "grid_set.h"
#include "hybrid_time.h"
#include "hybrid_set.h"
#include "hybrid_automaton.h"
#include "integrator.h"
#include "map_evolver.h"
#include "vector_field_evolver.h"
#include "hybrid_evolver.h"
#include "orbit.h"
#include "discretiser.h"
#include "graphics.h"

#include "models.h"

#include "logging.h"

#include "test.h"

using namespace Ariadne;
using namespace std;
using Models::VanDerPol;

class TestDiscretisedEvolution
{
  public:
    void test() const;
  private:
    void test_discrete_time() const;
    void test_continuous_time() const;
    void test_hybrid_time() const;
};

int main()
{
    std::cerr<<"SKIPPED "; return 1u;
    TestDiscretisedEvolution().test();
    return ARIADNE_TEST_FAILURES;
}

void TestDiscretisedEvolution::test() const
{
    //ARIADNE_TEST_CALL(test_discrete_time());
    //ARIADNE_TEST_CALL(test_continuous_time());
    ARIADNE_TEST_CALL(test_hybrid_time());
}


void TestDiscretisedEvolution::test_discrete_time() const
{
    typedef MapEvolver::EnclosureType EnclosureType;

    cout << __PRETTY_FUNCTION__ << endl;

    // Set up the evolution parameters and grid
    uint steps(6);
    double maximum_step_size(0.125);
    int depth=8;

    EvolutionParameters parameters;
    parameters.maximum_step_size=maximum_step_size;
    Grid grid(2);

    // Set up the evaluators
    MapEvolver evolver(parameters);

    Discretiser< IteratedMap, EnclosureType > discrete_evolver(evolver);


    // Set up the vector field
    Real a=1.5; Real b=0.375;
    RealScalarFunction x=RealScalarFunction::coordinate(2,0);
    RealScalarFunction y=RealScalarFunction::coordinate(2,1);
    RealVectorFunction henon=join(a-x*x+b*y,x);
    cout << "henon=" << henon << endl;
    IteratedMap system(henon);

    // Define a bounding box for the evolution
    std::cout<<"making bounding_box"<<std::endl;
    Box bounding_box=make_box("[-4,4]x[-4,4]") ;
    std::cout<<"bounding_box="<<bounding_box<<"\n"<<std::endl;

    // Define the initial cell
    Box box=make_box("[1.001,1.002]x[0.501,0.502]");
    GridTreeSet approx_tree_set=outer_approximation(box,grid,depth);
    GridCell initial_cell=*approx_tree_set.begin();
    cout << "initial_cell=" << initial_cell << endl << endl;
    Box initial_box=initial_cell.box();
    cout << "initial_box=" << initial_box << endl << endl;
    //[1.00098:1.00122],
    cout << "steps=" << steps << endl << endl;

    // Compute the reachable sets
    cout << "Computing evolution... " << flush;
    Orbit<EnclosureType> evolve_orbit
     = evolver.orbit(system,initial_box,steps,UPPER_SEMANTICS);
    cout << "done." << endl;

    EnclosureType const& initial_set=evolve_orbit.initial();
    ListSet<EnclosureType> const& reach_set=evolve_orbit.reach();
    ListSet<EnclosureType> const& intermediate_set=evolve_orbit.intermediate();
    ListSet<EnclosureType> const& final_set=evolve_orbit.final();

    // Compute the reachable sets
    cout << "Computing discretised evolution... " << flush;
    Orbit<GridCell> discrete_orbit
     = discrete_evolver.upper_evolution(system,initial_cell,steps,grid,depth);
    cout << "done." << endl;

    GridTreeSet const& reach_cells=discrete_orbit.reach();
    GridTreeSet const& intermediate_cells=discrete_orbit.intermediate();
    GridTreeSet const& final_cells=discrete_orbit.final();

    cout << "initial_set=" << initial_set.bounding_box() << endl << endl;
    cout << "initial_cell=" << initial_cell.box() << endl << endl;
    cout << "reach_set=" << reach_set << endl << endl;
    cout << "reach_cells=" << reach_cells << endl << endl;
    cout << "intermediate_set=" << intermediate_set << endl << endl;
    cout << "intermediate_cells=" << intermediate_cells << endl << endl;
    cout << "final_set=" << final_set << endl << endl;
    cout << "final_cells=" << final_cells << endl << endl;

    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_cells;
     fig << fill_colour(yellow) << initial_cell;
     fig << fill_colour(green) << final_cells;
     fig.write("test_discretised_evolution-henon-cells");
    }

    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_set;
     fig << fill_colour(yellow) << initial_set;
     fig << fill_colour(green) << final_set;
     fig.write("test_discretised_evolution-henon-sets");
    }
}

void TestDiscretisedEvolution::test_continuous_time() const
{
    typedef TaylorConstrainedImageSet EnclosureType;

    cout << __PRETTY_FUNCTION__ << endl;

    // Set up the evolution parameters and grid
    Real time(1.0);
    double maximum_step_size(0.125);
    int depth=8;

    EvolutionParameters parameters;
    parameters.maximum_step_size=maximum_step_size;
    Grid grid(2);

    // Set up the evaluators
    TaylorIntegrator integrator(4,1e-4);
    VectorFieldEvolver evolver(parameters,integrator);
    Discretiser< VectorField, EnclosureType > discretiser(evolver);


    // Set up the vector field
    Float mu=0.865;
    VectorUserFunction<VanDerPol> vdp(Vector<Float>(1,&mu));
    cout << "vdp=" << vdp << endl;
    VectorField system(vdp);

    // Define a bounding box for the evolution
    Box bounding_box=make_box("[-4,4]x[-4,4]") ;
    //Box eps_bounding_box=bounding_box.neighbourhood(0.1);

    // Define the initial cell
    Box box=make_box("[1.01,1.02]x[0.51,0.52]");
    cout << "box=" << box << endl;
    GridTreeSet approx_tree_set=outer_approximation(box,grid,depth);
    GridCell initial_cell=*approx_tree_set.begin();
    Box initial_box=initial_cell.box();
    cout << "initial_box=" << initial_box << endl << endl;

    // Compute the reachable sets
    cout << "Computing evolution... " << flush;
    Orbit<EnclosureType> evolve_orbit
     = evolver.orbit(system,initial_box,time,UPPER_SEMANTICS);
    cout << "done." << endl;
    EnclosureType const& initial_set=evolve_orbit.initial();
    ListSet<EnclosureType> const& reach_set=evolve_orbit.reach();
    ListSet<EnclosureType> const& intermediate_set=evolve_orbit.intermediate();
    ListSet<EnclosureType> const& final_set=evolve_orbit.final();

    // Compute the reachable sets
    cout << "Computing discretised evolution... " << flush;
    Orbit<GridCell> discrete_orbit
     = discretiser.upper_evolution(system,initial_cell,time,grid,depth);
    cout << "done." << endl;
    GridTreeSet const& reach_cells=discrete_orbit.reach();
    GridTreeSet const& intermediate_cells=discrete_orbit.intermediate();
    GridTreeSet const& final_cells=discrete_orbit.final();

    cout << "initial_set_bounding_box=" << initial_set.bounding_box() << endl << endl;
    cout << "initial_cell=" << initial_cell.box() << endl << endl;
    cout << "final_set=" << final_set << endl << endl;
    cout << "final_cells=" << final_cells << endl << endl;

    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_cells;
     fig << fill_colour(magenta) << intermediate_cells;
     fig << fill_colour(blue) << initial_cell;
     fig << fill_colour(blue) << final_cells;
     fig.write("test_discretised_evolution-vdp-cells");
    }

    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_set;
     fig << fill_colour(magenta) << intermediate_set;
     fig << fill_colour(blue) << initial_set;
     fig << fill_colour(blue) << final_set;
     fig.write("test_discretised_evolution-vdp-sets");
    }


}


void TestDiscretisedEvolution::test_hybrid_time() const
{
    typedef GeneralHybridEvolver EvolverType;
    typedef EvolverType::EnclosureType EnclosureType;
    typedef EnclosureType::ContinuousStateSetType ContinuousEnclosureType;

    cout << __PRETTY_FUNCTION__ << endl;

    // Set up the evolution parameters and grid
    Real time(1.0);
    uint steps(6);
    double maximum_step_size(0.125);
    int depth=8;
    DiscreteLocation location(1);
    DiscreteEvent event(1);

    EvolutionParameters parameters;
    parameters.maximum_step_size=maximum_step_size;
    Grid grid(2);

    // Set up the evaluators
    EvolverType evolver(parameters);
    HybridDiscretiser< EnclosureType > discrete_evolver(evolver);


    // Set up the vector field
    Real a=1.5; Real b=0.375;
    RealScalarFunction zero=RealScalarFunction::constant(2,0.0);
    RealScalarFunction one=RealScalarFunction::constant(2,1.0);
    RealScalarFunction x=RealScalarFunction::coordinate(2,0);
    RealScalarFunction y=RealScalarFunction::coordinate(2,1);

    MonolithicHybridAutomaton ha("Decay");
    ha.new_mode(location,(one,-y));
    ha.new_transition(location,event,location,(x-1,y),(x-1),urgent);

    // Define a bounding box for the evolution
    std::cout<<"making bounding_box"<<std::endl;
    Box bounding_box=make_box("[-4,4]x[-4,4]") ;
    std::cout<<"bounding_box="<<bounding_box<<"\n"<<std::endl;
    //Box eps_bounding_box=bounding_box.neighbourhood(0.1);

    // Define the initial cell
    Box initial_box=make_box("[1.0001,1.0002]x[0.5001,0.5002]");
    ARIADNE_TEST_PRINT(initial_box);
    GridTreeSet approx_tree_set=outer_approximation(initial_box,grid,depth);
    GridCell initial_cell=*approx_tree_set.begin();
    HybridGridCell hybrid_initial_cell(location,initial_cell);
    ARIADNE_TEST_PRINT(hybrid_initial_cell);
    HybridBox hybrid_initial_set=hybrid_initial_cell.box();
    ARIADNE_TEST_PRINT(hybrid_initial_set);
    //[1.00098:1.00122],
    HybridTime htime(time,steps);
    ARIADNE_TEST_PRINT(htime);

    // Compute the reachable sets
    cout << "Computing evolution... " << flush;
    // evolver.verbosity=1;
    Orbit<EnclosureType> evolve_orbit
     = evolver.orbit(ha,EnclosureType(hybrid_initial_set),htime,UPPER_SEMANTICS);
    cout << "done." << endl;

    ARIADNE_TEST_PRINT(evolve_orbit);

    cout << "Extracting grid... " << flush;
    HybridGrid hagrid=ha.grid();
    cout << "done." << endl;

    // Compute the reachable sets
    cout << "Computing discretised evolution... " << flush;
    Orbit<HybridGridCell> discrete_orbit
     = discrete_evolver.evolution(ha,hybrid_initial_cell,htime,depth,UPPER_SEMANTICS);
    cout << "done." << endl;

    ContinuousEnclosureType const& initial_set=evolve_orbit.initial().continuous_state_set();
    ListSet<ContinuousEnclosureType> const& reach_set=evolve_orbit.reach()[location];
    ListSet<ContinuousEnclosureType> const& intermediate_set=evolve_orbit.intermediate()[location];
    ListSet<ContinuousEnclosureType> const& final_set=evolve_orbit.final()[location];

    GridTreeSet const& reach_cells=discrete_orbit.reach()[location];
    GridTreeSet const& intermediate_cells=discrete_orbit.intermediate()[location];
    GridTreeSet const& final_cells=discrete_orbit.final()[location];


    ARIADNE_TEST_PRINT(initial_set);
    ARIADNE_TEST_PRINT(initial_cell);
    ARIADNE_TEST_PRINT(reach_set);
    ARIADNE_TEST_PRINT(reach_cells);
    ARIADNE_TEST_PRINT(intermediate_set);
    ARIADNE_TEST_PRINT(intermediate_cells);
    ARIADNE_TEST_PRINT(final_set);
    ARIADNE_TEST_PRINT(final_cells);



    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_cells;
     fig << fill_colour(magenta) << intermediate_cells;
     fig << fill_colour(yellow) << initial_cell;
     fig << fill_colour(green) << final_cells;
     fig.write("test_discrete_evolver-hybrid-cells");
    }

    // Plot the intial, evolve and reach sets
    {
     Figure fig;
     fig.set_bounding_box(Box(2,Interval(-3,3)));
     fig << line_style(true);
     fig << fill_colour(cyan) << reach_set;
     fig << fill_colour(magenta) << intermediate_set;
     fig << fill_colour(yellow) << initial_set;
     fig << fill_colour(green) << final_set;
     fig.write("test_discrete_evolver-hybrid-sets");
    }

}

