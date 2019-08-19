/***************************************************************************
 *            watertank-proportional.cpp
 *
 *  Copyright  2017  Luca Geretti
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

#include <cstdarg>
#include "ariadne.hpp"
#include "tank.hpp"
#include "valve-proportional-urgent.hpp"

using namespace Ariadne;
using std::cout; using std::endl;

inline char activity_symbol(SizeType step) {
    switch (step % 4) {
    case 0: return '\\';
    case 1: return '|';
    case 2: return '/';
    default: return '-';
    }
}

void discretize(HybridGridTreePaving& hgts, GeneralHybridEvolver::OrbitType& orbit, unsigned precision)
{
  int oSize=orbit.reach().size();
  std::cerr<<"\n";
  int index=1;
  for (ListSet<HybridEnclosure>::ConstIterator it = orbit.reach().begin(); it != orbit.reach().end(); it++,index++)
  {
      std::cerr << "\r[" << activity_symbol(static_cast<unsigned>(index)) << "] " << static_cast<int>((index*100)/oSize) << "% " << std::flush;
      it->state_auxiliary_set().adjoin_outer_approximation_to(hgts,precision);
  }
  fprintf(stderr,"\n");
}

Int main(Int argc, const char* argv[])
{
    Nat evolver_verbosity=get_verbosity(argc,argv);

    // Declare the shared system variables
    RealVariable aperture("aperture");
    RealVariable height("height");

    StringVariable valve("valve");
    StringConstant opened("opened");
    StringConstant modulated("modulated");
    StringConstant closed("closed");

    // Get the automata and compose them
    HybridAutomaton tank_automaton = getTank();
    HybridAutomaton valve_automaton = getValve();
    CompositeHybridAutomaton watertank_system({tank_automaton,valve_automaton});

    // Print the system description on the command line
    cout << watertank_system << endl;

    // Compute the system evolution

    // Create a GeneralHybridEvolver object
    GeneralHybridEvolver evolver(watertank_system);
    evolver.verbosity = evolver_verbosity;

    // Set the evolution parameters
    evolver.configuration().set_maximum_enclosure_radius(3.05); // The maximum size of an evolved set before early termination
    evolver.configuration().set_maximum_step_size(0.25); // The maximum value that can be used as a time step for integration

    // Declare the type to be used for the system evolution
    typedef GeneralHybridEvolver::OrbitType OrbitType;

    std::cout << "Computing evolution... " << std::flush;
<<<<<<< HEAD

    // Define the initial set, by supplying the location as a list of locations for each composed automata, and
    // the continuous set as a list of variable assignments for each variable controlled on that location
    // (the assignment can be either a singleton value using the == symbol or an interval using the <= symbols)
    HybridSet initial_set({valve|opened},{height==0});
    // Define the evolution time: continuous time and maximum number of transitions
=======
    Real a_max(0.0);

    //HybridSet initial_set({valve|closed},{0<=height<=0.0_decimal});
    HybridSet initial_set({valve|opened},{0.0_decimal<=height<=0.0_decimal});
>>>>>>> Small fixes. Implemented temporary __feasible__ function to test barrier method.
    HybridTime evolution_time(80.0,5);
    // Compute the orbit using upper semantics
    OrbitType orbit = evolver.orbit(initial_set,evolution_time,Semantics::UPPER);
    std::cout << "done." << std::endl;

    // Plot the trajectory using two different projections
    std::cout << "Plotting trajectory... "<<std::flush;
    Axes2d time_height_axes(0<=TimeVariable()<=80,-0.1<=height<=9.1);
    // plot("watertank_proportional_t-height",time_height_axes, Colour(0.0,0.5,1.0), orbit);
    Axes2d height_aperture_axes(-0.1,height,9.1, -0.1,aperture,1.3);
    // plot("watertank_proportional_height-aperture",height_aperture_axes, Colour(0.0,0.5,1.0), orbit);
    std::cout << "done." << std::endl;


    std::cout << "Discretising orbit" << std::flush;
    HybridGrid grid(watertank_system.state_auxiliary_space());
    HybridGridTreePaving hgts(grid);


    for(unsigned i=2;i<=5;++i)
    {
      auto h = hgts;
      clock_t s_time = clock();
      // run code
      discretize(h,orbit,i);
      // End time
      clock_t e_time = clock();
      float elapsed_time =static_cast<float>(e_time - s_time) / CLOCKS_PER_SEC;
      std::cout << "instance "<<i<<" in "<<elapsed_time<<" sec" << std::endl;
      char title[32];
      sprintf(title,"%d",i);
      plot(title, height_aperture_axes, Colour(0.0,0.5,1.0), h);
    }
    // Start time
    std::cerr<<"done.\n";

}
