/***************************************************************************
 *            watertank-monolithic-proportional-verify.cc
 *
 *  Copyright  2011  Luca Geretti
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

#include "ariadne.h"
#include "examples.h"

using namespace Ariadne;

int main(int argc,char *argv[]) 
{
	int analyserVerbosity = 1;
	if (argc > 1)
		analyserVerbosity = atoi(argv[1]);

	// The system
	HybridAutomaton system = Ariadne::getWatertankMonolithicProportional();

	// The initial values
	HybridImageSet initial_set;
	initial_set[DiscreteState(3)] = Box(2, 5.5,5.5, 1.0,1.0);

	// The domain
	HybridBoxes domain = bounding_boxes(system.state_space(),Box(2,-0.1,10.0,-0.1,1.1));

	// The safe region
	HybridBoxes safe_box = bounding_boxes(system.state_space(),Box(2, 5.25, 8.25, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()));

	/// Verification

	// Create an evolver and analyser objects, then set their verbosity
	HybridEvolver evolver;
	evolver.verbosity = 0;
	HybridReachabilityAnalyser analyser(evolver);
	analyser.verbosity = analyserVerbosity;
	evolver.parameters().enable_set_model_reduction = true;
	analyser.parameters().enable_lower_pruning = true;
	analyser.parameters().lowest_maximum_grid_depth = 0;
	analyser.parameters().highest_maximum_grid_depth = 6;

	SystemVerificationInfo verInfo(system, initial_set, domain, safe_box);

	/// Analysis parameters
	RealConstantSet parameters;
	parameters.insert(RealConstant("bfp",Interval(0.01,0.6)));
	parameters.insert(RealConstant("Kp",Interval(0.01,0.6)));
	Float tolerance = 0.1;
	uint numPointsPerAxis = 11;

	Parametric2DBisectionResults results = analyser.parametric_verification_2d_bisection(verInfo, parameters, tolerance, numPointsPerAxis);
	//ParametricPartitioningOutcomeList results = analyser.parametric_verification_partitioning(verInfo, parameters, tolerance);
	results.draw();

}
