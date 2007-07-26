/***************************************************************************
 *            function_evaluation.h
 *
 *  Copyright  2004-7  Pieter Collins
 *  Pieter.Collins@cwi.nl
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

/*! 

\file function_evaluation.h
\brief Documentation on evaluation of functions



\page functionevaluation Function Evaluation

%Ariadne is primarily a module for set-based computation.
For this reason, functions are best defined by their actions on sets.
However, we sometimes also want to compute function values on points, and to evaluate real-valued functions.
For this reason, we also allow definition of functions on points.

We distinguish between computations on \em fixed-precision and \em multiple-precision types, 
between computations on \em points and \em sets, and between \em exact and \em approximate computations.

The basic computation on sets is to compute \em over-approximations to the image of <em>basic sets</em>
From these computations, arbitrarily-accurate computations can be performed.
The basic computation on points can be \em exact and \em approximate, as appropriate.

\section set_functions  Computations on sets.

A valid arbitrary-precision computation on sets is defined as follows
If \f$f\f$ is a mathematical function, and   BS is a basic set type,
then a valid representation \c f of \f$f\f$ is a function which, 
for any basic set \f$A\f$, returns a set \f$B\f$ such that \f$f(A)\subset B\f$,
and such that whenever \f$A_n\f$ is a decreasing sequence of sets with \f$\bigcap_{n=1}^{\infty} A_n=\{x\}\f$,
then \f$\bigcap_{n=1}^{\infty} B_n=\{y\}\f$, where \f$y=f(x)\f$.

A valid fixed-precision computation on sets is an over-approximation. 
In other words, a valid representation \c f of \f$f\f$ is a function which,
for any basic set \f$A\f$, returns a set \f$B\f$ such that \f$f(A)\subset B\f$.
No guarentees on the accuracy are required.
Note that it does not make sense to consider a sequence \f$A_n\f$ converging to a point for fixed-precision types. 

\section point_functions Computations on points.

If a denotable state type is invariant under a class of functions (e.g. polynomial functions on a ring),
then the image of a point is given exactly. Otherwise, a \a fuzzy \a point is given,
which is a point defined using interval coefficients. A \c Point<Interval<R>> can be automatically 
converted to a \c Rectangle<R>.

Note that even if \c f is exact, it is impossible to compute 
the action of \f$f\f$ on a set just from the action of \c f on points,
unless a modulus of continuity for \f$f\f$ is known.
However, it is possible to compute the action of \f$f\f$ on sets from the action on fuzzy points.

\section real_functions Computations on real numbers.

Most continuous functions used in science and engineering are built up from elementary real-valued functions,
either arithmetical or defined using Taylor series expansions. 
For this reason, Ariadne provides extended operations for computation on real-valued functions.

Arbitrary-precision computations may be exact or approximate. 
The function \c f(x) computes \f$f(x)\f$ exactly, if possible. 
If exact computation is not possible, then \c f(x) either returns an interval containing the exact result,
or gives an error at compile time.
If a floating-point (i.e. non-interval) result is required, then the type of error must be explicitly specified.
The function \c f_down(x) computes a lower-approximation to \f$f(x)\f$, and \c f_up(x) computes an upper-approximation.
The function \c f_approx(x) computes \f$f(x)\f$ approximately, with no control on the error bound.
For some arithmetical operations, the error of \c f_approx(x,y) may be specified.

Note that in many cases, including arithmetic and simple functions, it is possible to compute an interval \f$J\f$ containing \f$f(I)\f$ 
using \c f_down and \c f_up. This allows an implementation of the standard set-based function \c f(I).
It is not, in general, possible to perform evaluation of functions on sets from their definitions on points,
though in many cases such a computation can be extracted. 
In particular, we can construct interval computations from pointwise computations in the following cases, if error bounds on \f$f(x)\f$ are known.
<ul>
   <li>The function is Lipschitz with a known Lipschitz constant e.g. \f$\sin,\ \cos\f$.</li>
   <li>The function is monotone e.g. \f$\exp\f$.</li>
   <li>The function is piecewise-monotone with known branches e.g. arithmetic.</li>
</ul>


*/
