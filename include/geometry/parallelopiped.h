/***************************************************************************
 *            parallelopiped.h
 *
 *  6 January 2006
 *  Copyright  2006  Alberto Casagrande, Pieter Collins
 *  casagrande@dimi.uniud.it, pieter.collins@cwi.nl
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
 *  along with this program; if not, write to bouthe Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */
 
/*! \file parallelopiped.h
 *  \brief Parallelopipeds.
 */

#ifndef _ARIADNE_PARALLELOPIPED_H
#define _ARIADNE_PARALLELOPIPED_H

#include <iosfwd>

#include <string>
#include <sstream>

#include <list>
#include <set>
#include <vector>
#include <valarray>

#include "base/utility.h"
#include "base/interval.h"

#include "linear_algebra/vector.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/constraint.h"

#include "geometry/point.h"
#include "geometry/rectangle.h"
#include "geometry/list_set.h"
#include "geometry/polyhedron.h"
#include "geometry/geometry_declarations.h"

namespace Ariadne {
  namespace Geometry {
    template < typename R > class Parallelopiped;

    template < typename R > class Rectangle;
    template < typename R > class Polyhedron;
    template < typename R, template <typename> class BS > class ListSet;

    template <typename R> Parallelopiped<R> intersection(const Parallelopiped<R>& A, const Parallelopiped<R>& B);
    template <typename R> Parallelopiped<R> regular_intersection(const Parallelopiped<R>& A, const Parallelopiped<R>& B);

    template<typename R> bool interiors_intersect(const Parallelopiped<R>& A, const Parallelopiped<R>& B);
    template<typename R> bool disjoint(const Parallelopiped<R>& A, const Parallelopiped<R>& B);
    template<typename R> bool inner_subset(const Parallelopiped<R>& A, const Parallelopiped<R>& B);
    template<typename R> bool subset(const Parallelopiped<R>& A, const Parallelopiped<R>& B);

    template<typename R> bool subset_of_open_cover(const Parallelopiped<R>& A, const ListSet<R, Parallelopiped >& list);
    template<typename R> bool inner_subset(const Parallelopiped<R>& rect, const ListSet<R,Parallelopiped>& A);
    template<typename R> bool subset(const Parallelopiped<R>& rect, const ListSet<R,Parallelopiped>& A);
    
    template<typename R> std::ostream& operator<<(std::ostream&, const Parallelopiped<R>&);
    template<typename R> std::istream& operator>>(std::istream&, Parallelopiped<R>&);

    /*! \brief A parallelopiped of arbitrary dimension.
     */
    template <typename R>
    class Parallelopiped {
      /*! \brief Makes intersection */
      friend Parallelopiped<R> intersection <> (const Parallelopiped<R>& A,
                                           const Parallelopiped<R>& B);

      /*! \brief Makes intersection of interiors */
      friend Parallelopiped<R> regular_intersection <> (const Parallelopiped<R>& A,
                                                   const Parallelopiped<R>& B);

       /*! \brief Tests intersection of interiors. */
      friend bool interiors_intersect <> (const Parallelopiped<R>& A,
                                          const Parallelopiped<R>& B);

       /*! \brief Tests disjointness */
      friend bool disjoint <> (const Parallelopiped<R>& A,
                               const Parallelopiped<R>& B);

      /*! \brief Tests if \a A is a subset of the interior of \a B. */
      friend bool inner_subset <> (const Parallelopiped<R>& A,
                                   const Parallelopiped<R>& B);

      /*! \brief Tests inclusion. */
      friend bool subset <> (const Parallelopiped<R>& A,
                             const Parallelopiped<R>& B);

      /*! \brief Tests if \a A is a subset of the interior of \a B. */
      friend bool inner_subset <> (const Parallelopiped<R>& A,
                                   const ListSet<R,::Ariadne::Geometry::Parallelopiped>& B);

      /*! \brief Tests if \a A is a subset of \a B. */
      friend bool subset <> (const Parallelopiped<R>& A,
                             const ListSet<R,::Ariadne::Geometry::Parallelopiped>& B);


      /*! \brief Tests inclusion in an open cover, represented as a ListSet.
       */
      friend bool subset_of_open_cover <> (const Parallelopiped<R>& A,
                                           const ListSet<R,::Ariadne::Geometry::Parallelopiped>& B);

     public:
      /*! \brief The unsigned integer type used to denote temphe array positions. */
      typedef size_t size_type;
      /*! \brief The type of denotable real number used for the corners. */
      typedef R Real;
      /*! \brief The type of denotable state contained by the rectangle. */
      typedef Point<R> State;
      /*! \brief The type of matrix giving principal directions. */
      typedef ::Ariadne::LinearAlgebra::vector<R> Vector;
      /*! \brief The type of matrix giving principal directions. */
      typedef ::Ariadne::LinearAlgebra::matrix<R> Matrix;
     private:
      /* Parallelopiped's centre. */
      State _centre;
      
      /* Parallelopiped's principal directions. */
      Matrix _generators;
      
     public:
      /*! \brief Default constructor constructs an empty parallelopiped of dimension \a n. */
      inline explicit Parallelopiped(size_type n = 0)
        : _centre(n),  _generators(n,n) 
      {
      }
      
      /*! \brief Construct from centre and directions. */
      inline explicit Parallelopiped(const State& c, const Matrix& m)
        : _centre(c), _generators(m)
      {
        if (LinearAlgebra::number_of_rows(m)!=LinearAlgebra::number_of_columns(m)) {
          throw std::domain_error(
              "The the matrix of principal directions is not a square matrix");
        }
        
        if (c.dimension()!=LinearAlgebra::number_of_rows(m)) {
          throw std::domain_error("The centre and directions have different dimensions.");
        }
      }
      
      /*! \brief Construct from a Rectangle. */
      inline explicit Parallelopiped(const Rectangle<Real>& r)
        : _centre(r.dimension()), _generators(r.dimension(),r.dimension())
      {
        for(size_type i=0; i!=dimension(); ++i) {
          _centre[i] = (r.lower_bound(i)+r.upper_bound(i))/2;
          _generators(i,i) = (r.upper_bound(i)-r.lower_bound(i))/2;
        }
      }
      
      /*! \brief Construct from a string literal. */
      inline explicit Parallelopiped(const std::string& s)
        : _centre(), _generators()
      {
        std::stringstream ss(s);
        ss >> *this;
      }
      
      /*! \brief Copy constructor. */
      Parallelopiped(const Parallelopiped<R>& original)
        : _centre(original._centre),
          _generators(original._generators)
      { }
      
      /*! \brief Copy assignment operator. */
      Parallelopiped<R>& operator=(const Parallelopiped<R>& original) {
        if(this != &original) {
          this->_centre = original._centre;
          this->_generators = original._generators;
        }
        return *this;
      }
      
      /*! \brief The dimension of the Euclidean space the parallelopiped lies in. */
      inline size_type dimension() const {
        return (this->_centre).dimension();
      }
      
      /*! \brief True if the parallelopiped is empty. */
      inline bool empty() const {
        // FIXME: This is probably ok since we're not talking about interior.
        return false;
      }
      
      /*! \brief True if the parallelopiped has empty interior. */
      inline bool empty_interior() const {
        throw std::domain_error("Parallelopiped::empty_interior() not implemented.");
      }
      
      /*! \brief The centre of the parallelopiped. */
      inline State centre() const {
        return this->_centre;
      }
      
      /*! \brief The \a n th of principle direction. */
      inline Vector generator(size_type n) const {
        return column(this->_generators,n);
      }
      
      /*! \brief The matrix of principle directions. */
      inline Matrix generators() const {
        return this->_generators;
      }
      
      /*! \brief Tests if the parallelopiped contains \a state. */
      inline bool contains(const State& state) const {
        if (state.dimension()!=this->dimension()) {
          throw std::domain_error("This object and parameter have different space dimensions");
        }  
        
        if (this->empty()) { return false; }
          
        Vector v=coordinates(state);
        for (size_type i=0; i<v.size(); ++i) {
          if(v[i]<-1 || v[i]>+1) {
            return false;
          }
        }
      
        return true;
      }
      
      /*! \brief Tests if the interior of the parallelopiped contains \a state. */
      inline bool interior_contains(const State& state) const {
        if (state.dimension()!=this->dimension()) {
         throw std::domain_error("This object and parameter have different space dimensions");
        } 
      
        if (this->empty()) { return false; }
        
        Vector v=coordinates(state);
        for (size_type i=0; i<v.size(); ++i) {
          if(v[i]<=-1 || v[i]>=+1) {
            return false;
          }
        }
      
        return true;
      }
      
      /*! \brief The equality operator (not implemented).
       *
       * Not currently implemented, since it requires matching the columns of 
       * the matrix of principal directions. 
       */
      inline bool operator==(const Parallelopiped<Real>& A) const
      {
        throw std::domain_error("Parallelopiped::operator==(...)  not implemented");
      }
      
      /*! \brief The inequality operator */
      inline bool operator!=(const Parallelopiped<Real>& A) const {
        throw std::domain_error("Parallelopiped::operator!=(...)  not implemented");
        return !(*this == A);
      }

      /*! \brief A rectangle containing the given parallelopiped. */
      Rectangle<R> bounding_box() const;
      
      /*! \brief Convert to a polyhedron. */
      operator Polyhedron<R> () const;
      
      /*! \subdivide into smaller pieces. */
      ListSet<R,::Ariadne::Geometry::Parallelopiped> subdivide() const;
      
      friend std::ostream&
      operator<< <> (std::ostream& os, 
                     const Parallelopiped<R>& r);
      
      friend std::istream&
      operator>> <> (std::istream& is, 
                     Parallelopiped<R>& r);
      
     private:
      void compute_linear_inequalities(Matrix&, Vector&, Vector&) const;
      Vector coordinates(const State& s) const;
    };
  
    /*! \brief Tests disjointness */
    template <typename R>
    bool disjoint(const Parallelopiped<R>& A, const Parallelopiped<R>& B) 
    {
      return disjoint(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    /*! \brief Tests disjointness */
    template <typename R>
    bool disjoint(const Parallelopiped<R>&A, const Rectangle<R>& B) 
    {
      return disjoint(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    /*! \brief Tests disjointness */
    template <typename R>
    bool disjoint(const Rectangle<R>&A, const Parallelopiped<R>& B) 
    {
      return disjoint(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    
    /*! \brief Tests intersection of interiors */
    template <typename R>
    bool interiors_intersect(const Parallelopiped<R>& A,
                             const Parallelopiped<R>& B) 
    {
      return interiors_intersect(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    /*! \brief Tests intersection of interiors */
    template <typename R>
    bool interiors_intersect(const Parallelopiped<R>& A,
                             const Rectangle<R>& B) 
    {
      return interiors_intersect(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    /*! \brief Tests intersection of interiors */
    template <typename R>
    bool interiors_intersect(const Rectangle<R>& A,
                             const Parallelopiped<R>& B) 
    {
      return interiors_intersect(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    
    
    /*! \brief Tests inclusion of \a A in the interior of \a B. */
    template <typename R>
    bool inner_subset(const Parallelopiped<R>& A,
                      const Parallelopiped<R>& B) 
    {
      return inner_subset(Polyhedron<R>(A), Polyhedron<R>(B));
    }

    /*! \brief Tests inclusion of \a A in the interior of \a B. */
    template <typename R>
    bool inner_subset(const Parallelopiped<R>& A,
                      const Rectangle<R>& B) 
    {
      return inner_subset(Polyhedron<R>(A), Polyhedron<R>(B));
    }

    /*! \brief Tests inclusion of \a A in the interior of \a B. */
    template <typename R>
    bool inner_subset(const Rectangle<R>& A,
                      const Parallelopiped<R>& B) 
    {
      return inner_subset(Polyhedron<R>(A), Polyhedron<R>(B));
    }

    /*! \brief Tests inclusion */
    template <typename R>
    bool subset(const Parallelopiped<R>& A, 
                const Parallelopiped<R>& B) 
    {
      return subset(Polyhedron<R>(A), Polyhedron<R>(B));
    }
    

    /*! \brief Tests inclusion in an open cover.  */
    template <typename R>
    bool subset_of_open_cover(const Parallelopiped<R>& A,
                              const ListSet<R, Parallelopiped >& cover) 
    {
      throw std::domain_error("subset_of_open_cover(Parallelopiped, std::vector<Parallelopiped>) not implemented");
    }

    
    /*! \brief Tests inclusion of \a A om the interior of \a B. */
    template <typename R>
    bool inner_subset(const Parallelopiped<R>& A,
                      const ListSet<R,Parallelopiped>& B) 
    {
      throw std::domain_error("subset_of_closed_cover(Parallelopiped, std::vector<Parallelopiped>) not implemented");
    }



    

  }
}

#endif /* _ARIADNE_PARALLELOPIPED_H */
