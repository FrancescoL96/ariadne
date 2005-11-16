/***************************************************************************
 *            rectangle.h
 *
 *  Mon 2 May 2005
 *  Copyright 2005  Alberto Casagrande, Pieter Collins
 *  Email casagrande@dimi.uniud.it, Pieter.Collins@cwi.nl
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
 
/*! \file rectangle.h
 *  \brief Rectangles and cuboids.
 */

#ifndef _ARIADNE_RECTANGLE_H
#define _ARIADNE_RECTANGLE_H

#include <iosfwd>
#include <list>
#include <set>
#include <vector>
#include <valarray>

#include "ariadne.h"
#include "utility.h"
#include "interval.h"
#include "state.h"

namespace Ariadne {
  namespace Geometry {
    template < typename R > class Rectangle;
    template < typename R, template <typename> class BS > class ListSet;

    template <typename R> Rectangle<R> regular_intersection(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool interiors_intersect(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool disjoint(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool inner_subset(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool subset(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool inner_subset(const Rectangle<R> &A, const ListSet<R,Rectangle> &B);
    template<typename R> bool subset_of_open_cover(const Rectangle<R> &A, const std::vector< Rectangle<R> > &list);
    template<typename R> bool subset_of_closed_cover(const Rectangle<R> &A, const std::vector< Rectangle<R> > &list);

    template <typename R> Rectangle<R> closure_of_intersection_of_interiors(const Rectangle<R> &A, const Rectangle<R> &B);
    template<typename R> bool subset_of_interior(const Rectangle<R> &A, const Rectangle<R> &B);

    template<typename R> std::ostream& operator<<(std::ostream&, const Rectangle<R>&);
    template<typename R> std::istream& operator>>(std::istream&, Rectangle<R>&);

    /*! \brief A cuboid of arbitrary dimension.
     */
    template <typename R>
    class Rectangle {
      /*! \brief Makes intersection of interiors */
      friend Rectangle<R> regular_intersection <> (const Rectangle<R> &A,
                                                   const Rectangle<R> &B);

       /*! \brief Tests intersection of interiors. */
      friend bool interiors_intersect <> (const Rectangle<R> &A,
                                          const Rectangle<R> &B);

       /*! \brief Tests disjointness */
      friend bool disjoint <> (const Rectangle<R> &A,
                               const Rectangle<R> &B);

      /*! \brief Tests if \a A is a subset of the interior of \a B. */
      friend bool inner_subset <> (const Rectangle<R> &A,
                                   const Rectangle<R> &B);

      /*! \brief Tests inclusion. */
      friend bool subset <> (const Rectangle<R> &A,
                             const Rectangle<R> &B);

      /*! \brief Tests if \a A is a subset of the interior of \a DS. */
      // FIXME: Compiler doesn't like template parameter
      //friend bool inner_subset <> (const Rectangle<R> &A,
      //                             const ListSet<R,Rectangle> &DS);


      /*! \brief Tests inclusion in an open cover.
       *  \internal We shouldn't restrict to a std::list.
       */
      friend bool subset_of_open_cover <> (const Rectangle<R> &A,
                                           const std::vector< Rectangle<R> > &list);

      /*! \brief Tests inclusion in an closed cover.
       *  \internal We shouldn't restrict to a std::list.
       */
      friend bool subset_of_closed_cover <> (const Rectangle<R> &A,
                                             const std::vector< Rectangle<R> > &list);

      /*! \brief Tests if \a A is a subset of the interior of \a B. (deprecated)*/
      friend bool subset_of_interior <> (const Rectangle<R> &A,
                                         const Rectangle<R> &B);

      /*! \brief Makes intersection of interiors (deprecated). */
      friend Rectangle<R> closure_of_intersection_of_interiors <> (const Rectangle<R> &A,
                                                                   const Rectangle<R> &B);

     public:
      /*! \brief The type of denotable real number used for the corners. */
      typedef R Real;
      /*! \brief The type of denotable state contained by the rectangle. */
      typedef State<R> State;
     private:
      /* Rectangle's lower corner */
      State _lower_corner;
      
      /* Rectangle's upper corner */
      State _upper_corner;
      
      /* The emptyness flag */
      bool _empty;
      
     public:
      /*! \brief Default constructor construcs an empty rectangle of dimension \a n. */
      Rectangle(size_t n = 0)
        : _lower_corner(n),  _upper_corner(n), _empty(true) {}
      
      /*! \brief Construct from an array of intervals. */
      Rectangle(size_t dim, const Interval<Real>* intervals)
        : _lower_corner(dim), _upper_corner(dim), _empty(true)
      {
        for(size_t i=0; i!=dim; ++i) {
          if(intervals[i].lower() >= intervals[i].upper()) {
            this->_empty=true;
          }
        }
        this->_empty=false;
        
        for(size_t i=0; i!=dim; ++i) {
          this->_lower_corner[i] = intervals[i].lower();
          this->_upper_corner[i] = intervals[i].upper();
        }
      }
      
      /*! \brief Construct from two corners. */
      Rectangle(const State &s1, const State &s2)
        : _lower_corner(s1.dimension()), _upper_corner(s2.dimension()), _empty(true)
      {
        /* Test to see if corners have same dimensions */
        if (s1.dimension()!=s2.dimension()) {
          throw std::domain_error("The parameters have different space dimensions");
        }
        
        /* Set coordinates */
        for (size_t i=0; i<this->dimension(); i++) {
          this->_lower_corner[i]=std::min(s1[i],s2[i]);
          this->_upper_corner[i]=std::max(s1[i],s2[i]);
        }
        
        /* Set emptyness flag */
        _empty=false;
        for (size_t i=0; i<this->dimension(); i++) {
          if(this->lower(i) >= this->upper(i)) {
            _empty=true;
          }
        }
      }
      
      /*! \brief Copy constructor. */
      Rectangle(const Rectangle<R> &original)
        : _lower_corner(original._lower_corner),
          _upper_corner(original._upper_corner),
          _empty(original._empty) {
      }
      
      /*! \brief Copy assignment operator. */
      Rectangle<R>& operator=(const Rectangle<R>& original) {
        if(this != &original) {
          this->_lower_corner = original._lower_corner;
          this->_upper_corner = original._upper_corner;
          this->_empty = original._empty;
        }
        return *this;
      }
      
      /*! \brief The dimension of the Euclidean space the rectangle lies in. (Deprecated) */
      inline size_t dim() const {
        return (this->_lower_corner).dimension();
      }
      
      /*! \brief The dimension of the Euclidean space the rectangle lies in. */
      inline size_t dimension() const {
        return (this->_lower_corner).dimension();
      }
      
      /*! \brief Returns \c true if the rectangle is empty */
      inline bool empty() const {
        return (this->_empty);
      }
      
      /*! \brief The lower corner. */
      inline State lower_corner() const {
        return this->_lower_corner;
      }
      
      /*! \brief The upper corner. */
      inline State upper_corner() const {
        return this->_upper_corner;
      }
      
      /*! \brief Returns the projection onto the \a n th coordinate. */
      inline Interval<Real> interval(size_t n) const {
        return Interval<Real>(this->_lower_corner[n],this->_upper_corner[n]);
      }
      
      /*! \brief Returns the projection onto the \a n th coordinate. */
      inline Interval<Real> operator[] (size_t n) const {
        return Interval<Real>(this->_lower_corner[n],this->_upper_corner[n]);
      }
      
      /*! \brief Gets the \a n th interval. */
      inline Interval<Real> get(size_t n) const {
        return Interval<Real>(this->_lower_corner[n],this->_upper_corner[n]);
      }
      
      /*! \brief Sets the \a n th interval. */
      inline void set(size_t n, Interval<Real> i) {
         this->_lower_corner[n]=i.lower();
         this->_upper_corner[n]=i.upper();
      }
      
      /*! \brief Sets the lower bound of the \a n th coordinate to \a r. */
      inline void set_lower(size_t n, const Real& r) {
        this->_lower_corner[n] = r;
      }
      
      /*! \brief Sets the upper bound of the \a n th coordinate to \a r. */
      inline void set_upper(size_t n, const Real& r) {
        this->_upper_corner[n] = r;
      }
      
      /*! \brief Returns the lower bound of the \a n th coordinate */
      inline Real lower(size_t n) const {
        return this->_lower_corner[n];
      }
      
      /*! \brief Returns the upper bound of the \a n th coordinate */
      inline Real upper(size_t n) const {
        return this->_upper_corner[n];
      }
      
      /*! \brief Tests if \a state is included into a rectangle. */
      inline bool contains(const State& state) const {
        
        if (state.dimension()!=this->dimension())
          throw std::domain_error("This object and parameter have different space dimensions");
        
        if (this->empty()) return false;
        
        /* for each dimension i */
        for (size_t i=0; i<this->dimension(); i++) {
          /* if the i dim of the state is smaller than the one of the
           * rectangle's lower corner then state is not contained into
           * this object */
          if (state[i] < this->_lower_corner[i]) return false;
          
          
          /* if the i dim of the state is greater than the one of the
           * rectangle's upper corner then state is not contained into
           * this object */
          if (state[i] > this->_upper_corner[i]) return false;
        }
        
        return true;
        
      }
      
      /*! \brief Tests if \a state is included into the interior a rectangle. */
      inline bool interior_contains(const State& state) const {
        
        if (state.dimension()!=this->dimension())
          throw std::domain_error("This object and parameter have different space dimensions");
        
        if (this->empty()) return false;
        
        /* for each dimension i */
        for (size_t i=0; i<this->dimension(); i++) {
          
          /* if the i dim of the state is greater or equal than the one
           * of the rectangle's upper corner then state is not contained 
           * in the interior of this object */
          if (state[i] >= this->_upper_corner[i]) return false;
          
          /* if the i dim of the state is smaller or equal than the one 
           * of the rectangle's lower corner then state is not contained
           * in the interior of this object */
          if (state[i] <= this->_lower_corner[i]) return false;
        }
        
        return true;
        
      }
      
      /*! \brief Compute a quadrant of the Rectangle determined by \a q. FIXME: use binary words.
       *
       *  \a q is an integer such that the ith bit of q is 0 if the lower half
       *  of the rectangle in the ith coordinate is used, and 1 if the upper
       *  half is used.
       */
      inline Rectangle<R> find_quadrant(size_t q) const {
        
        size_t j;
        
        Rectangle<R> quadrant(this->dimension());
        
        for (j=0; j< this->dimension(); j++) {
          
          if (q%2) {
            quadrant._lower_corner[j]=(this->_upper_corner[j]+
                                       this->_lower_corner[j])/2;
            quadrant._upper_corner[j]=this->_upper_corner[j];
          } else {
            quadrant._upper_corner[j]=(this->_upper_corner[j]+
                                       this->_lower_corner[j])/2;
            quadrant._lower_corner[j]=this->_lower_corner[j];
          }
          q=q/2;
        }
        
        return quadrant;
      }
      
      /*! \brief Expand the Rectangle by \a delta in each direction. */
      inline Rectangle<R> &expand_by(const Real &delta) {
        
        for (size_t j=0; j< this->dimension(); ++j) {
          
          this->_upper_corner[j]+=delta;
          this->_lower_corner[j]-=delta;
        }
        
        return *this;
      }
      
      /*! \brief The equality operator */
      inline bool operator==(const Rectangle<Real> &A) const
      {
        if (this->dimension() != A.dimension()) return false ;
        
        if (A.empty() && this->empty()) { return true; }
        if (A.empty() || this->empty()) { return false; }
        
        for (size_t j=0; j != this->dimension(); ++j) {
          if (this->_lower_corner[j] != A._lower_corner[j]) { return false; }
          if (this->_upper_corner[j] != A._upper_corner[j]) { return false; }
        }
        
        return true;
      }
      
      /*! \brief The inequality operator */
      inline bool operator!=(const Rectangle<Real> &A) const {
        return !(*this == A);
      }

      /*! \brief Tests if the Rectangle is disjoint from the set \a s. */
      template<class SET>
      inline bool disjoint(const SET& s) const {
        return Ariadne::Geometry::disjoint(*this,s);
      }
      
      /*! \brief Tests if the Rectangle intersects the interior of the set \a s. */
      template<class SET>
      inline bool intersects_interior(const SET& s) const {
        return Ariadne::Geometry::interiors_intersect(*this,s);
      }
      
      /*! \brief Tests if the Rectangle is a subset of the interior of set \a s. */
      template<class SET>
      inline bool inner_subset(const SET& s) const {
        return Ariadne::Geometry::subset_of_interior(*this,s);
      }

      /*! \brief Tests if the Rectangle is a subset of the set \a s. */
      template<class SET>
      inline bool subset(const SET& s) const {
        return Ariadne::Geometry::subset(*this,s);
      }

      /*! \brief Tests if the Rectangle is a subset of the interior of set \a s. (Deprecated) */
      template<class SET>
      inline bool is_subset_of_interior_of(const SET& s) const {
        return Ariadne::Geometry::subset_of_interior(*this,s);
      }

      /*! \brief Tests if the Rectangle is a subset of the the union of the open sets in \a u. */
      template<class LIST>
      inline bool subset_of_open_cover(const LIST& u) const {
        return Ariadne::Geometry::subset_of_open_cover(*this,u);
      }
      
      /*! \brief Tests if the Rectangle is a subset of the the union of the closed sets in \a u. */
      template<class LIST>
      inline bool subset_of_open_closed_cover(const LIST& u) const {
        return Ariadne::Geometry::subset_of_closed_cover(*this,u);
      }
      
      friend std::ostream&
      operator<< <> (std::ostream &os, 
                     const Rectangle<R> &r);
      
      friend std::istream&
      operator>> <> (std::istream &is, 
                     Rectangle<R> &r);
      
    };
    
    /*! \brief Tests disjointness */
    template <typename R>
    bool disjoint(const Rectangle<R> &A, const Rectangle<R> &B) {

      if (A.dimension()!=B.dimension())
        throw std::domain_error("The two parameters have different space dimensions");
      
      
      for (size_t i=0; i< A.dimension(); i++) {
        if ((A._upper_corner[i]<B._lower_corner[i])|| 
            (B._upper_corner[i]<A._lower_corner[i])) return true;
      }
      
      return false;
    }
    
    /*! \brief Tests intersection of interiors */
    template <typename R>
    bool interiors_intersect(const Rectangle<R> &A,
                             const Rectangle<R> &B) {
      
      if (A.dimension()!=B.dimension()) 
        throw std::domain_error("The two parameters have different space dimensions");
      
      if (A.empty()||B.empty()) return false;
      
      for (size_t i=0; i< A.dimension(); i++) {
        if ((A._upper_corner[i]<=B._lower_corner[i])|| 
            (B._upper_corner[i]<=A._lower_corner[i])) return false;
      }
      
      return true;
    }
    
    
    /*! \brief Tests inclusion of \a A in the interior of \a B. */
    template <typename R>
    bool inner_subset(const Rectangle<R> &A,
                      const Rectangle<R> &B) {

      if (A.dimension()!=B.dimension())
        throw std::domain_error("The two parameters have different space dimensions");

      if (A.empty()||B.empty()) return false;

      for (size_t i=0; i< A.dimension(); i++) {
        if ((A._upper_corner[i] >= B._upper_corner[i])||
            (B._lower_corner[i] >= A._lower_corner[i])) return false;
      }

      return true;
    }

    /*! \brief Tests inclusion */
    template <typename R>
    bool subset(const Rectangle<R> &A, 
                const Rectangle<R> &B) {
      
      if (A.dimension()!=B.dimension())
        throw std::domain_error("The two parameters have different space dimensions");
      
      if (A.empty()||B.empty()) return false;
      
      for (size_t i=0; i< A.dimension(); i++) {
        if ((A._upper_corner[i] > B._upper_corner[i])||
            (B._lower_corner[i] > A._lower_corner[i])) return false;
      }
      
      return true;
    }
    
    /*! \brief Tests inclusion of \a A in the interior of \a B. */
    template <typename R>
    bool subset_of_interior(const Rectangle<R> &A,
                            const Rectangle<R> &B)
    {
      return inner_subset(A,B);
    }

    /* Compute all points in A on the grid of vertices of rectangles in the cover */
    template <typename R>
    void
    compute_gridpoints(std::vector< std::set<R> >& gridpoints,
                       const Rectangle<R> &A, 
                       const std::vector< Rectangle<R> >& cover)
    {
      typedef R Real;
      typedef typename std::set<Real> Set;
      typedef typename std::vector< Rectangle<R> >::const_iterator list_iterator;
      
      size_t dimension = A.dimension();
      
      for(size_t i=0; i!=dimension; ++i) {
        Real lower=A.lower(i);
        Real upper=A.upper(i);
        gridpoints[i].insert(lower);
        gridpoints[i].insert(upper);
        for(list_iterator rect=cover.begin(); rect!=cover.end(); ++rect) {
          Real bound = rect->lower(i);
          if(lower<bound && bound<upper) {
            gridpoints[i].insert(bound);
          }
          bound = rect->upper(i);
          if(lower<bound && bound<upper) {
            gridpoints[i].insert(bound);
          }
        }
      }
      
#ifdef DEBUG
      std::cerr << "Gridpoints: " << gridpoints << '\n';
#endif
    }
    
    //*! \brief Tests inclusion in an open cover.  */
    template <typename R>
    bool subset_of_open_cover(const Rectangle<R> &A,
                              const std::vector< Rectangle<R> >& cover) 
    {
      typedef R Real;
      typedef typename std::set<Real> Set;
      typedef typename Set::const_iterator set_iterator;
      typedef typename std::vector< Rectangle<R> >::const_iterator list_iterator;
      
      size_t dimension = A.dimension();
      
      std::vector<Set> gridpoints(dimension);
      compute_gridpoints(gridpoints, A, cover);
      
      /* This is a hack. We really need a "grid" class based on binary words. */
      typedef std::valarray<size_t> index_type;
      
      /* Whether the jth gridpoint (in some ordering) is covered */
      std::vector<bool> cover_flags;
      
      /* Strides for indexing */
      index_type strides(dimension);
      size_t stride=1;
      for(size_t i=0; i!=dimension; ++i) {
        strides[i]=stride;
        stride*=gridpoints[i].size();
      }
      cover_flags.resize(stride);
      
      std::vector<size_t> lower_indices(dimension+1);
      std::vector<size_t> upper_indices(dimension+1);
      lower_indices[dimension] = 0;
      upper_indices[dimension] = 2;
      
      for(list_iterator rect=cover.begin(); rect!=cover.end(); ++rect) {
        for(size_t i=0; i!=dimension; ++i) {
          Real lower_bound=rect->lower(i);
          size_t lower_index=0;
          set_iterator iter=gridpoints[i].begin();
          while(iter!=gridpoints[i].end() && (*iter) <= lower_bound) {
            ++iter;
            ++lower_index;
          }
          lower_indices[i]=lower_index;
          
          Real upper_bound=rect->upper(i);
          size_t upper_index=0;
          iter=gridpoints[i].begin();
          while(iter!=gridpoints[i].end() && (*iter) < upper_bound) {
            ++iter;
            ++upper_index;
          }
          upper_indices[i]=upper_index;
        }
        
#ifdef DEBUG
        std::cerr << "Rectangle: " <<  (*rect)
                  << " lower_indices: " << lower_indices
                  << " upper_indices: " << upper_indices << '\n';
#endif
        
        index_type index(dimension+1);
        for(size_t i=0; i!=dimension; ++i) {
          index[i] = lower_indices[i];
        }
        index[dimension]=0;
        
        while(index[dimension] != 1) {
#ifdef DEBUG
          std::cerr << index << " " << upper_indices << "\n";
#endif
          
          size_t entry = 0;
          for(size_t j=0; j!=dimension; ++j) {
            entry += index[j]*strides[j];
          }
          cover_flags[entry] = true;
          
          size_t inc=0;
          ++(index[inc]);
          while(index[inc] == upper_indices[inc]) {
            index[inc]=0;
            ++inc;
            ++(index[inc]);
          }
        }
        
#ifdef DEBUG
        std::cerr << index << "\n\n";
        std::cerr << cover_flags << '\n';
#endif
      }
      
      for( std::vector<bool>::const_iterator flag = cover_flags.begin();
           flag != cover_flags.end(); ++flag) {
        if(*flag == false) {
          return false;
        }
      }
      
      return true;
    }
    
    //*! \brief Tests inclusion in a closed cover.  */
    template <typename R>
    bool subset_of_closed_cover(const Rectangle<R> &A,
                                const std::vector< Rectangle<R> >& cover) 
    {
      typedef typename Rectangle<R>::Real Real;
      typedef typename std::set<Real> Set;
      typedef typename Set::const_iterator set_iterator;
      typedef typename std::vector< Rectangle<R> >::const_iterator list_iterator;
      
      size_t dimension = A.dimension();
      
      std::vector<Set> gridpoints(dimension);
      compute_gridpoints(gridpoints, A, cover);
      
      /* This is a hack. We really need a "grid" class based on binary words. */
      typedef std::valarray<size_t> index_type;
      
      /* Whether the jth gridpoint (in some ordering) is covered */
      std::vector<bool> cover_flags;
      
      /* Strides for indexing */
      index_type strides(dimension);
      size_t stride=1;
      for(size_t i=0; i!=dimension; ++i) {
        strides[i]=stride;
        stride*=gridpoints[i].size();
      }
      cover_flags.resize(stride);
      
      std::vector<size_t> lower_indices(dimension+1);
      std::vector<size_t> upper_indices(dimension+1);
      lower_indices[dimension] = 0;
      upper_indices[dimension] = 2;
      
      for(list_iterator rect=cover.begin(); rect!=cover.end(); ++rect) {
        for(size_t i=0; i!=dimension; ++i) {
          Real lower_bound=rect->lower(i);
          size_t lower_index=0;
          set_iterator iter=gridpoints[i].begin();
          while(iter!=gridpoints[i].end() && (*iter) < lower_bound) {
            ++iter;
            ++lower_index;
          }
          lower_indices[i]=lower_index;
          
          Real upper_bound=rect->upper(i);
          size_t upper_index=0;
          iter=gridpoints[i].begin();
          while(iter!=gridpoints[i].end() && (*iter) <= upper_bound) {
            ++iter;
            ++upper_index;
          }
          upper_indices[i]=upper_index;
        }
        
#ifdef DEBUG
        std::cerr << "Rectangle: " <<  (*rect)
                  << " lower_indices: " << lower_indices
                  << " upper_indices: " << upper_indices << '\n';
#endif
        
        index_type index(dimension+1);
        for(size_t i=0; i!=dimension; ++i) {
          index[i] = lower_indices[i];
        }
        index[dimension]=0;
        
        while(index[dimension] != 1) {
#ifdef DEBUG
          std::cerr << index << " " << upper_indices << "\n";
#endif
          
          size_t entry = 0;
          for(size_t j=0; j!=dimension; ++j) {
            entry += index[j]*strides[j];
          }
          cover_flags[entry] = true;
          
          size_t inc=0;
          ++(index[inc]);
          while(index[inc] == upper_indices[inc]) {
            index[inc]=0;
            ++inc;
            ++(index[inc]);
          }
        }
        
#ifdef DEBUG
        std::cerr << index << "\n\n";
        std::cerr << cover_flags << '\n';
#endif
        
      }
      
      for( std::vector<bool>::const_iterator flag = cover_flags.begin(); 
           flag != cover_flags.end(); ++flag) {
        if(*flag == false) {
          return false;
        }
      }
      
      return true;
    }
    
    
    /*! \brief Makes intersection of interior */
    template <typename R>
    Rectangle<R>
    regular_intersection(const Rectangle<R> &A, const Rectangle<R> &B)
    {
      if (A.dimension()!=B.dimension()) {
        throw std::domain_error("The two parameters have different space dimensions");
      }

      Rectangle<R> C(A.dimension());

      if (A.empty() || B.empty()) {
        return C;
      }

      for (size_t i=0; i != C.dimension(); ++i) {
        C._lower_corner[i] = std::max(A._lower_corner[i],B._lower_corner[i]);
        C._upper_corner[i] = std::min(A._upper_corner[i],B._upper_corner[i]);
        if(C._lower_corner[i] >= C._upper_corner[i]) {
          C._empty=true;
          return C;
        }
      }

      C._empty=false;
      return C;
    }

    /*! \brief Makes intersection of interior */
    template <typename R>
    Rectangle<R>
    closure_of_intersection_of_interiors(const Rectangle<R> &A, const Rectangle<R> &B) {
      return regular_intersection(A,B);
    }


    template <typename R>
    std::ostream&
    operator<<(std::ostream &os, const Rectangle<R> &r) 
    {
      
      /*
        os << "{ lower=" << (r._lower_corner) << ", " ;
        os << "upper=" << (r._upper_corner) << " }" ;
      */
      
      /*
        if(r.empty()) {
        return os << "[ ]";
        }
      */
      if(r.dimension() > 0) {
        os << r[0];
        for(size_t i=1; i!=r.dimension(); ++i) {
          os << "x" << r[i];
        }
      }

      return os;
    }
    
    template <typename R>
    std::istream& 
    operator>>(std::istream &is, Rectangle<R> &r)
    {
      typedef typename Rectangle<R>::Real Real;
      typedef typename Ariadne::Interval<Real> Interval;
      
      char c;
      is >> c;
      is.putback(c);
      if(c=='[') {
        /* Representation as list of intervals */ 
        std::vector< Interval > v;
        is >> v;
        r=Rectangle<R>(v.size(),&v[0]);
      }
      else {
        /* representation as lower and upper corners */
        /* FIXME */
        // throw invalid_input("Not implemented");
      }
      return is;
    }
    
  }
}

#endif /* _ARIADNE_RECTANGLE_H */
