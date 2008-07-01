/***************************************************************************
 *            approximator_base.h
 *
 *  Copyright  2008  Pieter Collins
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
 
/*! \file approximator_base.h
 *  \brief %Base class for approximators implementing standard methods which can be implemented in terms of others. 
 */

#ifndef ARIADNE_APPROXIMATOR_BASE_H
#define ARIADNE_APPROXIMATOR_BASE_H

#include "base/types.h"
#include "base/declarations.h"
#include "numeric/declarations.h"
#include "linear_algebra/declarations.h"
#include "geometry/declarations.h"

#include "geometry/list_set.h"
#include "evaluation/approximator_interface.h"


namespace Ariadne {
  

    /* \brief %Base class for approximators implementing standard methods which can be implemented in terms of others. */
    template<class Aprx, class ES> 
    class ApproximatorBase
      : public ApproximatorInterface<Aprx,ES> 
    { 
      typedef typename Aprx::real_type R;
      typedef typename Aprx::BasicSet BasicSet;
      typedef typename Aprx::Paving Paving;
      typedef typename Aprx::CoverListSet CoverListSet;
      typedef typename Aprx::PartitionListSet PartitionListSet;
      typedef typename Aprx::PartitionTreeSet PartitionTreeSet;
      typedef ES EnclosureSet;
      typedef ListSet<ES> EnclosureSetList;
     protected:
      ApproximatorBase(const Paving& pv) : _grid_spacing(pv.lengths()[0]) { }
      ApproximatorBase(const R& l) : _grid_spacing(l) { }
     public:
      /*! \brief Computes a bounding box for a set. */
      BasicSet bounding_box(const EnclosureSetList& esl) const;

      /*! \brief Computes an lower-approximation of a set. */
      CoverListSet lower_approximation(const EnclosureSetList& esl) const;

      /*! \brief Computes an inner-approximation of a set. */
      PartitionListSet inner_approximation(const EnclosureSet& es) const;

      /*! \brief Computes an outer-approximation of a set. */
      PartitionListSet outer_approximation(const EnclosureSet& es) const;

      /*! \brief Computes an inner-approximation of a set. */
      PartitionListSet inner_approximation(const EnclosureSetList& esl) const;

      /*! \brief Computes an outer-approximation of a set. */
      PartitionListSet outer_approximation(const EnclosureSetList& esl) const;

      /*! \brief Computes an outer-approximation of a set on a given paving. */
      PartitionListSet outer_approximation(const EnclosureSetList& esl, const Paving& pv) const;


      /*! \brief Adjoins an outer approximation of a basic set to a grid set. */
      void adjoin_outer_approximation(PartitionListSet& pls, const EnclosureSet& es) const;

      /*! \brief Adjoins an outer approximation to a basic set to a grid mask set. */
      void adjoin_outer_approximation(PartitionTreeSet& pts, const EnclosureSet& es) const;


      /*! \brief Computes over-approximations of the sets. */
      void adjoin_over_approximations(CoverListSet& cls, const EnclosureSetList& esl) const;

      /*! \brief Computes over-approximations of the sets. */
      void adjoin_outer_approximation(PartitionListSet& pls, const EnclosureSetList& esl) const;

      /*! \brief Computets and over-approximation of a set from a rectangle. */
      void adjoin_outer_approximation(PartitionTreeSet& pts, const EnclosureSetList& esl) const;
     protected:
      Paving paving(uint d) const { return Paving(d,this->_grid_spacing); }
     private:
      R _grid_spacing;
   };



template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::BasicSet
ApproximatorBase<Aprx,ES>::bounding_box(const EnclosureSetList& esl) const
{
  if(esl.size()==0) { 
    return BasicSet(esl.dimension());
  } 
  
  const ApproximatorInterface<Aprx,ES>* base=this;
  BasicSet bb=base->bounding_box(esl[0]);
  for(size_type i=1; i!=esl.size(); ++i) {
    bb=rectangular_hull(bb,base->bounding_box(esl[i]));
  }
  return bb;
}

template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::CoverListSet
ApproximatorBase<Aprx,ES>::lower_approximation(const EnclosureSetList& esl) const
{
  CoverListSet cls;
  const ApproximatorInterface<Aprx,ES>* base=this;
  for(size_type i=0; i!=esl.size(); ++i) {
    cls.adjoin(base->lower_approximation(esl[i]));
  }
  return cls;
}

  
template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::PartitionListSet
ApproximatorBase<Aprx,ES>::inner_approximation(const EnclosureSet& es) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  return base->inner_approximation(es,this->paving(es.dimension()));
}

template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::PartitionListSet
ApproximatorBase<Aprx,ES>::outer_approximation(const EnclosureSet& es) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  return base->inner_approximation(es,this->paving(es.dimension()));
}

template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::PartitionListSet
ApproximatorBase<Aprx,ES>::inner_approximation(const EnclosureSetList& esl) const
{
  return this->inner_approximation(esl,this->paving(esl.dimension()));
}

template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::PartitionListSet
ApproximatorBase<Aprx,ES>::outer_approximation(const EnclosureSetList& esl) const
{
  return this->outer_approximation(esl,this->paving(esl.dimension()));
}

template<class Aprx, class ES> inline
typename ApproximatorBase<Aprx,ES>::PartitionListSet
ApproximatorBase<Aprx,ES>::outer_approximation(const EnclosureSetList& esl, const Paving& pv) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  PartitionListSet pls(pv);
  for(size_type i=0; i!=esl.size(); ++i) {
    pls.adjoin(base->outer_approximation(esl[i],pv));
  }
  pls.unique_sort();
  return pls;
}

  
template<class Aprx, class ES> inline
void
ApproximatorBase<Aprx,ES>::adjoin_outer_approximation(PartitionTreeSet& pts, 
                                                      const EnclosureSet& es) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  pts.adjoin(base->outer_approximation(es,pts.grid()));
}


template<class Aprx, class ES> inline
void
ApproximatorBase<Aprx,ES>::adjoin_outer_approximation(PartitionListSet& pls, 
                                                      const EnclosureSet& es) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  pls.adjoin(base->outer_approximation(es,pls.grid()));
}


template<class Aprx, class ES> inline
void
ApproximatorBase<Aprx,ES>::adjoin_over_approximations(CoverListSet& cls, 
                                                      const EnclosureSetList& esl) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  cls.adjoin(base->bounding_box(esl));
}

template<class Aprx, class ES> inline
void
ApproximatorBase<Aprx,ES>::adjoin_outer_approximation(PartitionListSet& pls, 
                                                      const EnclosureSetList& esl) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  for(typename EnclosureSetList::const_iterator iter=esl.begin(); iter!=esl.end(); ++iter) {
    base->adjoin_outer_approximation(pls,*iter);
  }
}

template<class Aprx, class ES> inline
void
ApproximatorBase<Aprx,ES>::adjoin_outer_approximation(PartitionTreeSet& pts, 
                                                      const EnclosureSetList& esl) const
{
  const ApproximatorInterface<Aprx,ES>* base=this;
  for(typename EnclosureSetList::const_iterator iter=esl.begin(); iter!=esl.end(); ++iter) {
    base->adjoin_outer_approximation(pts,*iter);
  }
}

} // namespace Ariadne



#endif /* ARIADNE_APPROXIMATOR_BASE_H */
