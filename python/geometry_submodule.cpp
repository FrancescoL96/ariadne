/***************************************************************************
 *            geometry_submodule.cpp
 *
 *  Copyright 2008--17  Pieter Collins
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "boost_python.hpp"
#include "utilities.hpp"

#include "config.hpp"

#include "geometry/geometry.hpp"
#include "output/geometry2d.hpp"
#include "geometry/point.hpp"
#include "geometry/curve.hpp"
#include "geometry/interval.hpp"
#include "geometry/box.hpp"
#include "geometry/grid_paving.hpp"
#include "geometry/function_set.hpp"
#include "geometry/affine_set.hpp"

#include "hybrid/discrete_location.hpp"
#include "hybrid/hybrid_set.hpp"



using namespace boost::python;
using namespace Ariadne;

namespace Ariadne {

template<class UB>
struct from_python_dict<Interval<UB>> {
    from_python_dict() { converter::registry::push_back(&convertible,&construct,type_id<Interval<UB>>()); }
    static Void* convertible(PyObject* obj_ptr) {
        if (!PyDict_Check(obj_ptr) || len(boost::python::extract<boost::python::dict>(obj_ptr))!=1) { return 0; } return obj_ptr; }
    static Void construct(PyObject* obj_ptr,converter::rvalue_from_python_stage1_data* data) {
        typedef typename Interval<UB>::LowerBoundType LB;
        boost::python::dict dct = boost::python::extract<boost::python::dict>(obj_ptr);
        boost::python::list lst=dct.items();
        assert(boost::python::len(lst)==1);
        Void* storage = ((converter::rvalue_from_python_storage<Interval<UB>>*)data)->storage.bytes;
        LB lb=boost::python::extract<LB>(lst[0][0]); UB ub=boost::python::extract<UB>(lst[0][1]);
        new (storage) Interval<UB>(lb,ub);
        data->convertible = storage;
    }
};


template<class UB>
struct from_python_list<Interval<UB>> {
    from_python_list() { converter::registry::push_back(&convertible,&construct,type_id<ExactIntervalType>()); }
    static Void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr) || len(boost::python::extract<boost::python::list>(obj_ptr))!=2) { return 0; } return obj_ptr; }
    static Void construct(PyObject* obj_ptr,converter::rvalue_from_python_stage1_data* data) {
        typedef typename Interval<UB>::LowerBoundType LB;
        boost::python::list lst = boost::python::extract<boost::python::list>(obj_ptr);
        assert(boost::python::len(lst)==2);
        Void* storage = ((converter::rvalue_from_python_storage<ExactIntervalType>*)data)->storage.bytes;
        LB lb=boost::python::extract<LB>(lst[0]); UB ub=boost::python::extract<UB>(lst[1]);
        new (storage) Interval<UB>(lb,ub);
        data->convertible = storage;
    }
};

template<class X>
struct from_python<Point<X>> {
    from_python() { converter::registry::push_back(&convertible,&construct,type_id<Point<X>>()); }
    static Void* convertible(PyObject* obj_ptr) { if (!PyList_Check(obj_ptr) && !PyTuple_Check(obj_ptr)) { return 0; } return obj_ptr; }
    static Void construct(PyObject* obj_ptr,converter::rvalue_from_python_stage1_data* data) {
        boost::python::extract<boost::python::tuple> xtup(obj_ptr);
        boost::python::extract<boost::python::list> xlst(obj_ptr);
        Point<X> pt;
        if(xtup.check()) {
            boost::python::tuple tup=xtup(); pt=Point<X>(len(tup));
            for(Nat i=0; i!=static_cast<Nat>(len(tup)); ++i) { pt[i]=boost::python::extract<X>(tup[i]); }
        } else if(xlst.check()) {
            boost::python::list lst=xlst(); pt=Point<X>(len(lst));
            for(Nat i=0; i!=static_cast<Nat>(len(lst)); ++i) { pt[i]=boost::python::extract<X>(lst[i]); }
        }
        Void* storage = ((converter::rvalue_from_python_storage<X>*)data)->storage.bytes;
        new (storage) Point<X>(pt);
        data->convertible = storage;
    }
};

template<class IVL>
struct from_python<Box<IVL>> {
    from_python() { converter::registry::push_back(&convertible,&construct,type_id<Box<IVL>>()); }
    static Void* convertible(PyObject* obj_ptr) { if (!PyList_Check(obj_ptr)) { return 0; } return obj_ptr; }
    static Void construct(PyObject* obj_ptr,converter::rvalue_from_python_stage1_data* data) {
        Void* storage = ((converter::rvalue_from_python_storage<IVL>*)data)->storage.bytes;
        boost::python::list lst=boost::python::extract<boost::python::list>(obj_ptr);
        Box<IVL>* bx_ptr = new (storage) Box<IVL>(static_cast<SizeType>(len(lst)));
        for(Int i=0; i!=len(lst); ++i) { (*bx_ptr)[static_cast<SizeType>(i)]=boost::python::extract<IVL>(lst[i]); }
        data->convertible = storage;
    }
};


template<class ES>
struct to_python< ListSet<ES> > {
    to_python() { boost::python::to_python_converter< ListSet<ES>, to_python< ListSet<ES> > >(); }

    static PyObject* convert(const ListSet<ES>& ls) {
        boost::python::list result;
        for(typename ListSet<ES>::ConstIterator iter=ls.begin(); iter!=ls.end(); ++iter) {
            result.append(boost::python::object(*iter));
        }
        return boost::python::incref(boost::python::list(result).ptr());
    }
    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

template<class ES>
struct to_python< ListSet< HybridBasicSet<ES> > > {
    typedef ListSet< HybridBasicSet<ES> > SetType;
    to_python() { boost::python::to_python_converter< SetType, to_python<SetType> >(); }

    static PyObject* convert(const SetType& hls) {
        boost::python::dict result;
        for(typename SetType::LocationsConstIterator iter=hls.locations_begin(); iter!=hls.locations_end(); ++iter) {
            result[iter->first]=iter->second;
        }
        return boost::python::incref(boost::python::dict(result).ptr());
    }
    static const PyTypeObject* get_pytype() { return &PyDict_Type; }
};

OutputStream& operator<<(OutputStream& os, const PythonRepresentation<FloatDPBounds>& x);
OutputStream& operator<<(OutputStream& os, const PythonRepresentation<ExactIntervalType>& x) {
    ExactIntervalType const& ivl=x.reference(); return os << PythonRepresentation<FloatDPBounds>(FloatDPBounds(ivl.lower(),ivl.upper()));
}



class OpenSetWrapper
  : public virtual OpenSetInterface, public wrapper< OpenSetInterface >
{
  public:
    OpenSetInterface* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean covers(const ExactBoxType& r) const { return this->get_override("covers")(); }
    LowerKleenean overlaps(const ExactBoxType& r) const { return this->get_override("overlaps")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};

class ClosedSetWrapper
  : public virtual ClosedSetInterface, public wrapper< ClosedSetInterface >
{
  public:
    ClosedSetInterface* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean separated(const ExactBoxType& r) const { return this->get_override("separated")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};


class OvertSetWrapper
  : public virtual OvertSetInterface, public wrapper< OvertSetInterface >
{
  public:
    OvertSetInterface* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean overlaps(const ExactBoxType& r) const { return this->get_override("overlaps")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};


class CompactSetWrapper
  : public virtual CompactSetInterface, public wrapper< CompactSetInterface >
{
  public:
    CompactSetInterface* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean separated(const ExactBoxType& r) const { return this->get_override("separated")(); }
    LowerKleenean inside(const ExactBoxType& r) const { return this->get_override("inside")(); }
    LowerKleenean is_bounded() const { return this->get_override("is_bounded")(); }
    UpperBoxType bounding_box() const { return this->get_override("bounding_box")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};

class RegularSetWrapper
  : public virtual LocatedSetInterface, public wrapper< RegularSetWrapper >
{
  public:
    RegularSetWrapper* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean overlaps(const ExactBoxType& r) const { return this->get_override("overlaps")(); }
    LowerKleenean covers(const ExactBoxType& r) const { return this->get_override("covers")(); }
    LowerKleenean separated(const ExactBoxType& r) const { return this->get_override("separated")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};

class LocatedSetWrapper
  : public virtual LocatedSetInterface, public wrapper< LocatedSetInterface >
{
  public:
    LocatedSetInterface* clone() const { return this->get_override("clone")(); }
    SizeType dimension() const { return this->get_override("dimension")(); }
    LowerKleenean overlaps(const ExactBoxType& r) const { return this->get_override("overlaps")(); }
    LowerKleenean separated(const ExactBoxType& r) const { return this->get_override("separated")(); }
    LowerKleenean inside(const ExactBoxType& r) const { return this->get_override("inside")(); }
    LowerKleenean is_bounded() const { return this->get_override("is_bounded")(); }
    UpperBoxType bounding_box() const { return this->get_override("bounding_box")(); }
    OutputStream& write(OutputStream&) const { return this->get_override("write")(); }
};


// Declare Geometry friend operations
ConstraintSet intersection(const ConstraintSet& cs1, const ConstraintSet& cs2);
BoundedConstraintSet intersection(const ConstraintSet& cs, const RealBox& bx);
BoundedConstraintSet intersection(const RealBox& bx, const ConstraintSet& cs);

BoundedConstraintSet intersection(const BoundedConstraintSet& bcs1, const BoundedConstraintSet& bcs2);
BoundedConstraintSet intersection(const BoundedConstraintSet& bcs1, const ConstraintSet& cs2);
BoundedConstraintSet intersection(const ConstraintSet& cs1, const BoundedConstraintSet& bcs2);
BoundedConstraintSet intersection(const BoundedConstraintSet& bcs1, const RealBox& bx2);
BoundedConstraintSet intersection(const RealBox& bx1, const BoundedConstraintSet& bcs2);
ConstrainedImageSet image(const BoundedConstraintSet& set, const EffectiveVectorFunction& function);

ValidatedConstrainedImageSet image(ValidatedConstrainedImageSet set, ValidatedVectorFunction const& h);
ValidatedConstrainedImageSet join(const ValidatedConstrainedImageSet& set1, const ValidatedConstrainedImageSet& set2);

ValidatedAffineConstrainedImageSet image(ValidatedAffineConstrainedImageSet set, ValidatedVectorFunction const& h);
}


Void export_set_interface() {
    class_<OpenSetInterface, boost::noncopyable> open_set_wrapper_class("OpenSetInterface", no_init);
    open_set_wrapper_class.def("covers",(LowerKleenean(OpenSetInterface::*)(const ExactBoxType& bx)const) &OpenSetInterface::covers);
    open_set_wrapper_class.def("overlaps",(LowerKleenean(OvertSetInterface::*)(const ExactBoxType& bx)const) &OpenSetInterface::overlaps);

    class_<ClosedSetInterface, boost::noncopyable> closed_set_wrapper_class("ClosedSetInterface", no_init);
    closed_set_wrapper_class.def("separated",(LowerKleenean(ClosedSetInterface::*)(const ExactBoxType& bx)const) &ClosedSetInterface::separated);

    class_<OvertSetInterface, boost::noncopyable> overt_set_wrapper_class("OvertSetInterface", no_init);
    overt_set_wrapper_class.def("overlaps",(LowerKleenean(OvertSetInterface::*)(const ExactBoxType& bx)const) &OvertSetInterface::overlaps);

    class_<CompactSetInterface, boost::noncopyable> compact_set_wrapper_class("CompactSetInterface", no_init);
    compact_set_wrapper_class.def("separated",(LowerKleenean(ClosedSetInterface::*)(const ExactBoxType& bx)const) &CompactSetInterface::separated);
    compact_set_wrapper_class.def("inside",(LowerKleenean(BoundedSetInterface::*)(const ExactBoxType& bx)const) &CompactSetInterface::inside);
    compact_set_wrapper_class.def("bounding_box",&CompactSetInterface::bounding_box);

    class_<LocatedSetInterface, boost::noncopyable> located_set_wrapper_class("LocatedSetInterface", no_init);
    class_<RegularSetInterface, boost::noncopyable> regular_set_wrapper_class("RegularSetInterface", no_init);

    class_<DrawableInterface,boost::noncopyable>("DrawableInterface",no_init);

}


Void export_point()
{
    class_<ExactPoint,bases<DrawableInterface>> point_class("ExactPoint",init<ExactPoint>());
    point_class.def(init<Nat>());
    point_class.def("__getitem__", &__getitem__<ExactPoint,Int,FloatDPValue>);
    point_class.def(self_ns::str(self));

    from_python<ExactPoint>();
    implicitly_convertible<Vector<FloatDPValue>,ExactPoint>();

}

typedef LogicalType<ExactTag> ExactLogicalType;


template<class IVL> Void export_interval(std::string name) {
    typedef IVL IntervalType;
    typedef typename IntervalType::LowerBoundType LowerBoundType;
    typedef typename IntervalType::UpperBoundType UpperBoundType;
    typedef typename IntervalType::MidpointType MidpointType;

    typedef decltype(contains(declval<IntervalType>(),declval<MidpointType>())) ContainsType;
    typedef decltype(disjoint(declval<IntervalType>(),declval<IntervalType>())) DisjointType;
    typedef decltype(subset(declval<IntervalType>(),declval<IntervalType>())) SubsetType;

    from_python_dict<IVL>();

    class_< IntervalType > interval_class(name.c_str(),init<IntervalType>());
    //interval_class.def(init<MidpointType>());
    interval_class.def(init<LowerBoundType,UpperBoundType>());

    // FIXME: Only export this if constructor exists
//    if constexpr (IsConstructibleGivenDefaultPrecision<UB,Dyadic>::value and not IsConstructible<UB,Dyadic>::value) {
        interval_class.def(init<Interval<Dyadic>>());
//    }

    interval_class.def(self == self);
    interval_class.def(self != self);
    interval_class.def("lower", &IntervalType::lower, return_value_policy<copy_const_reference>());
    interval_class.def("upper", &IntervalType::upper, return_value_policy<copy_const_reference>());
    interval_class.def("midpoint", &IntervalType::midpoint);
    interval_class.def("radius", &IntervalType::radius);
    interval_class.def("width", &IntervalType::width);
    interval_class.def("contains", (ContainsType(*)(IntervalType const&,MidpointType const&)) &contains);
    interval_class.def("empty", &IntervalType::is_empty);
    interval_class.def(boost::python::self_ns::str(self));
    interval_class.def(boost::python::self_ns::repr(self));

    //from_python_list<IntervalType>();
    //from_python_str<ExactIntervalType>();

    def("midpoint", &IntervalType::midpoint);
    def("radius", &IntervalType::radius);
    def("width", &IntervalType::width);

    def("contains", (ContainsType(*)(IntervalType const&,MidpointType const&)) &contains);
    def("disjoint", (DisjointType(*)(IntervalType const&,IntervalType const&)) &disjoint);
    def("subset", (SubsetType(*)(IntervalType const&,IntervalType const&)) &subset);

    def("intersection", (IntervalType(*)(IntervalType const&,IntervalType const&)) &intersection);
    def("hull", (IntervalType(*)(IntervalType const&, IntervalType const&)) &hull);
}

Void export_intervals() {
    export_interval<ExactIntervalType>("ExactInterval");
    export_interval<UpperIntervalType>("UpperInterval");
    export_interval<ApproximateIntervalType>("ApproximateInterval");
    export_interval<DyadicInterval>("DyadicInterval");
    export_interval<RealInterval>("RealInterval");
}

template<class BX> Void export_box(std::string name)
{
    using IVL = typename BX::IntervalType;
    using UB = typename IVL::UpperBoundType;
    //class_<Vector<ExactIntervalType>> interval_vector_class("ExactIntervalVectorType");

    typedef decltype(disjoint(declval<BX>(),declval<BX>())) DisjointType;
    typedef decltype(subset(declval<BX>(),declval<BX>())) SubsetType;
    typedef decltype(separated(declval<BX>(),declval<BX>())) SeparatedType;
    typedef decltype(overlap(declval<BX>(),declval<BX>())) OverlapType;
    typedef decltype(covers(declval<BX>(),declval<BX>())) CoversType;
    typedef decltype(inside(declval<BX>(),declval<BX>())) InsideType;

//    class_<ExactBoxType,bases<CompactSetInterface,OpenSetInterface,Vector<ExactIntervalType>,DrawableInterface > >
    class_<BX,bases< > > box_class(name.c_str(),init<BX>());
    if constexpr (IsConstructibleGivenDefaultPrecision<UB,Dyadic>::value and not IsConstructible<UB,Dyadic>::value) {
        box_class.def(init<Box<Interval<Dyadic>>>());
    }

    static_assert(IsConstructibleGivenDefaultPrecision<FloatDPValue,Dyadic>::value and not IsConstructible<FloatDPValue,Dyadic>::value);

    box_class.def(init<DimensionType>());
    box_class.def(init< Vector<IVL> >());
    //box_class.def("__eq__", (ExactLogicalType(*)(const Vector<ExactIntervalType>&,const Vector<ExactIntervalType>&)) &operator==);
    box_class.def("dimension", (DimensionType(BX::*)()const) &BX::dimension);
    box_class.def("centre", (typename BX::CentreType(BX::*)()const) &BX::centre);
    box_class.def("radius", (typename BX::RadiusType(BX::*)()const) &BX::radius);
    box_class.def("separated", (SeparatedType(BX::*)(const BX&)const) &BX::separated);
    box_class.def("overlaps", (OverlapType(BX::*)(const BX&)const) &BX::overlaps);
    box_class.def("covers", (CoversType(BX::*)(const BX&)const) &BX::covers);
    box_class.def("inside", (InsideType(BX::*)(const BX&)const) &BX::inside);
    box_class.def("is_empty", (SeparatedType(BX::*)()const) &BX::is_empty);
    box_class.def("split", (Pair<BX,BX>(BX::*)()const) &BX::split);
    box_class.def("split", (Pair<BX,BX>(BX::*)(SizeType)const) &BX::split);
    box_class.def("split", (Pair<BX,BX>(BX::*)()const) &BX::split);
    box_class.def(self_ns::str(self));

    def("disjoint", (DisjointType(*)(const BX&,const BX&)) &disjoint);
    def("subset", (SubsetType(*)(const BX&,const BX&)) &subset);

    def("product", (BX(*)(const BX&,const IVL&)) &product);
    def("product", (BX(*)(const BX&,const BX&)) &product);
    def("hull", (BX(*)(const BX&,const BX&)) &hull);
    def("intersection", (BX(*)(const BX&,const BX&)) &intersection);

    from_python<BX>();
    to_python< Pair<BX,BX> >();
}

template<> Void export_box<DyadicBox>(std::string name)
{
    using BX=DyadicBox;
    class_<BX,bases< > > box_class(name.c_str(),init<BX>());
    from_python<BX>();
}

Void export_boxes() {
    export_box<RealBox>("RealBox");
    export_box<DyadicBox>("DyadicBox");
    export_box<ExactBoxType>("ExactBox");
    export_box<UpperBoxType>("UpperBox");
    export_box<ApproximateBoxType>("ApproximateBox");

    implicitly_convertible<ExactBoxType,UpperBoxType>();
    implicitly_convertible<ExactBoxType,ApproximateBoxType>();
    implicitly_convertible<UpperBoxType,ApproximateBoxType>();

    def("widen", (UpperBoxType(*)(ExactBoxType const&, FloatDPValue eps)) &widen);
    def("image", (UpperBoxType(*)(UpperBoxType, ValidatedVectorFunction const&)) &image);
}

/*
Pair<Zonotope,Zonotope> split_pair(const Zonotope& z) {
    ListSet<Zonotope> split_list=split(z);
    ARIADNE_ASSERT(split_list.size()==2);
    return Pair<Zonotope,Zonotope>(split_list[0],split_list[1]);
}

Void export_zonotope()
{
    class_<Zonotope,bases<CompactSetInterface,OpenSetInterface,DrawableInterface> > zonotope_class("Zonotope",init<Zonotope>());
    zonotope_class.def(init< Vector<FloatDPValue>, Matrix<FloatDPValue>, Vector<FloatDPError> >());
    zonotope_class.def(init< Vector<FloatDPValue>, Matrix<FloatDPValue> >());
    zonotope_class.def(init< ExactBoxType >());
    zonotope_class.def("centre",&Zonotope::centre,return_value_policy<copy_const_reference>());
    zonotope_class.def("generators",&Zonotope::generators,return_value_policy<copy_const_reference>());
    zonotope_class.def("error",&Zonotope::error,return_value_policy<copy_const_reference>());
    zonotope_class.def("contains",&Zonotope::contains);
    zonotope_class.def("split", (ListSet<Zonotope>(*)(const Zonotope&)) &split);
    zonotope_class.def("__str__",&__cstr__<Zonotope>);

    def("contains", (ValidatedKleenean(*)(const Zonotope&,const ExactPoint&)) &contains);
    def("separated", (ValidatedKleenean(*)(const Zonotope&,const ExactBoxType&)) &separated);
    def("overlaps", (ValidatedKleenean(*)(const Zonotope&,const ExactBoxType&)) &overlaps);
    def("separated", (ValidatedKleenean(*)(const Zonotope&,const Zonotope&)) &separated);

    def("polytope", (Polytope(*)(const Zonotope&)) &polytope);
    def("orthogonal_approximation", (Zonotope(*)(const Zonotope&)) &orthogonal_approximation);
    def("orthogonal_over_approximation", (Zonotope(*)(const Zonotope&)) &orthogonal_over_approximation);
    def("error_free_over_approximation", (Zonotope(*)(const Zonotope&)) &error_free_over_approximation);

//    def("apply", (Zonotope(*)(const ValidatedVectorFunction&, const Zonotope&)) &apply);

    to_python< ListSet<Zonotope> >();
}


Void export_polytope()
{
    class_<Polytope,bases<LocatedSetInterface,DrawableInterface> > polytope_class("Polytope",init<Polytope>());
    polytope_class.def(init<Int>());
    polytope_class.def("new_vertex",&Polytope::new_vertex);
    polytope_class.def("__iter__",boost::python::range(&Polytope::vertices_begin,&Polytope::vertices_end));
    polytope_class.def(self_ns::str(self));

}
*/

Void export_curve()
{
    to_python< Pair<const FloatDPValue,ExactPoint> >();

    class_<InterpolatedCurve,bases<DrawableInterface> > interpolated_curve_class("InterpolatedCurve",init<InterpolatedCurve>());
    interpolated_curve_class.def(init<FloatDPValue,ExactPoint>());
    interpolated_curve_class.def("insert", (Void(InterpolatedCurve::*)(const FloatDPValue&, const Point<FloatDPApproximation>&)) &InterpolatedCurve::insert);
    interpolated_curve_class.def("__iter__",boost::python::range(&InterpolatedCurve::begin,&InterpolatedCurve::end));
    interpolated_curve_class.def(self_ns::str(self));


}



Void export_affine_set()
{
    class_<ValidatedAffineConstrainedImageSet,bases<CompactSetInterface,DrawableInterface> >
        affine_set_class("ValidatedAffineConstrainedImageSet",init<ValidatedAffineConstrainedImageSet>());
    affine_set_class.def(init<RealBox>());
    affine_set_class.def(init<ExactBoxType>());
    affine_set_class.def(init<Vector<ExactIntervalType>, Matrix<FloatDPValue>, Vector<FloatDPValue> >());
    affine_set_class.def(init<Matrix<FloatDPValue>, Vector<FloatDPValue> >());
    affine_set_class.def("new_parameter_constraint", (Void(ValidatedAffineConstrainedImageSet::*)(const Constraint<Affine<FloatDPBounds>,FloatDPBounds>&)) &ValidatedAffineConstrainedImageSet::new_parameter_constraint);
    affine_set_class.def("new_constraint", (Void(ValidatedAffineConstrainedImageSet::*)(const Constraint<AffineModel<ApproximateTag,FloatDP>,FloatDPBounds>&)) &ValidatedAffineConstrainedImageSet::new_constraint);
    affine_set_class.def("dimension", &ValidatedAffineConstrainedImageSet::dimension);
    affine_set_class.def("is_bounded", &ValidatedAffineConstrainedImageSet::is_bounded);
    affine_set_class.def("is_empty", &ValidatedAffineConstrainedImageSet::is_empty);
    affine_set_class.def("bounding_box", &ValidatedAffineConstrainedImageSet::bounding_box);
    affine_set_class.def("separated", &ValidatedAffineConstrainedImageSet::separated);
    affine_set_class.def("adjoin_outer_approximation_to", &ValidatedAffineConstrainedImageSet::adjoin_outer_approximation_to);
    affine_set_class.def("outer_approximation", &ValidatedAffineConstrainedImageSet::outer_approximation);
    affine_set_class.def("boundary", &ValidatedAffineConstrainedImageSet::boundary);
    affine_set_class.def(self_ns::str(self));

    def("image", (ValidatedAffineConstrainedImageSet(*)(ValidatedAffineConstrainedImageSet,ValidatedVectorFunction const&)) &image);
}

Void export_constraint_set()
{
    from_python< List<EffectiveConstraint> >();

    class_<ConstraintSet,bases<RegularSetInterface,OpenSetInterface> >
        constraint_set_class("ConstraintSet",init<ConstraintSet>());
    constraint_set_class.def(init< List<EffectiveConstraint> >());
    constraint_set_class.def("dimension", &ConstraintSet::dimension);
    constraint_set_class.def(self_ns::str(self));

    class_<BoundedConstraintSet,bases<DrawableInterface> >
        bounded_constraint_set_class("BoundedConstraintSet",init<BoundedConstraintSet>());
    bounded_constraint_set_class.def(init< RealBox, List<EffectiveConstraint> >());
    bounded_constraint_set_class.def("dimension", &BoundedConstraintSet::dimension);
    bounded_constraint_set_class.def(self_ns::str(self));

    def("intersection", &_intersection_<ConstraintSet,ConstraintSet>);
    def("intersection", &_intersection_<ConstraintSet,RealBox>);
    def("intersection", &_intersection_<RealBox,ConstraintSet>);

    def("intersection", &_intersection_<BoundedConstraintSet,BoundedConstraintSet>);
    def("intersection", &_intersection_<BoundedConstraintSet,RealBox>);
    def("intersection", &_intersection_<RealBox,BoundedConstraintSet>);
    def("intersection", &_intersection_<ConstraintSet,BoundedConstraintSet>);
    def("intersection", &_intersection_<BoundedConstraintSet,ConstraintSet>);

    def("image", &_image_<BoundedConstraintSet,EffectiveVectorFunction>);

}


Void export_constrained_image_set()
{
    from_python< List<ValidatedConstraint> >();

    class_<ConstrainedImageSet,bases<CompactSetInterface,DrawableInterface> >
        constrained_image_set_class("ConstrainedImageSet",init<ConstrainedImageSet>());
    constrained_image_set_class.def("dimension", &ConstrainedImageSet::dimension);
    constrained_image_set_class.def(self_ns::str(self));

    class_<ValidatedConstrainedImageSet,bases<CompactSetInterface,DrawableInterface> >
        validated_constrained_image_set_class("ValidatedConstrainedImageSet",init<ValidatedConstrainedImageSet>());
    validated_constrained_image_set_class.def(init<ExactBoxType>());
    validated_constrained_image_set_class.def(init<ExactBoxType,EffectiveVectorFunction>());
    validated_constrained_image_set_class.def(init<ExactBoxType,ValidatedVectorFunction>());
    validated_constrained_image_set_class.def(init<ExactBoxType,ValidatedVectorFunction,List<ValidatedConstraint> >());
    validated_constrained_image_set_class.def(init<ExactBoxType,ValidatedVectorFunctionModelDP>());
    validated_constrained_image_set_class.def("domain", &ValidatedConstrainedImageSet::domain,return_value_policy<copy_const_reference>());
    validated_constrained_image_set_class.def("function", &ValidatedConstrainedImageSet::function,return_value_policy<copy_const_reference>());
    validated_constrained_image_set_class.def("constraint", &ValidatedConstrainedImageSet::constraint);
    validated_constrained_image_set_class.def("number_of_parameters", &ValidatedConstrainedImageSet::number_of_parameters);
    validated_constrained_image_set_class.def("number_of_constraints", &ValidatedConstrainedImageSet::number_of_constraints);
    validated_constrained_image_set_class.def("apply", &ValidatedConstrainedImageSet::apply);
    validated_constrained_image_set_class.def("new_space_constraint", (Void(ValidatedConstrainedImageSet::*)(const EffectiveConstraint&))&ValidatedConstrainedImageSet::new_space_constraint);
    validated_constrained_image_set_class.def("new_parameter_constraint", (Void(ValidatedConstrainedImageSet::*)(const EffectiveConstraint&))&ValidatedConstrainedImageSet::new_parameter_constraint);
    //constrained_image_set_class.def("outer_approximation", &ValidatedConstrainedImageSet::outer_approximation);
    validated_constrained_image_set_class.def("affine_approximation", &ValidatedConstrainedImageSet::affine_approximation);
    	constrained_image_set_class.def("affine_over_approximation", &ValidatedConstrainedImageSet::affine_over_approximation);
    validated_constrained_image_set_class.def("adjoin_outer_approximation_to", &ValidatedConstrainedImageSet::adjoin_outer_approximation_to);
    validated_constrained_image_set_class.def("bounding_box", &ValidatedConstrainedImageSet::bounding_box);
    validated_constrained_image_set_class.def("inside", &ValidatedConstrainedImageSet::inside);
    validated_constrained_image_set_class.def("separated", &ValidatedConstrainedImageSet::separated);
    validated_constrained_image_set_class.def("overlaps", &ValidatedConstrainedImageSet::overlaps);
    validated_constrained_image_set_class.def("split", (Pair<ValidatedConstrainedImageSet,ValidatedConstrainedImageSet>(ValidatedConstrainedImageSet::*)()const) &ValidatedConstrainedImageSet::split);
    validated_constrained_image_set_class.def("split", (Pair<ValidatedConstrainedImageSet,ValidatedConstrainedImageSet>(ValidatedConstrainedImageSet::*)(Nat)const) &ValidatedConstrainedImageSet::split);
    validated_constrained_image_set_class.def(self_ns::str(self));
    validated_constrained_image_set_class.def("__repr__", &__cstr__<ValidatedConstrainedImageSet>);

//    def("product", (ValidatedConstrainedImageSet(*)(const ValidatedConstrainedImageSet&,const ExactBoxType&)) &product);
}




Void geometry_submodule() {
    export_set_interface();
    export_point();

    export_intervals();
    export_boxes();
//    export_zonotope();
//    export_polytope();
    export_curve();

    export_affine_set();

    export_constraint_set();
    export_constrained_image_set();

}
