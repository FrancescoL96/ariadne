/***************************************************************************
 *            c1_taylor_function.cc
 *
 *  Copyright 2008  Pieter Collins
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

#include <iostream>
#include <iomanip>

#include "config.h"

#include "macros.h"
#include "exceptions.h"
#include "numeric.h"
#include "vector.h"
#include "multi_index.h"
#include "expansion.h"

#include "c1_taylor_function.h"

#define VOLATILE ;

namespace Ariadne {

C1TaylorSeries::C1TaylorSeries(uint d)
    : _coefficients(d+1,ExactFloat(0))
    , _zero_error(0)
    , _uniform_error(0)
    , _derivative_error(0)
{
}

C1TaylorSeries C1TaylorSeries::constant(ExactFloat c) {
    C1TaylorSeries result(1u);
    result._coefficients[0]=c;
    return result;
}

C1TaylorSeries C1TaylorSeries::coordinate() {
    C1TaylorSeries result(1u);
    result._coefficients[1]=1;
    return result;
}

Interval C1TaylorSeries::domain() const {
    return Interval(-1,+1);
}

uint C1TaylorSeries::degree() const {
    return this->_coefficients.size()-1;
}

C1TaylorSeries& operator+=(C1TaylorSeries& f, ExactFloat ec) {
    set_rounding_upward();
    Float& fv=f._coefficients[0];
    Float& fze=f._zero_error;
    Float& fe=f._uniform_error;
    const Float& c=ec;
    set_rounding_upward();
    VOLATILE Float fvu=fv+c;
    VOLATILE Float mfvl=(-fv)-c;
    fze+=(fvu+mfvl)/2;
    fe+=(fvu+mfvl)/2;
    set_rounding_to_nearest();
    fv+=c;
    ARIADNE_ASSERT_MSG(f._zero_error>=0,"f="<<f<<" c="<<c);
    return f;
}

C1TaylorSeries& operator*=(C1TaylorSeries&, ExactFloat);

C1TaylorSeries operator+(C1TaylorSeries f1, C1TaylorSeries f2) {
    C1TaylorSeries r(max(f1.degree(),f2.degree()));

    const std::vector<Float>& f1a=f1._coefficients;
    const std::vector<Float>& f2a=f2._coefficients;
    std::vector<Float>& f0a=r._coefficients;

    set_rounding_upward();
    VOLATILE Float vu=f1a[0]+f2a[0];
    VOLATILE Float mvl=(-f1a[0])-f2a[0];
    r._zero_error=(vu+mvl)/2;
    r._uniform_error=(vu+mvl)/2;
    for(uint i=1; i!=min(f1a.size(),f2a.size()); ++i) {
        vu=f1a[i]+f2a[i];
        mvl=(-f1a[i])-f2a[i];
        r._uniform_error+=(vu+mvl)/2;
        r._derivative_error+=i*((vu+mvl)/2);
    }
    r._zero_error+=(f1._zero_error+f2._zero_error);
    r._uniform_error+=f1._uniform_error+f2._uniform_error;
    r._derivative_error+=f1._derivative_error+f2._derivative_error;

    set_rounding_to_nearest();
    for(uint i=0; i!=min(f1a.size(),f2a.size()); ++i) {
        f0a[i]=f1a[i]+f2a[i];
    }
    for(uint i=min(f1a.size(),f2a.size()); i!=f1a.size(); ++i) {
        f0a[i]=f1a[i];
    }
    for(uint i=min(f1a.size(),f2a.size()); i!=f2a.size(); ++i) {
        f0a[i]=f2a[i];
    }
    return r;
}

inline Float abssum(std::vector<Float> const& a) {
    Float s=0;
    for(uint i=0; i!=a.size(); ++i) {
        s+=abs(a[i]);
    }
    return s;
}

inline Float indabssum(std::vector<Float> const& a) {
    ARIADNE_DEBUG_ASSERT(a.size()>=1);
    Float s=0;
    for(uint i=1; i!=a.size(); ++i) {
        s+=i*abs(a[i]);
    }
    return s;
}

C1TaylorSeries operator*(C1TaylorSeries f1, C1TaylorSeries f2) {
    C1TaylorSeries fr(f1.degree()+f2.degree());
    // std::cerr<<"d0="<<fr.degree()<<", d1="<<f1.degree()<<", d2="<<f2.degree()<<"\n";

    const std::vector<Float>& f1a=f1._coefficients;
    const std::vector<Float>& f2a=f2._coefficients;
    std::vector<Float>& fra=fr._coefficients;

    const Float& f1ze0=f1._zero_error;
    const Float& f2ze0=f2._zero_error;
    Float& frze0=fr._zero_error;
    const Float& f1e0=f1._uniform_error;
    const Float& f2e0=f2._uniform_error;
    Float& fre0=fr._uniform_error;
    const Float& f1e1=f1._derivative_error;
    const Float& f2e1=f2._derivative_error;
    Float& fre1=fr._derivative_error;

    set_rounding_upward();

    Float f1sa0=abssum(f1a);
    Float f2sa0=abssum(f2a);
    Float f1sa1=indabssum(f1a);
    Float f2sa1=indabssum(f2a);

    VOLATILE Float vu;
    VOLATILE Float mvl;

    vu=f1a[0]*f2a[0];
    mvl=(-f1a[0])*f2a[0];
    frze0=(vu+mvl)/2;
    fre0=(vu+mvl)/2;
    // std::cerr<<"ir="<<0<<", i1="<<0<<", i2="<<0<<"\n";
    for(uint ir=1; ir!=f1a.size()+f2a.size()-1; ++ir) {
        vu=0.0;
        mvl=0.0;
        for(uint i1=max(0,int(ir)-int(f2a.size()-1)); i1!=min(ir+1,f1a.size()); ++i1) {
            uint i2=ir-i1;
            // std::cerr<<"ir="<<ir<<", i1="<<i1<<", i2="<<i2<<"\n";
            ARIADNE_DEBUG_ASSERT(i2<f2a.size());
            vu+=f1a[i1]*f2a[i2];
            mvl+=(-f1a[i1])*f2a[i2];
        }
        fre0+=((vu+mvl)/2);
        fre1+=ir*((vu+mvl)/2);
    }

    frze0+=f1ze0*f2a[0]+f1a[0]*f2ze0+f1ze0*f2ze0;

    fre0+=f1e0*f2sa0+f1sa0*f2e0+f1e0*f2e0;

    fre1 += ( (f1e1*f2sa0 + f1sa1*f2e0 + f1e1*f2e0)
             + (f1e0*f2sa1 + f1sa0*f2e1 + f1e0*f2e1) );

    set_rounding_to_nearest();
    for(uint i1=0; i1!=f1a.size(); ++i1) {
        for(uint i2=0; i2!=f2a.size(); ++i2) {
            uint i0=i1+i2;
            fra[i0]+=f1a[i1]*f2a[i2];
        }
    }

    return fr;
}


C1TaylorSeries compose(C1TaylorSeries f, C1TaylorSeries g) {
    Nat i=f.degree();
    C1TaylorSeries r=C1TaylorSeries::constant(ExactFloat(f._coefficients[i]));
    while (i!=0) {
        i=i-i;
        r=r*g;
        r+=ExactFloat(f._coefficients[i]);
    }

    set_rounding_upward();
    r._zero_error+=f._zero_error;
    r._uniform_error+=f._uniform_error;
    r._derivative_error+=f._derivative_error;
    set_rounding_to_nearest();

    return r;
}

Interval evaluate(C1TaylorSeries f, ExactFloat x) {
    Nat i=f.degree();
    Interval r=ExactFloat(f._coefficients[i]);
    while (i!=0) {
        i=i-i;
        r*=Interval(x);
        r+=Interval(f._coefficients[i]);
    }
    if(f._zero_error+Float(x)*f._derivative_error < f._uniform_error) {
        r+=Interval(-f._zero_error,+f._zero_error);
        r+=Interval(x)*Interval(-f._derivative_error,+f._derivative_error);
    } else {
        r+=Interval(-f._uniform_error,+f._uniform_error);
    }
    return r;
}


static const char* plusminus = u8"\u00B1";

OutputStream& operator<<(OutputStream& os, const C1TaylorSeries& f) {
    os << "(" << f._coefficients[0] << "±" << f._zero_error << "/" << f._uniform_error << ")+(" << f._coefficients[1] << plusminus << f._derivative_error <<")*x";
    for(Nat i=2; i<f._coefficients.size(); ++i) {
        if (f._coefficients[i]>=0) { os << "+"; }
        os << f._coefficients[i] << "*x^" << i;
    }
    return os;
}

} // namespace Ariadne
