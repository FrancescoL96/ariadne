/***************************************************************************
 *            taylor_variable.cc
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
 *  GNU Library General Public License for more detai1ls.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */
 
#ifndef ARIADNE_COSY_TAYLOR_ARITHMETIC
#define ARIADNE_ROUNDED_TAYLOR_ARITHMETIC
#endif

#include <iomanip>

#include "config.h"
#include "rounding.h"
#include "numeric.h"
#include "sparse_differential.h"
#include "differential_vector.h"
#include "taylor_variable.h"

namespace Ariadne {

inline void acc(Interval& e, const Interval& d) {
    e+=d;
}

inline void acc(Interval& e, Float& x, const Float& y) {
    Float z=x;
    x+=y;
    e+=(Interval(y)+Interval(z)-Interval(x));
}

inline void acc(Interval& e, Float& x, const Interval& y) {
    Float z=x;
    x+=midpoint(y);
    e+=(y+Interval(z)-Interval(x));
}

inline void acc(Interval& e, Float& x, const Float& y, const Float& z) {
    Float xold=x;
    x+=y*z;
    e+=Interval(xold)+Interval(y)*Interval(z)-Interval(x);
}


const double TaylorVariable::em=2.2204460492503131e-16;
const double TaylorVariable::ec=em/2;
   

TaylorVariable add_cosy(const TaylorVariable&, const Float&);
TaylorVariable mul_cosy(const TaylorVariable&, const Float&);
TaylorVariable add_cosy(const TaylorVariable&, const TaylorVariable&);
TaylorVariable mul_cosy(const TaylorVariable&, const TaylorVariable&);

TaylorVariable add_rounded(const TaylorVariable&, const Float&);
TaylorVariable mul_rounded(const TaylorVariable&, const Float&);
TaylorVariable add_rounded(const TaylorVariable&, const TaylorVariable&);
TaylorVariable mul_rounded(const TaylorVariable&, const TaylorVariable&);

#if defined ARIADNE_COSY_TAYLOR_ARITHMETIC
inline TaylorVariable add(const TaylorVariable& x, const Float& c) { return add_cosy(x,c); }
inline TaylorVariable add(const TaylorVariable& x, const TaylorVariable& y) { return add_cosy(x,y); }
inline TaylorVariable mul(const TaylorVariable& x, const Float& c) { return mul_cosy(x,c); }
inline TaylorVariable mul(const TaylorVariable& x, const TaylorVariable& y) { return mul_cosy(x,y); }
#elif defined ARIADNE_ROUNDED_TAYLOR_ARITHMETIC
inline TaylorVariable add(const TaylorVariable& x, const Float& c) { return add_rounded(x,c); }
inline TaylorVariable add(const TaylorVariable& x, const TaylorVariable& y) { return add_rounded(x,y); }
inline TaylorVariable mul(const TaylorVariable& x, const Float& c) { return mul_rounded(x,c); }
inline TaylorVariable mul(const TaylorVariable& x, const TaylorVariable& y) { return mul_rounded(x,y); }
#endif



TaylorVariable&
TaylorVariable::scal(const Float& c) 
{
    //std::cerr<<"TaylorVariable::scal_rounded(Float)"<<std::endl;

    // Shortcuts for special cases
    if(c==0.0) {
        this->_expansion.data().clear();
        this->_error=0;
        return *this;
    }
    if(c==1.0) { 
        return *this; 
    }
    if(c==0.5 || c==2.0 || c==-2.0 || c==-1.0 || c==-0.5) {
        // Operation can be performed exactly
        for(iterator iter=this->begin(); iter!=this->end(); ++iter) {
            iter->second*=c;
        }
        this->_error*=c;
        return *this;
    }
    
    // General case with error analysis
    TaylorVariable& x=*this;
    Interval& xe=x.error();
    set_rounding_upward();
    volatile Float te=0; // Twice the maximum accumulated error
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        volatile Float u=xiter->second*c;
        volatile Float t=-xiter->second;
        volatile Float ml=t*c;
        te+=(u+ml);
    }
    xe.u+=te/2; xe.l=-xe.u;

    set_rounding_to_nearest();
    for(iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        xiter->second*=c;
    }
    
    return x;
}


TaylorVariable&
TaylorVariable::scal(const Interval& c) 
{
    // General case with error analysis
    TaylorVariable& x=*this;
    Interval& xe=x.error();
    set_rounding_upward();
    volatile Float u,ml;
    Float te=0; // Twice the maximum accumulated error
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        const Float& xv=xiter->second;
        volatile Float mxv=-xv;
        if(xv>=0) {
            u=xv*c.u;
            ml=mxv*c.l;
        } else {
            u=xv*c.l;
            ml=mxv*c.u;
        }
        te+=(u+ml);
    }
    xe.u+=te/2; xe.l=-xe.u;

    set_rounding_to_nearest();
    Float m=(c.u+c.l)/2;
    for(iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        xiter->second*=m;
    }

    return x;
}


TaylorVariable&
TaylorVariable::acc(const Float& c) 
{
    // Compute self+=c

    TaylorVariable& r=*this;
    Float& rv=r.value();
    Interval& re=r.error();
    set_rounding_upward();
    volatile Float rvu=rv+c;
    volatile Float mrvl=(-rv)-c;
    //std::cerr<<"re="<<re.u<<" ";
    re.u+=(rvu+mrvl)/2;
    //std::cerr<<"nre="<<re.u<<"\n";
    re.l=-re.u;
    set_rounding_to_nearest();
    rv+=c;
    return r;
}


TaylorVariable&
TaylorVariable::acc(const Interval& c) 
{
    // Compute self+=c

    TaylorVariable& r=*this;
    Float& rv=r.value();
    Interval& re=r.error();
    set_rounding_upward();
    volatile Float rvu=rv+c.u;
    volatile Float mrvl=(-rv)-c.l;
    re.u+=(rvu+mrvl)/2;
    re.l=-re.u;
    set_rounding_to_nearest();
    volatile Float m=(c.u+c.l)/2;
    rv+=m;
    return r;
}


TaylorVariable&
TaylorVariable::acc(const TaylorVariable& x)
{
    // Compute self+=x
    TaylorVariable& r=*this;
    Interval& re=r.error();
    //assert(re.l==-re.u);
    //assert(x.error().l==-x.error().u);
    //std::cerr<<std::setprecision(20);
    
    set_rounding_upward();
    Float te=0;
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        const Float& xv=xiter->second;;
        Float& rv=r[xiter->first];
        if(rv!=0) {
            volatile Float u=rv+xv;
            volatile Float t=-rv;
            volatile Float ml=t-xv;
            te+=(u+ml);
            //std::cerr<<" xv="<<xv<<" rv="<<rv<<" u="<<u<<" ml="<<ml<<" d="<<(u+ml)<<" te="<<te<<"\n";
        } 
    }
    re.u+=te/2; 
    re.u+=x.error().u;
    re.l=-re.u;

    set_rounding_to_nearest();
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        r[xiter->first]+=xiter->second;
    }

    return r;
}


struct Ivl { double u; double ml; };

TaylorVariable&
TaylorVariable::acc(const TaylorVariable& x, const TaylorVariable& y)
{
    // Compute self+=x*y
    typedef std::map<MultiIndex,Ivl>::const_iterator ivl_const_iterator;  
    TaylorVariable& r=*this;
    Interval& re=r.error();
    assert(re.l==-re.u);
    std::map<MultiIndex,Ivl> z;

    set_rounding_upward();
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        for(const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
            const Float& xv=xiter->second;;
            const Float& yv=yiter->second;;
            Ivl& zv=z[xiter->first+yiter->first];
            zv.u+=xv*yv;
            volatile double t=-xv;
            zv.ml+=t*yv;
        }
    }
    
    for(const_iterator riter=r.begin(); riter!=r.end(); ++riter) {
        Ivl& zv=z[riter->first];
        const Float& rv=riter->second;
        zv.u+=rv; zv.ml-=rv;
    }
    
    volatile Float te=0;
    for(ivl_const_iterator ziter=z.begin(); ziter!=z.end(); ++ziter) {
        const Ivl& zv=ziter->second;
        te+=(zv.u+zv.ml);
    }
    te/=2;

    Float xs=0;
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        xs+=abs(xiter->second);
    }

    Float ys=0;
    for(const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
        ys+=abs(yiter->second);
    }

    const Float& xe=x.error().u;
    const Float& ye=y.error().u;
    
    re.u+=xs*ye+ys*xe+te; re.l=-re.u; 

    set_rounding_to_nearest();
    for(ivl_const_iterator ziter=z.begin(); ziter!=z.end(); ++ziter) {
        const Ivl& zv=ziter->second;
        r[ziter->first]=(zv.u-zv.ml)/2;
    }

    return r;
}

TaylorVariable&
TaylorVariable::sweep(const Float& m)
{
    for(iterator iter=this->_expansion.begin(); iter!=this->end(); ) {
        Float a=abs(iter->second);
        if(abs(iter->second)<=m) {
            this->_error+=Interval(-a,a);
            this->_expansion.data().erase(iter++);
        } else {
            ++iter;
        }
    }
    return *this;
}

TaylorVariable&
TaylorVariable::truncate(uint d)
{
    set_rounding_upward();
    Float e=0;
    for(iterator iter=this->_expansion.begin(); iter!=this->end(); ) {
        if(iter->first.degree()>d) {
            e+=abs(iter->second);
            this->_expansion.data().erase(iter++);
        } else {
            ++iter;
        }
    }
    this->_error.u+=e;
    this->_error.l=-this->_error.u;
    set_rounding_to_nearest();
    return *this;
}

TaylorVariable&
TaylorVariable::truncate(const MultiIndex& a)
{
    ARIADNE_ASSERT(a.size()==this->argument_size());
    set_rounding_upward();
    Float e=0;
    for(iterator iter=this->_expansion.begin(); iter!=this->end(); ) {
        bool erase=false;
        for(uint i=0; i!=a.size(); ++i) {
            if(a[i] < iter->first[i]) {
                erase=true; 
                break;
            }
        }
        if(erase) {
            e+=abs(iter->second);
            this->_expansion.data().erase(iter++);
        } else {
            ++iter;
        }
    }
    this->_error.u+=e;
    this->_error.l=-this->_error.u;
    set_rounding_to_nearest();
    return *this;
}

TaylorVariable&
TaylorVariable::clobber()
{
    this->_error=0;
    return *this;
}

TaylorVariable&
TaylorVariable::clobber(uint o)
{
    for(iterator iter=this->_expansion.begin(); iter!=this->end(); ) {
        const MultiIndex& a=iter->first;
        if(a.degree()>o) {
            this->_expansion.data().erase(iter++);
        } else {
            ++iter;
        }
    }
    this->_error=0;
    return *this;
}

TaylorVariable&
TaylorVariable::clobber(uint so, uint to)
{
    uint n=this->argument_size()-1;
    for(iterator iter=this->_expansion.begin(); iter!=this->end(); ) {
        const MultiIndex& a=iter->first;
        if(a[n]>to || a.degree()>so+a[n]) {
            this->_expansion.data().erase(iter++);
        } else {
            ++iter;
        }
    }
    this->_error=0;
    return *this;
}

void
TaylorVariable::clean()
{
    this->sweep(0.0);
}



TaylorVariable&
operator+=(TaylorVariable& x, const TaylorVariable& y)
{
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    return x.acc(y);
   
}

TaylorVariable&
operator-=(TaylorVariable& x, const TaylorVariable& y)
{
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    return x.acc(neg(y));
}


TaylorVariable&
operator+=(TaylorVariable& x, const Float& c)
{
    return x.acc(c);
}
	
TaylorVariable&
operator-=(TaylorVariable& x, const Float& c)
{
    return x.acc(-c);
}


TaylorVariable&
operator+=(TaylorVariable& x, const Interval& c)
{
    return x.acc(c);
}

TaylorVariable&
operator-=(TaylorVariable& x, const Interval& c)
{
    return x.acc(-c);
}

TaylorVariable&
operator*=(TaylorVariable& x, const Float& c)
{
    return x.scal(c);
}

TaylorVariable&
operator*=(TaylorVariable& x, const Interval& c)
{
    return x.scal(c);
}


TaylorVariable&
operator/=(TaylorVariable& x, const Float& c)
{
    return x.scal(Interval(1.0)/c);
}


TaylorVariable&
operator/=(TaylorVariable& x, const Interval& c)
{
    return x.scal(1.0/c);
}




TaylorVariable 
operator+(const TaylorVariable& x) {
    return x; 
}

TaylorVariable 
operator-(const TaylorVariable& x) {
    return neg(x); 
}


TaylorVariable 
operator+(const TaylorVariable& x, const TaylorVariable& y) {
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    TaylorVariable r(x); r.acc(y); return r; 
}

TaylorVariable 
operator-(const TaylorVariable& x, const TaylorVariable& y) {
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    TaylorVariable r=neg(y); r.acc(x); return r;
}

TaylorVariable 
operator*(const TaylorVariable& x, const TaylorVariable& y) {
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    TaylorVariable r(x.argument_size()); r.acc(x,y); return r;
}

TaylorVariable 
operator/(const TaylorVariable& x, const TaylorVariable& y) {
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    TaylorVariable r(x.argument_size()); r.acc(x,rec(y)); return r;
}



TaylorVariable 
operator+(const TaylorVariable& x, const Float& c) {
    TaylorVariable r(x); r.acc(c); return r;
}

TaylorVariable 
operator-(const TaylorVariable& x, const Float& c) {
    TaylorVariable r(x); r.acc(-c); return r;
}

TaylorVariable 
operator*(const TaylorVariable& x, const Float& c) {
    TaylorVariable r(x); r.scal(c); return r;
}

TaylorVariable 
operator/(const TaylorVariable& x, const Float& c) {
    TaylorVariable r(x); r.scal(Interval(1)/c); return r;
}

TaylorVariable 
operator+(const Float& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.acc(c); return r;
}

TaylorVariable 
operator-(const Float& c, const TaylorVariable& x) {
    TaylorVariable r=neg(x); r.acc(c); return r;
}

TaylorVariable 
operator*(const Float& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.scal(c); return r;
}

TaylorVariable 
operator/(const Float& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.scal(Interval(1)/c); return r;
}



TaylorVariable 
operator+(const TaylorVariable& x, const Interval& c) {
    TaylorVariable r(x); r.acc(c); return r;
}

TaylorVariable 
operator-(const TaylorVariable& x, const Interval& c) {
    TaylorVariable r(x); r.acc(-c); return r;
}

TaylorVariable 
operator*(const TaylorVariable& x, const Interval& c) {
    TaylorVariable r(x); r.scal(c); return r;
}

TaylorVariable 
operator/(const TaylorVariable& x, const Interval& c) {
    TaylorVariable r(x); r.scal(1/c); return r;
}

TaylorVariable 
operator+(const Interval& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.acc(c); return r;
}

TaylorVariable 
operator-(const Interval& c, const TaylorVariable& x) {
    TaylorVariable r=neg(x); r.acc(c); return r;
}

TaylorVariable 
operator*(const Interval& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.scal(c); return r;
}

TaylorVariable 
operator/(const Interval& c, const TaylorVariable& x) {
    TaylorVariable r(x); r.scal(1/c); return r;
}




TaylorVariable
add_cosy(const TaylorVariable& x, const Float& c) {
    //std::cerr<<"add_cosy(TaylorVariable,Float)"<<std::endl;
    TaylorVariable r(x);
    r.expansion().set_value(r.expansion().value()+c);
    Float t=abs(r.expansion().value());
    r.error()=r.error()+TaylorVariable::em*Interval(-t,t);
    return r;
}


TaylorVariable
mul_cosy(const TaylorVariable& x, const Float& c) {
    //std::cerr<<"mul_cosy(TaylorVariable,Float)"<<std::endl;
    uint as=x.argument_size();
    const SparseDifferential<Float>& expansion=x.expansion();
    const Interval& I=x.error();

    static const double em=2.2204460492503131e-16;
    static const double ec=em/2;
    
    TaylorVariable r(as);
    Float s=0;
    Float t=0;
    Float rj;
    for(SparseDifferential<Float>::const_iterator iter=expansion.begin();
        iter!=expansion.end(); ++iter) 
    {
        const MultiIndex& j=iter->first;
        const Float& xj=iter->second;
        rj=xj*c;
        t=t+abs(rj);
        if(abs(rj)<ec) {
            s=s+abs(rj);
        } else {
            r[j]=rj;
        }
    }
    r.error()=c*I+2*(em*Interval(-t,t))+2*Interval(-s,s);
    return r;
}


TaylorVariable 
add_cosy(const TaylorVariable& x, const TaylorVariable& y) 
{
    //std::cerr<<"add_cosy(TaylorVariable,TaylorVariable)"<<std::endl;
    const Interval& I=x.error();
    const Interval& J=y.error();

    static const double em=2.2204460492503131e-16;
    static const double ec=em/2;
    
    TaylorVariable r=x;
    Float s=0; // sweep error
    Float t=0; // tally
    Float rj;
    for(SparseDifferential<Float>::const_iterator yiter=y.expansion().begin();
        yiter!=y.expansion().end(); ++yiter) 
    {
        const MultiIndex& j=yiter->first;
        const Float& yj=yiter->second;
        SparseDifferential<Float>::const_iterator xiter=x.expansion().data().find(j);
        if(xiter!=x.expansion().end()) {
            const Float& xj=xiter->second;
            rj=xj+yj;
            t=t+max(abs(xj),abs(yj));
            if(rj<ec) {
                s=s+abs(rj);
            } else {
                r[j]=rj;
            }
        } else {
            r[j]=yj;
        }
    }
    //std::cerr<<" s="<<s<<" t="<<t<<std::endl;
    r.error()=I+J+(2*em)*Interval(-t,t)+2*Interval(-s,s);
    return r;
}


TaylorVariable 
mul_cosy(const TaylorVariable& x, const TaylorVariable& y) 
{
    static const uint MAX_DEGREE=20;
    const Interval& xI=x.error();
    const Interval& yI=y.error();
    TaylorVariable r(x.argument_size());
    MultiIndex rj(r.argument_size());
    Interval& rJ=r.error();
    Float t =0;
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        Interval Jtmp = 0;
        const MultiIndex& xj=xiter->first;
        const Float& xv=xiter->second;
        for(TaylorVariable::const_iterator yiter=x.begin(); yiter!=y.end(); ++yiter) {
            const MultiIndex& yj=yiter->first;
            const Float& yv=yiter->second;
            if(xj.degree()+yj.degree() <= MAX_DEGREE) {
                rj=xj+yj;
                Float& rv=r[rj];
                Float p=xv*yv;
                t=t+abs(p);
                t=t+max(abs(rv),abs(p));
                rv=rv+p;
            } else {
                Jtmp=Jtmp+Interval(-yv,yv);
            }
        }
        rJ=rJ+Interval(-xv,xv)*(Jtmp+yI);
    }

    Interval Jtmp=0;
    for(TaylorVariable::const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
        const Float& yv=yiter->second;
        Jtmp+=Interval(-yv,+yv);
    }
    rJ=rJ+xI*(yI+Jtmp);
    
    Float s=0;
    for(TaylorVariable::iterator riter=r.begin(); riter!=r.end(); ++riter) {
        Float& rv=riter->second;
        if(abs(rv)<TaylorVariable::ec && riter->first.degree()!=0) {
            s=s+abs(rv);
            TaylorVariable::iterator tmpriter=riter;
            --riter;
            r.expansion().data().erase(tmpriter);
        }
    }
    rJ=rJ+(2*TaylorVariable::em)*Interval(-t,t)+2*Interval(-s,s);
    
    return r;
}


 
TaylorVariable 
mul_ivl(const TaylorVariable& x, const TaylorVariable& y) 
{
    typedef std::map<MultiIndex,Float>::const_iterator flt_const_iterator;
    typedef std::map<MultiIndex,Interval>::const_iterator ivl_const_iterator;
    TaylorVariable r(x.argument_size());
    std::map<MultiIndex,Interval> z;
 
    set_rounding_upward();
    for(flt_const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        for(flt_const_iterator yiter=x.begin(); yiter!=y.end(); ++yiter) {
            const Float& xv=xiter->second;
            const Float& yv=yiter->second;
            Interval& zv=z[xiter->first+yiter->first];
            zv.u=add_rnd(zv.u,mul_rnd(xv,yv));
            zv.l=add_opp(zv.l,mul_opp(xv,yv));
        }
    }
    
    set_rounding_to_nearest();
    for(ivl_const_iterator ziter=z.begin(); ziter!=z.end(); ++ziter) {
        const Interval& zv=ziter->second;
        r[ziter->first]=(zv.u+zv.l)/2;
    }
    
    set_rounding_upward();
    Float re=0;
    for(ivl_const_iterator ziter=z.begin(); ziter!=z.end(); ++ziter) {
        const Interval& zv=ziter->second;
        const Float& rv=r[ziter->first];
        re+=max(zv.u-rv,rv-zv.l);
    }
    
    Float xs=0;
    for(flt_const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        xs+=abs(xiter->second);
    }
    
    Float ys=0;
    for(flt_const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
        ys+=abs(yiter->second);
    }
         
    Float xe=x.error().u;
    Float ye=y.error().u;
    r.error() = re + xs*ye + ys*xe;
    
    return r;
}
    
TaylorVariable max(const TaylorVariable& x, const TaylorVariable& y) {
    Interval xr=x.range();
    Interval yr=y.range();
    if(xr.lower()>=yr.upper()) {
        return x;
    } else if(yr.lower()>=xr.upper()) {
        return y;
    } else {
        TaylorVariable z(x.argument_size());
        z.value()=max(x.value(),y.value());
        z.error()=(max(xr,yr)-max(x.value(),y.value()));
        return z;
    }
}




TaylorVariable
add_rounded(const TaylorVariable& x, const Float& c) {
    //std::cerr<<"add_rounded(TaylorVariable,Float)"<<std::endl;
    volatile Float tmp;
    TaylorVariable r(x);
    const Float& xv=x.expansion().value();
    Interval& re=r.error();
    Float rva=xv+c;
    set_rounding_upward();
    volatile Float rvu=xv+c;
    volatile Float mrvl=(-xv)-c;
    //std::cerr<<"rvl="<<-mrvl<<" rva="<<rva<<" rvu="<<rvu<<"\n";
    //std::cerr<<"rvle="<<rva+mrvl<<" rvue="<<rvu-rva<<"\n";
    Float rve=max(rvu-rva,rva+mrvl);
    r.value()=(rva);
    re.u+=rve; 
    tmp=-re.l+rve; re.l=-tmp;
    set_rounding_to_nearest();
    return r;
}


TaylorVariable
mul_rounded(const TaylorVariable& x, const Float& c) 
{
    //std::cerr<<"mul_rounded(TaylorVariable,Float)"<<std::endl;
    volatile Float tmp;
    TaylorVariable r(x.argument_size());
    Interval& re=r.error();
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        r[xiter->first]=xiter->second*c;
    }
    set_rounding_upward();
    volatile Float eu=0;
    volatile Float el=0;
    for(TaylorVariable::const_iterator riter=r.begin(); riter!=r.end(); ++riter) {
        const MultiIndex& j=riter->first;
        TaylorVariable::const_iterator xiter=x.expansion().data().find(j);
        assert(xiter!=x.end());
        const Float& xv=xiter->second;
        const Float& rv=riter->second;
        tmp=xv*c;
        eu+=(tmp-rv);
        tmp=(-xv)*c;
        el+=rv+tmp;
    }
    Float e=max(eu,el);
    re.u=x.error().u*c+e;
    re.l=-re.u;
    set_rounding_to_nearest();
    return r;
}

TaylorVariable
add_rounded(const TaylorVariable& x, const TaylorVariable& y) 
{
    //std::cerr<<"add_rounded(TaylorVariable,TaylorVariable)"<<std::endl;
    volatile Float tmp;
    TaylorVariable r(x);
    Interval& re=r.error();
    for(TaylorVariable::const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
        r[yiter->first]+=yiter->second;
    }
    set_rounding_upward();
    volatile Float eu=0;
    volatile Float el=0;
    for(TaylorVariable::const_iterator riter=y.begin(); riter!=y.end(); ++riter) {
        const MultiIndex& j=riter->first;
        TaylorVariable::const_iterator xiter=x.expansion().data().find(j);
        TaylorVariable::const_iterator yiter=y.expansion().data().find(j);
        const Float& xv=xiter->second;
        const Float& yv=yiter->second;
        const Float& rv=riter->second;
        tmp=xv+yv;
        eu+=(tmp-rv);
        tmp=(-xv)-yv;
        el+=rv+tmp;
    }
    re.u=x.error().u+y.error().u+eu;
    tmp=-x.error().l-y.error().l+el;
    re.l=-tmp;
    set_rounding_to_nearest();
    return r;
}

TaylorVariable
mul_rounded(const TaylorVariable& x, const TaylorVariable& y) 
{
    //std::cerr<<"mul_rounded(TaylorVariable,TaylorVariable)"<<std::endl;
    TaylorVariable r(x.argument_size());
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        for(TaylorVariable::const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
            r[xiter->first+yiter->first]+=xiter->second*yiter->second;
        }
    }
    
    Float t=0;
    for(TaylorVariable::const_iterator riter=r.begin(); riter!=r.end(); ++riter) {
        const MultiIndex& rj=riter->first;
        const Float& rv=riter->second;
        volatile Float ru=0;
        volatile Float mrl=0;
        for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
            const MultiIndex& xj=xiter->first;
            MultiIndex yj=rj-xj;
            TaylorVariable::const_iterator yiter=y.expansion().data().find(yj);
            if(yiter!=y.expansion().data().end()) {
                const Float& xv=xiter->second;
                const Float& yv=yiter->second;
                ru+=xv*yv;
                volatile float tmp=-xv;
                mrl+=tmp*yv;
            }
        }
        Float re=max(ru-rv,rv+mrl);
        t+=re;
    }
    
    Float xs=0;
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        xs+=abs(xiter->second);
    }
    Float ys=0;
    for(TaylorVariable::const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
        ys+=abs(yiter->second);
    }
    
    Float xe=x.error().u;
    Float ye=y.error().u;
    Float re=xe*ys+ye*xs+t;
    r.error()=Interval(-re,re);
    
    return r;
}




TaylorVariable min(const TaylorVariable& x, const TaylorVariable& y) {
    return -max(-x,-y);
}

TaylorVariable abs(const TaylorVariable& x) {
    Interval xr=x.range();    
    if(xr.lower()>=0.0) {
        return x;
    } else if(xr.upper()<=0.0) {
        return -x;
    } else {
        TaylorVariable z(x.argument_size());
        Float xv=x.value();
        z.value()=(abs(xv));
        z.error()=(abs(xr)-abs(xv));
        return z;
    }

}

Interval _sum(const TaylorVariable& x) {
    typedef TaylorVariable::const_iterator const_iterator;
    Interval r=0;
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        r+=xiter->second;
    }
    return r;
}

/*
TaylorVariable add(const TaylorVariable& x, const TaylorVariable& y) {
    TaylorVariable r=x;
    r+=y;
    return r;
}

TaylorVariable mul(const TaylorVariable& x, const TaylorVariable& y) {
    ARIADNE_ASSERT(x.argument_size()==y.argument_size());
    typedef TaylorVariable::const_iterator const_iterator;
    TaylorVariable r(x.argument_size());
    Interval& e=r.error();
    for(const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        for(const_iterator yiter=y.begin(); yiter!=y.end(); ++yiter) {
            acc(e,r[xiter->first+yiter->first],xiter->second,yiter->second);
        }
    }
    e += x.error() * _sum(y) + _sum(x) * y.error() + x.error() * y.error();
    return r;
}
*/

TaylorVariable neg(const TaylorVariable& x) {
    return TaylorVariable(-x.expansion(),-x.error());
}

Vector<Interval> 
TaylorVariable::domain() const
{
    return Vector<Interval>(this->argument_size(),Interval(-1,1));
}

Interval 
TaylorVariable::range() const {
    Interval r=this->error();
    for(const_iterator iter=this->begin(); iter!=this->end(); ++iter) {
        if(iter->first.degree()==0) {
            r+=iter->second;
        } else {
            r+=abs(iter->second)*Interval(-1,1);
        }
    }
    return r;
}
 

Interval 
TaylorVariable::evaluate(const Vector<Interval>& v) const
{
    ARIADNE_ASSERT(this->argument_size()==v.size());
    ARIADNE_ASSERT(subset(v,this->domain()));
    Interval r=this->error();
    for(const_iterator iter=this->begin(); iter!=this->end(); ++iter) {
        Interval t=iter->second;
        for(uint j=0; j!=iter->first.size(); ++j) {
            t*=pow(v[j],iter->first[j]);
        }
        r+=t;
    }
    return r;
}

template<class X> class Series;  
typedef Series<Interval>(*series_function_pointer)(uint,const Interval&);

struct TaylorSeries {
    typedef Series<Interval>(*series_function_pointer)(uint,const Interval&); 
    TaylorSeries(uint d) : expansion(d+1), error(0) { }
    TaylorSeries(uint degree, series_function_pointer function, 
                 const Float& centre, const Interval& domain);
    uint degree() const { return expansion.size()-1; }
    Float& operator[](uint i) { return expansion[i]; }
    array<Float> expansion;
    Interval error;
    void sweep(Float e) { 
        for(uint i=0; i<=degree(); ++i) {
            if(abs(expansion[i])<=e) { 
                error+=expansion[i]*Interval(-1,1); 
                expansion[i]=0; } } }
};

TaylorSeries::TaylorSeries(uint d, series_function_pointer fn, 
                           const Float& c, const Interval& r)
    : expansion(d+1), error(0) 
{
    Series<Interval> centre_series=fn(d,Interval(c));
    Series<Interval> range_series=fn(d,r);
    Interval p=1;
    Interval e=r-c;
    //std::cerr<<"\nc="<<c<<" r="<<r<<" e="<<e<<"\n";
    //std::cerr<<"centre_series="<<centre_series<<"\nrange_series="<<range_series<<"\n";
    for(uint i=0; i!=d; ++i) {
        this->expansion[i]=midpoint(centre_series[i]);
        this->error+=(centre_series[i]-this->expansion[i])*p;
        p*=e;
    }
    //this->expansion[d]=midpoint(centre_series[d]);
    this->expansion[d]=midpoint(range_series[d]);
    this->error+=(range_series[d]-this->expansion[d])*p;
    //std::cerr<<"expansion="<<this->expansion<<"\nerror="<<this->error<<"\n";
}


std::ostream& 
operator<<(std::ostream& os, const TaylorSeries& ts) {
    return os<<"TS("<<ts.expansion<<","<<ts.error<<")";
}


TaylorVariable 
_compose(const TaylorSeries& ts, const TaylorVariable& tv, Float eps)
{
    //std::cerr<<"\ncompose\n";
    //std::cerr<<"\n  ts="<<ts<<"\n  tv="<<tv<<"\n";
    Float& vref=const_cast<Float&>(tv.expansion().value());
    Float vtmp=vref; 
    vref=0.0;
    TaylorVariable r(tv.argument_size());
    r+=ts.expansion[ts.expansion.size()-1];
    for(uint i=1; i!=ts.expansion.size(); ++i) {
        //std::cerr<<"    r="<<r<<std::endl;
        r=r*tv;
        r+=ts.expansion[ts.expansion.size()-i-1];
        r.sweep(eps);
    }
    //std::cerr<<"    r="<<r<<std::endl;
    r+=ts.error;
    //std::cerr<<"    r="<<r<<std::endl;
    vref=vtmp;
    return r;
}

TaylorVariable 
compose(const TaylorSeries& ts, const TaylorVariable& tv)
{
    return _compose(ts,tv,TaylorVariable::ec);
}


// Compose using the Taylor formula directly. The final term is the Taylor series computed
// over the range of the series. This method tends to suffer from blow-up of the 
// truncation error
TaylorVariable 
_compose1(const series_function_pointer& fn, const TaylorVariable& tv, Float eps)
{
    static const uint DEGREE=18;
    static const Float TRUNCATION_ERROR=1e-8;
    uint d=DEGREE;
    Float c=tv.expansion().value();
    Interval r=tv.range();
    Series<Interval> centre_series=fn(d,Interval(c));
    Series<Interval> range_series=fn(d,r);
    
    Float truncation_error_estimate=mag(range_series[d])*pow(mag(r-c),d);
    if(truncation_error_estimate>TRUNCATION_ERROR) {
        std::cerr<<"Warning: Truncation error estimate "<<truncation_error_estimate
                 <<" is greater than maximum allowable truncation error "<<TRUNCATION_ERROR<<"\n";
    }

    TaylorVariable x=tv-c;
    TaylorVariable res(tv.argument_size());
    res+=range_series[d];
    for(uint i=0; i!=d; ++i) {
        //std::cerr<<"i="<<i<<" r="<<res<<"\n";
        res=centre_series[d-i-1]+x*res;
        res.sweep(eps);
    }
    //std::cerr<<"i="<<d<<" r="<<res<<"\n";
    return res;
}

// Compose using the Taylor formula with a constant truncation error. This method
// is usually better than _compose1 since there is no blow-up of the trunction 
// error. The radius of convergence of this method is still quite low,
// typically only half of the radius of convergence of the power series itself
TaylorVariable 
_compose2(const series_function_pointer& fn, const TaylorVariable& tv, Float eps)
{
    static const uint DEGREE=20;
    static const Float TRUNCATION_ERROR=1e-8;
    uint d=DEGREE;
    Float c=tv.expansion().value();
    Interval r=tv.range();
    Series<Interval> centre_series=fn(d,Interval(c));
    Series<Interval> range_series=fn(d,r);
    
    //std::cerr<<"c="<<c<<" r="<<r<<" r-c="<<r-c<<" e="<<mag(r-c)<<"\n";
    //std::cerr<<"cs[d]="<<centre_series[d]<<" rs[d]="<<range_series[d]<<"\n";
    //std::cerr<<"cs="<<centre_series<<"\nrs="<<range_series<<"\n";
    Float truncation_error=mag(range_series[d]-centre_series[d])*pow(mag(r-c),d);
    //std::cerr<<"te="<<truncation_error<<"\n";
    if(truncation_error>TRUNCATION_ERROR) {
        std::cerr<<"Warning: Truncation error estimate "<<truncation_error
                 <<" is greater than maximum allowable truncation error "<<TRUNCATION_ERROR<<"\n";
    }

    TaylorVariable x=tv-c;
    TaylorVariable res(tv.argument_size());
    res+=centre_series[d];
    for(uint i=0; i!=d; ++i) {
        res=centre_series[d-i-1]+x*res;
        res.sweep(eps);
    }
    res+=truncation_error*Interval(-1,1);
    return res;
}


// Compose using the Taylor formula with a constant truncation error. This method
// is usually better than _compose1 since there is no blow-up of the trunction 
// error. This method is better than _compose2 since the truncation error is
// assumed at the ends of the intervals
TaylorVariable 
_compose3(const series_function_pointer& fn, const TaylorVariable& tv, Float eps)
{
    static const uint DEGREE=20;
    static const Float TRUNCATION_ERROR=1e-8;
    uint d=DEGREE;
    Float c=tv.expansion().value();
    Interval r=tv.range();
    Series<Interval> centre_series=fn(d,Interval(c));
    Series<Interval> range_series=fn(d,r);
    
    //std::cerr<<"c="<<c<<" r="<<r<<" r-c="<<r-c<<" e="<<mag(r-c)<<"\n";
    //std::cerr<<"cs[d]="<<centre_series[d]<<" rs[d]="<<range_series[d]<<"\n";
    //std::cerr<<"cs="<<centre_series<<"\nrs="<<range_series<<"\n";
    Interval se=range_series[d]-centre_series[d];
    Interval e=r-c;
    Interval p=pow(e,d-1); p.l*=-e.l; p.u*=e.u;
    //std::cerr<<"se="<<se<<" e="<<e<<" p="<<p<<std::endl;
    // FIXME: Here we assume the dth derivative of f is monotone increasing
    Float truncation_error=max(se.l*p.l,se.u*p.u);
    //std::cerr<<"te="<<truncation_error<<"\n";
    if(truncation_error>TRUNCATION_ERROR) {
        std::cerr<<"Warning: Truncation error estimate "<<truncation_error
                 <<" is greater than maximum allowable truncation error "<<TRUNCATION_ERROR<<"\n";
    }

    TaylorVariable x=tv-c;
    TaylorVariable res(tv.argument_size());
    res+=centre_series[d];
    for(uint i=0; i!=d; ++i) {
        res=centre_series[d-i-1]+x*res;
        res.sweep(eps);
    }
    res+=truncation_error*Interval(-1,1);
    return res;
}


TaylorVariable 
_compose(const series_function_pointer& fn, const TaylorVariable& tv, Float eps) {
    return _compose3(fn,tv,eps);
}


TaylorVariable sqr(const TaylorVariable& x) {
    TaylorVariable r=x*x;
    return r;
}

TaylorVariable pow(const TaylorVariable& x, int n) {
    TaylorVariable r(x.argument_size()); r+=1;
    TaylorVariable p(x);
    while(n) {
        if(n%2) { r=r*p; } 
        p=sqr(p);
        n/=2;
    }
    return r;
}

TaylorVariable sqrt(const TaylorVariable& x) {
    //std::cerr<<"rec(TaylorVariable)\n";
    // Use a special routine to minimise errors
    static const Float max_trunc_err=1e-12;
    static const Float max_sweep_err=1e-12;
    // Given range [rl,ru], scale by constant a such that rl/a=1-d; ru/a=1+d
    Interval r=x.range();
    assert(r.l>0);
    Float a=(r.l+r.u)/2;
    set_rounding_upward();
    Float eps=(r.u-r.l)/(r.u+r.l);
    set_rounding_to_nearest();
    assert(eps<1);
    uint d=uint(log((1-eps)*max_trunc_err)/log(eps)+1);
    //std::cerr<<"x="<<x<<std::endl;
    //std::cerr<<"x/a="<<x/a<<" a="<<a<<std::endl;
    TaylorVariable y=(x/a)-1.0;
    //std::cerr<<"y="<<y<<std::endl;
    TaylorVariable z(x.argument_size());
    Series<Interval> sqrt_series=Series<Interval>::sqrt(d,Interval(1));
    //std::cerr<<"sqrt_series="<<sqrt_series<<std::endl;
    //std::cerr<<"y="<<y<<std::endl;
    z+=sqrt_series[d-1];
    for(uint i=0; i!=d; ++i) {
        z=sqrt_series[d-i-1] + z * y;
        z.sweep(max_sweep_err);
        //std::cerr<<"z="<<z<<std::endl;
    }
    Float trunc_err=pow(eps,d)/(1-eps)*mag(sqrt_series[d]);
    //std::cerr<<"te="<<trunc_err<<" te*[-1,+1]="<<trunc_err*Interval(-1,1)<<std::endl;
    z.error()+=(trunc_err*Interval(-1,1));
    //std::cerr<<"z="<<z<<std::endl;
    Interval sqrta=sqrt(Interval(a));
    //std::cerr<<"sqrt(a)="<<sqrta<<std::endl;
    z*=sqrt(Interval(a));
    //std::cerr<<"z="<<z<<std::endl;
    return z;
}

TaylorVariable rec(const TaylorVariable& x) {
    //std::cerr<<"rec(TaylorVariable)\n";
    // Use a special routine to minimise errors
    static const Float max_trunc_err=1e-12;
    static const Float max_sweep_err=1e-12;
    // Given range [rl,ru], scale by constant a such that rl/a=1-d; ru/a=1+d
    Interval r=x.range();
    assert(r.l>0 || r.u<0);
    Float a=(r.l+r.u)/2;
    set_rounding_upward();
    Float eps=abs((r.u-r.l)/(r.u+r.l));
    set_rounding_to_nearest();
    assert(eps<1);
    uint d=uint(log((1-eps)*max_trunc_err)/log(eps))+1;
    //std::cerr<<"  x="<<x<<"\n";
    TaylorVariable y=1-(x/a);
    //std::cerr<<"  y="<<y<<"\n";
    TaylorVariable z(x.argument_size());
    z+=Float(d%2?-1:+1);
    for(uint i=0; i!=d; ++i) {
        z=1.0 + z * y;
        z.sweep(max_sweep_err);
    }
    //std::cerr<<"  z="<<z<<"\n";
    Float te=pow(eps,d)/(1-eps);
    //std::cerr<<"  te="<<te<<"\n";
    set_rounding_upward();
    Float nze=te+z.error().u;
    //std::cerr<<"  nze="<<nze<<"\n";

    set_rounding_to_nearest();
    z.set_error(nze);
    //z.error().u=nze; 
    //z.error().l=-nze;
    //std::cerr<<"  z="<<z<<"\n";
    z/=a;
    //std::cerr<<"  z="<<z<<"\n";
    return z;
}

TaylorVariable log(const TaylorVariable& x) {
    // Use a special routine to minimise errors
    static const Float max_trunc_err=1e-12;
    static const Float max_sweep_err=1e-12;
    // Given range [rl,ru], scale by constant a such that rl/a=1-d; ru/a=1+d
    Interval r=x.range();
    assert(r.l>0);
    Float a=(r.l+r.u)/2;
    set_rounding_upward();
    Float eps=(r.u-r.l)/(r.u+r.l);
    set_rounding_to_nearest();
    assert(eps<1);
    uint d=uint(log((1-eps)*max_trunc_err)/log(eps)+1);
    TaylorVariable y=x/a-1;
    TaylorVariable z(x.argument_size());
    z+=Float(d%2?-1:+1)/d;
    for(uint i=1; i!=d; ++i) {
        z=Float((d-i)%2?+1:-1)/(d-i) + z * y;
        z.sweep(max_sweep_err);
    }
    z=z*y;
    z.sweep(max_sweep_err);
    Float trunc_err=pow(eps,d)/(1-eps)/d;
    return z+log(Interval(a))+trunc_err*Interval(-1,1);
}

TaylorVariable exp(const TaylorVariable& x) {
    static const uint DEG=18;
    return _compose(&Series<Interval>::exp,x,TaylorVariable::ec);
    return compose(TaylorSeries(DEG,&Series<Interval>::exp,
                                x.value(),x.range()),x);
}

TaylorVariable sin(const TaylorVariable& x) {
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::sin,
                                x.value(),x.range()),x);
}

TaylorVariable cos(const TaylorVariable& x) {
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::cos,
                                x.value(),x.range()),x);
}

TaylorVariable tan(const TaylorVariable& x) {
    return sin(x)*rec(cos(x));
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::tan,
                                x.value(),x.range()),x);
}

TaylorVariable asin(const TaylorVariable& x) {
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::asin,
                                x.value(),x.range()),x);
}

TaylorVariable acos(const TaylorVariable& x) {
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::acos,
                                x.value(),x.range()),x);
}

TaylorVariable atan(const TaylorVariable& x) {
    static const uint DEG=18;
    return compose(TaylorSeries(DEG,&Series<Interval>::atan,
                                x.value(),x.range()),x);
}


inline int pow2(uint k) { return 1<<k; }
inline int powm1(uint k) { return (k%2) ? -1 : +1; }



Interval 
evaluate(const TaylorVariable& tv, const Vector<Interval>& x)
{
    return tv.evaluate(x);
}

Vector<Interval> 
evaluate(const Vector<TaylorVariable>& tv, const Vector<Interval>& x)
{
    Vector<Interval> r(tv.size());
    for(uint i=0; i!=r.size(); ++i) {
        r[i]=evaluate(tv[i],x);
    }
    return r;
}


TaylorVariable antiderivative(const TaylorVariable& x, const Interval& dk, uint k) {
    ARIADNE_ASSERT(k<x.argument_size());
    // General case with error analysis
    TaylorVariable r(x.argument_size());
    const Interval& xe=x.error();
    Interval& re=r.error();
    re=xe;
    set_rounding_upward();
    volatile Float dkru=(dk.u-dk.l)/2;
    volatile Float dkrl=-dk.u; dkrl+=dk.l; dkrl/=(-2);
    volatile Float u,ml;
    Float te=0; // Twice the maximum accumulated error
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        const uint c=xiter->first[k]+1;
        const Float& xv=xiter->second;
        volatile Float mxv=-xv;
        if(xv>=0) {
            u=xv*dkru/c;
            ml=mxv*dkrl/c;
        } else {
            u=xv*dkru/c;
            ml=mxv*dkrl/c;
        }
        te+=(u+ml);
    }
    re.u+=te/2; re.l=-re.u;

    set_rounding_to_nearest();
    Float dkr=(dk.u-dk.l)/2;
    MultiIndex rindex(r.argument_size());
    for(TaylorVariable::const_iterator xiter=x.begin(); xiter!=x.end(); ++xiter) {
        const uint c=xiter->first[k]+1;
        rindex=xiter->first;
        rindex.set(k,c);
        r[rindex]=xiter->second*dkr/c;
    }

    return r;
}

Vector<TaylorVariable> antiderivative(const Vector<TaylorVariable>& x, const Interval& dk, uint k) {
    ARIADNE_ASSERT(k<x.argument_size());
    Vector<TaylorVariable> r(x.size());
    for(uint i=0; i!=x.size(); ++i) {
        r[i]=antiderivative(x[i],dk,k);
    }
    return r;
}


pair<TaylorVariable,TaylorVariable>
split(const TaylorVariable& tv, uint j) 
{
    ARIADNE_ASSERT(j<tv.argument_size());
    uint as=tv.argument_size();
    uint deg=tv.expansion().degree();
    const SparseDifferential<Float>& expansion=tv.expansion();

    MultiIndex index(as);
    Float value;
    
    SparseDifferential<Float> expansion1(as,deg);
	for(SparseDifferential<Float>::const_iterator iter=expansion.begin();
        iter!=expansion.end(); ++iter) 
    {
        index=iter->first;
        value=iter->second;
        uint k=index[j];
        for(uint l=0; l<=k; ++l) {
            index.set(j,l);
            expansion1[index] += bin(k,l) * value / pow2(k);
        }
    }

    SparseDifferential<Float> expansion2(as,deg);
	for(SparseDifferential<Float>::const_iterator iter=expansion.begin();
        iter!=expansion.end(); ++iter) 
    {
        index=iter->first;
        value=iter->second;
        uint k=index[j];
        for(uint l=0; l<=k; ++l) {
            index.set(j,l);
            // Need brackets in expression below to avoid converting negative signed
            // integer to unsigned
            expansion2[index] += powm1(k-l) * (bin(k,l) * value / pow2(k));
        }
    }
    // FIXME: Add roundoff errors when computing new expansions
    
    Interval error1=tv.error();
    Interval error2=tv.error();
        
    return make_pair(TaylorVariable(expansion1,error1),TaylorVariable(expansion2,error2));
}


pair< Vector<TaylorVariable>, Vector<TaylorVariable> >
split(const Vector<TaylorVariable>& tv, uint j) 
{
    ARIADNE_ASSERT(j<tv.argument_size());
    Vector<TaylorVariable> r1(tv.size());
    Vector<TaylorVariable> r2(tv.size());
    for(uint i=0; i!=tv.size(); ++i) {
        make_lpair(r1[i],r2[i])=split(tv[i],j);
    }
    return make_pair(r1,r2);
}
    
TaylorVariable
unscale(const TaylorVariable& tv, const Interval& ivl) 
{
    // Scale tv so that the interval ivl maps into [-1,1]
    // The result is given by  (tv-c)*s where c is the centre
    // and s the reciprocal of the radius of ivl
    const Float& l=ivl.l;
    const Float& u=ivl.u;
    
    TaylorVariable r=tv;
    Interval c=Interval(l/2)+Interval(u/2);
    Interval s=2/(Interval(u)-Interval(l));
    r-=c;
    r*=s;

    return r;
}

Vector<TaylorVariable>
unscale(const Vector<TaylorVariable>& tvs, const Vector<Interval>& ivls) 
{
    Vector<TaylorVariable> r(tvs.size());
    for(uint i=0; i!=r.size(); ++i) {
        r[i]=unscale(tvs[i],ivls[i]);
    }
    return r;
}

 
TaylorVariable
scale(const TaylorVariable& tv, const Interval& ivl) 
{
    // Scale tv so that the interval [-1,1] maps into ivl
    // The result is given by  (tv*s)+c where c is the centre
    // and r the radius of ivl
    const Float& l=ivl.l;
    const Float& u=ivl.u;
    
    TaylorVariable r=tv;
    Interval c=add_ivl(l/2,u/2);
    Interval s=sub_ivl(u/2,l/2);
    r*=s;
    r+=c;

    return r;
}

Vector<TaylorVariable>
scale(const Vector<TaylorVariable>& tvs, const Vector<Interval>& ivls) 
{
    Vector<TaylorVariable> r(tvs.size());
    for(uint i=0; i!=r.size(); ++i) {
        r[i]=scale(tvs[i],ivls[i]);
    }
    return r;
}

 
       

bool 
refines(const TaylorVariable& tv1, const TaylorVariable& tv2)
{
    ARIADNE_ASSERT(tv1.argument_size()==tv2.argument_size());
    TaylorVariable d=tv2;
    d.error()=0.0;
    d-=tv1;

    set_rounding_upward();
    Float e=d.error().u;
    for(TaylorVariable::const_iterator iter=d.begin(); iter!=d.end(); ++iter) {
        e+=abs(iter->second);
    }
    set_rounding_to_nearest();
   return e <= tv2.error().u;
}

bool 
refines(const Vector<TaylorVariable>& tv1, const Vector<TaylorVariable>& tv2)
{
    ARIADNE_ASSERT(tv1.size()==tv2.size());
    for(uint i=0; i!=tv1.size(); ++i) {
        if(!refines(tv1,tv2)) { return false; }
    }
    return true;
}


void
_compose1(Vector<TaylorVariable>& r,
          const Vector<TaylorVariable>& x, 
          const Vector<TaylorVariable>& ys)
{
    //std::cerr<<"x ="<<x<<"\n";
    //std::cerr<<"ys="<<ys<<"\n";
    // Brute-force, unoptimised approach
    for(uint i=0; i!=x.size(); ++i) {
        r[i]=TaylorVariable::constant(ys.argument_size(),0.0);
        r[i].set_error(x[i].error());
        for(TaylorVariable::const_iterator iter=x[i].begin(); iter!=x[i].end(); ++iter) {
            TaylorVariable t=TaylorVariable::constant(ys.argument_size(),iter->second);
            for(uint j=0; j!=iter->first.size(); ++j) {
                TaylorVariable p=pow(ys[j],iter->first[j]);
                t=t*p;
            }
            r[i]+=t;
            //std::cerr<<"a="<<iter->first<<" c="<<iter->second<<" t="<<t<<" r="<<r<<"\n";
        }
    }
}


Vector<TaylorVariable> 
compose(const Vector<TaylorVariable>& x, 
        const Vector<Interval>& bx, 
        const Vector<TaylorVariable>& y)
{
    if(x.size()==0) { return x; }

    ARIADNE_ASSERT(x.size()>0);
    ARIADNE_ASSERT(x.argument_size()==bx.size());
    ARIADNE_ASSERT(y.size()==bx.size());
    ARIADNE_ASSERT(y.argument_size()>=0);



    Vector<TaylorVariable> ys(y.size());
    for(uint i=0; i!=y.size(); ++i) {
        ys[i]=unscale(y[i],bx[i]);
    }

    Vector<TaylorVariable> r(x.size());
    _compose1(r,x,ys);
    
    //FIXME: Remove setting error to zero
    for(uint i=0; i!=r.size(); ++i) { r[i].sweep(1e-8); }
    for(uint i=0; i!=r.size(); ++i) { r[i].clobber(); }

    ARIADNE_ASSERT(r.result_size()==x.result_size());
    ARIADNE_ASSERT(r.argument_size()==y.argument_size());

    return r;
}

TaylorVariable 
compose(const TaylorVariable& x, 
        const Vector<Interval>& bx, 
        const Vector<TaylorVariable>& y)
{
    Vector<TaylorVariable> xv(1,x);
    Vector<TaylorVariable> rv=compose(xv,bx,y);
    return rv[0];
}
    
TaylorVariable 
compose(const TaylorVariable& x, 
        const Interval& b, 
        const TaylorVariable& y)
{
    Vector<TaylorVariable> xv(1,x);
    Vector<Interval> bv(1,b);
    Vector<TaylorVariable> yv(1,y);
    Vector<TaylorVariable> rv=compose(xv,bv,yv);
    return rv[0];
}
    
TaylorVariable 
compose(const TaylorVariable& x, 
        const TaylorVariable& y)
{
    Vector<TaylorVariable> xv(1,x);
    Vector<Interval> bv(1,Interval(-1,1));
    Vector<TaylorVariable> yv(1,y);
    Vector<TaylorVariable> rv=compose(xv,bv,yv);
    return rv[0];
}
    

Vector<TaylorVariable>
compose(const Matrix<Float>& A, 
        const Vector<TaylorVariable>& x)
{
    ARIADNE_ASSERT(A.column_size()==x.size());

    Vector<TaylorVariable> r=TaylorVariable::zeroes(A.row_size(),A.column_size());
    for(uint i=0; i!=A.row_size(); ++i) {
        for(uint j=0; j!=A.row_size(); ++j) {
            r[i]+=A[i][j]*x[j];
        }
    }
    return r;
}

    

Vector<TaylorVariable>
_compose(const Vector<TaylorVariable>& x, const Vector<TaylorVariable>& y)
{
    Vector<Interval> d(y.size(),Interval(-1,+1));
    return compose(x,d,y);
}

SparseDifferential<Float> expansion(const TaylorVariable& x, const Vector<Interval>& d) {
    return compose(x,d,TaylorVariable::variables(Vector<Float>(d.size()))).expansion();
}

Vector< SparseDifferential<Float> > expansion(const Vector<TaylorVariable>& x, const Vector<Interval>& d) {
    Vector<TaylorVariable> y=compose(x,d,TaylorVariable::variables(Vector<Float>(d.size())));
    Vector< SparseDifferential<Float> > r(y.size());
    for(uint i=0; i!=r.size(); ++i) { r[i]=y[i].expansion(); }
    return r;
}

inline Vector<Float> centre(const Vector<TaylorVariable>& x) 
{
    Vector<Float> c(x.size());
    for(uint i=0; i!=x.size(); ++i) {
        c[i]=x[i].value();
    }
    return c;
}

Vector<Float> 
value(const Vector<TaylorVariable>& x) {
    Vector<Float> r(x.size());
    for(uint i=0; i!=x.size(); ++i) {
        r[i]=x[i].expansion().value();
    }
    return r;
}


Matrix<Float> 
jacobian(const Vector<TaylorVariable>& x, const Vector<Float>& sy) 
{
    ARIADNE_ASSERT(x.argument_size()==sy.size()); 
    const uint rs=x.size();
    const uint as=sy.size();

    Matrix<Float> J(rs,as);
    for(uint i=0; i!=x.size(); ++i) {
        for(uint j=0; j!=x.argument_size(); ++j) {
            for(TaylorVariable::const_iterator k=x[i].begin(); k!=x[i].end(); ++k) {
                Float c=k->first[j];
                c*=k->second;
                if(c!=0) {
                    for(uint l=0; l!=as; ++l) {
                        if(l==j) {
                            c*=pow(sy[l],k->first[l]-1);
                        } else {
                            c*=pow(sy[l],k->first[l]);
                        }
                    }
                }
                J[i][j]+=c;
            }
        }
    }
    return J;
}


Vector<TaylorVariable>
implicit(const Vector<TaylorVariable>& f, const Vector<Interval>& d)
{
    //std::cerr << ARIADNE_PRETTY_FUNCTION << std::endl;    
    ARIADNE_ASSERT(f.result_size()>=1);
    ARIADNE_ASSERT(f.argument_size()>=f.result_size());

    const uint frs=f.size();
    const uint fas=f.argument_size();
    
    Vector<TaylorVariable> fs=unscale(f,d);
    Vector<TaylorVariable> p=TaylorVariable::zeroes(fas,frs);
    for(uint i=0; i!=frs; ++i) { p[i+(fas-frs)]=TaylorVariable::variable(frs,0.0,i); }
    
    Vector<TaylorVariable> pfs=_compose(fs,p);

    Vector<TaylorVariable> h=TaylorVariable::zeroes(frs,fas-frs);
    for(uint i=0; i!=frs; ++i) { h[i].error()=Interval(-1,+1); }

    Vector<TaylorVariable> id=TaylorVariable::variables(Vector<Float>(frs,0.0));
    //std::cerr<<"\nfs="<<fs<<std::endl;
    //std::cerr<<"pr(fs)="<<pfs<<std::endl;
    //FIXME: Perform proper convergence test
    for(uint k=0; k!=10; ++k) {
        //std::cerr<<"\nh="<<h<<std::endl;
        //std::cerr<<"\njoin(id,h)="<<join(id,h)<<std::endl;
        Vector<TaylorVariable> fxhx=_compose(fs,join(id,h));
        //std::cerr<<"f(x,h(x))="<<fxhx<<std::endl;
        Matrix<Float> J=jacobian(pfs,value(h));
        //std::cerr<<"J="<<J<<std::endl;
        Matrix<Float> Jinv=inverse(J);
        //std::cerr<<"Jinv="<<Jinv<<std::endl;
        h-=compose(Jinv,fxhx);
        for(uint i=0; i!=frs; ++i) { h[i].truncate(18); }
        //std::cerr<<"h="<<h<<std::endl;
        //FIXME: Don't clobber error here!
        for(uint i=0; i!=frs; ++i) { h[i].error()=0.0; }
   }

    h=scale(h,project(d,range(fas-frs,fas)));

    ARIADNE_ASSERT(h.result_size()==f.result_size());
    ARIADNE_ASSERT(h.argument_size()+h.result_size()==f.argument_size());

    return h;
}

std::pair< Interval, Vector<Interval> >
bounds(Vector<TaylorVariable> const& vfm, 
       Vector<Interval> const& d, 
       Float const& hmax,
       Vector<Interval> dmax) 
{ 
    ARIADNE_ASSERT(vfm.result_size()==vfm.argument_size());
    ARIADNE_ASSERT(vfm.result_size()==d.size());
    ARIADNE_ASSERT(vfm.result_size()==dmax.size());
    // Try to find a time h and a set b such that inside(r+Interval<R>(0,h)*vf(b),b) holds

    // Set up constants of the method.
    // TODO: Better estimates of constants
    const Float INITIAL_MULTIPLIER=2;
    const Float MULTIPLIER=1.125;
    //const Float BOX_RADIUS_MULTIPLIER=1.03125;
    const uint EXPANSION_STEPS=8;
    const uint REDUCTION_STEPS=8;
    const uint REFINEMENT_STEPS=4;
  
    Vector<Interval> delta=d-midpoint(d);
  
    Float h=hmax;
    Float hmin=hmax/(1<<REDUCTION_STEPS);
    bool success=false;
    Vector<Interval> b,nb,df;
    Interval ih(0,h);
    while(!success) {
        ARIADNE_ASSERT(h>hmin);
        df=evaluate(vfm,d);
        b=d+(INITIAL_MULTIPLIER*ih)*df+delta;
        for(uint i=0; i!=EXPANSION_STEPS; ++i) {
            df=evaluate(vfm,b);
            nb=d+ih*df;
            if(subset(nb,b)) {
                success=true;
                break;
            } else {
                b=d+MULTIPLIER*ih*df+delta;
            }
        }
        if(!success) {
            h/=2;
            ih=Interval(0,h);
        }
    }

    ARIADNE_ASSERT(subset(nb,b));
  
    Vector<Interval> vfb;
    vfb=evaluate(vfm,b);

    for(uint i=0; i!=REFINEMENT_STEPS; ++i) {
        b=nb;
        vfb=evaluate(vfm,b);
        nb=d+ih*vfb;
        ARIADNE_ASSERT_MSG(subset(nb,b),std::setprecision(20)<<"refinement "<<i<<": "<<nb<<" is not a inside of "<<b);
    }
  
    // Check result of operation
    // We use "possibly" here since the bound may touch 
    ARIADNE_ASSERT(subset(nb,b));
  
    ARIADNE_ASSERT(b.size()==vfm.size());
    return std::make_pair(h,nb);
}

Vector<TaylorVariable> 
flow(const Vector<TaylorVariable>& vf, const Vector<Interval>& d, const Interval& h, const Vector<Interval>& b)
{
    ARIADNE_ASSERT(vf.result_size()==vf.argument_size());
    ARIADNE_ASSERT(vf.size()==d.size());
    ARIADNE_ASSERT(vf.size()==b.size());

    ARIADNE_ASSERT(h.l==0.0 || h.l==-h.u);

    typedef Interval I;
    typedef Float A;
    uint n=vf.size();
    uint so=1u;
    uint to=6u;

    for(uint i=0; i!=n; ++i) { const_cast<TaylorVariable&>(vf[i]).clobber(so,to-1); }

    Interval hh(-h.u,h.u);

    Vector<TaylorVariable> y(n,TaylorVariable(n+1));
    Vector<TaylorVariable> yp(n,TaylorVariable(n+1));
    Vector<TaylorVariable> yz(n,TaylorVariable(n+1));
    for(uint i=0; i!=n; ++i) {
        yz[i].expansion().set_gradient(i,1.0);
        yz[i]*=sub_ivl(d[i].u/2,d[i].l/2);
        yz[i]+=add_ivl(d[i].l/2,d[i].u/2);
    }
    
    

    for(uint i=0; i!=n; ++i) {
        y[i]=yz[i];
    }


    //std::cerr << "\ny[0]=" << y << std::endl << std::endl;
    for(uint j=0; j!=to; ++j) {
        yp=compose(vf,b,y);
        //std::cerr << "yp["<<j+1<<"]=" << yp << std::endl;
        //for(uint i=0; i!=n; ++i) { yp[i].clobber(so,to-1); }
        //std::cerr << "yp["<<j+1<<"]=" << yp << std::endl;
        for(uint i=0; i!=n; ++i) {  
            y[i]=antiderivative(yp[i],hh,n);
            y[i]+=yz[i];
        }
        //std::cerr << "y["<<j+1<<"]=" << y << std::endl << std::endl;
    } 

    if(h.l==0) {
        Vector<TaylorVariable> ht=TaylorVariable::variables(Vector<Float>(n+1,0.0));
        ht[n].expansion().set_value(0.5);
        ht[n].expansion().set_gradient(n,0.5);
        y=compose(y,Vector<Interval>(n+1,Interval(-1,1)),ht);
    }

    for(uint i=0; i!=n; ++i) { y[i].clobber(so,to); }
    for(uint i=0; i!=n; ++i) { y[i].sweep(0.0); }

    ARIADNE_ASSERT(y.result_size()==vf.result_size());
    ARIADNE_ASSERT(y.argument_size()==vf.argument_size()+1);
    return y;
}




TaylorVariable embed(const TaylorVariable& x, uint new_size, uint start)
{
    ARIADNE_ASSERT(x.argument_size()+start<=new_size);
    return TaylorVariable(embed(x.expansion(),new_size,start),x.error());
}

Vector<TaylorVariable> embed(const Vector<TaylorVariable>& x, uint new_size, uint start)
{
    ARIADNE_ASSERT(x.argument_size()+start<=new_size);
    Vector<TaylorVariable> r(x.size());
    for(uint i=0; i!=x.size(); ++i) {
        r[i]=embed(x[i],new_size,start);
    }
    return r;
}


std::string 
TaylorVariable::str() const
{
    std::stringstream ss;
    for(SparseDifferential<Float>::const_iterator iter=this->_expansion.begin();
        iter!=this->_expansion.end(); ++iter) 
    {
        const MultiIndex& j=iter->first;
        const Float& c=iter->second;
        ss << j << ": " << c << "\n";
    }
    ss<<"error: "<<this->_error<<"\n";
    return ss.str();
}


std::ostream& 
operator<<(std::ostream& os, const TaylorVariable& tv) {
    //    return os << "TaylorVariable( expansion=" << tv.expansion() << ", error=" << tv.error() 
    //              << ", range="<<tv.range()<<" )";
    return os << "TaylorVariable(" << tv.expansion().data() << "," << tv.error() << ")"; 
}

 
} //namespace Ariadne


