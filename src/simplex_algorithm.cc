/***************************************************************************
 *            simplex_algorithm.cc
 *
 *  Copyright 2008-10 Pieter Collins
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

#include "config.h"

#include "tuple.h"
#include "numeric.h"
#include "vector.h"
#include "matrix.h"
#include "affine.h"
#include "linear_programming.h"

#include "macros.h"
#include "logging.h"

#include <boost/numeric/ublas/matrix_sparse.hpp>

static const int verbosity=0;

namespace Ariadne {

template<class X> inline Vector<X> operator*(const Matrix<X>& A, const Vector<X>& x) {
    return prod(A,x); }
template<class X> inline Vector<X> operator*(const Vector<X>& y, const Matrix<X>& A) {
    return prod(y,A); }
template<class X> inline Matrix<X> operator*(const Matrix<X>& A, const Matrix<X>& B) {
    return prod(A,B); }

// Threshold for matrix diagonal elements below which it may be considered singular
static const double SINGULARITY_THRESHOLD = std::numeric_limits<double>::epsilon();

// Threshold for slack variables z_N = (c_N - c_B A_B^-1 A_N) below which basis
// may be considered optimal
static const double PROGRESS_THRESHOLD = std::numeric_limits<double>::epsilon() * 1024;

// Threshold for primal variable direction of change below which we do not
// need to see if it would violate its constraints
const double CUTOFF_THRESHOLD=std::numeric_limits<double>::epsilon() * 16;

// Threshold for primal variables to exceed their bounds
const double BOUNDS_TOLERANCE=std::numeric_limits<double>::epsilon() * 4098;

std::ostream& operator<<(std::ostream& os, Slackness t) {
    return os << (t==BASIS ? 'B' : t==LOWER ? 'L' : t==UPPER ? 'U' : t==FIXED ? 'E' : '?');
}


// Compare a matrix with a projection
// Used for checking whether a projection is the identity
template<class Mx, class X> bool operator==(const boost::numeric::ublas::matrix_range<Mx>& A, const boost::numeric::ublas::matrix<X>& B)
{
    if(A.size1()!=B.size1() || A.size2()!=B.size2()) {
        return false;
    }


    for(size_t i=0; i!=A.size1(); ++i) {
        for(size_t j=0; j!=A.size1(); ++j) {
            if(A(i,j)!=B(i,j)) {
                return false;
            }
        }
    }
    return true;
}




// Extend an array of size m to an array of size n
// such that the first m elements are the same,
// and the new array contains the elements [0,n)
array<size_t>
extend_p(const array<size_t>& p, const size_t n)
{
    const size_t m=p.size();
    array<size_t> q(n);
    for(size_t j=0; j!=n; ++j) {
        q[j]=n;
    }
    for(size_t k=0; k!=m; ++k) {
        ARIADNE_ASSERT(p[k]<n);
        ARIADNE_ASSERT(q[p[k]]==n);
        q[p[k]]=k;
    }
    size_t k=m;
    for(size_t j=0; j!=n; ++j) {
        if(q[j]==n) { q[j]=k; ++k; }
    }
    array<size_t> r(n);
    for(size_t j=0; j!=n; ++j) {
        r[q[j]]=j;
    }
    for(size_t i=0; i!=m; ++i) {
        ARIADNE_ASSERT(p[i]==r[i]);
    }
    return r;
}





// Check that the basic variable array p is consistent with the variable type array vt.
// There are two cases; p just lists the basic variables, or p lists all variables
// Returns the number of basic variables
size_t
consistency_check(const array<Slackness>& vt, const array<size_t>& p)
{
    if(p.size()!=vt.size()) {
        const size_t m=p.size();
        const size_t n=vt.size();
        ARIADNE_ASSERT(m<n);
        for(size_t i=0; i!=m; ++i) {
            ARIADNE_ASSERT_MSG(p[i]<n && vt[p[i]]==BASIS, "vt="<<vt<<" p="<<p);
        }
        return m;
    } else {
        const size_t n=vt.size();
        size_t m=n;
        for(size_t i=0; i!=m; ++i) {
            ARIADNE_ASSERT_MSG(p[i]<n, "vt="<<vt<<" p="<<p);
            if(vt[p[i]]!=BASIS) { m=n; break; }
        }
        for(size_t i=n; i!=n; ++i) {
            ARIADNE_ASSERT_MSG(p[i]<n && vt[p[i]]==BASIS, "vt="<<vt<<" p="<<p);
        }
        return m;
    }
}


template<class X>
size_t
SimplexSolver<X>::consistency_check(const array<Slackness>& vt, const array<size_t>& p) const
{
    return Ariadne::consistency_check(vt,p);
}

// Check that the matrix B is the inverse of the matrix A_B with columns p[0],...,p[m-1] of A.
template<class X>
void
SimplexSolver<X>::consistency_check(const Matrix<X>& A, const array<size_t>& p, const Matrix<X>& B) const
{
    static const X MAXIMUM_ERROR=1e-8;
    const size_t m=A.row_size();
    Matrix<X> A_B(m,m);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        for(size_t i=0; i!=m; ++i) {
            A_B[i][k]=A[i][j];
        }
    }

    array<size_t> p_B(p.begin(),p.begin()+m);

    Matrix<X> Z=prod(B,A_B);
    ARIADNE_LOG(9,"        p_B="<<p_B<<" B="<<B<<" A_B="<<A_B<<" B*A_B-I="<<Z<<"\n");
    for(size_t i=0; i!=m; ++i) { Z[i][i]-=1; }
    ARIADNE_ASSERT_MSG(norm(Z)<MAXIMUM_ERROR, "A="<<A<<"\np="<<p<<"\nB="<<B<<"\nZ=B*A_B-I="<<Z<<"\nnorm(Z)="<<norm(Z));
}


// Check that Ax=b.
template<class X>
void
SimplexSolver<X>::consistency_check(const Matrix<X>& A, const Vector<X>& b,const Vector<X>& x) const
{
    static const X MAXIMUM_ERROR=1e-8;
    Vector<X> z=prod(A,b)-x;
    ARIADNE_ASSERT(norm(z)<MAXIMUM_ERROR);
}


// Check that the matrix B is the inverse of the matrix A_B with columns p[0],...,p[m-1] of A, and that
// the vector x is given by x_L=l_L, x_U=x_U and x_B=B^{-1} A_N x_N.
template<class X>
void
SimplexSolver<X>::consistency_check(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                                    const array<Slackness>& vt, const array<size_t>& p, const Matrix<X>& B, const Vector<X>& x) const
{
    ARIADNE_LOG(9,"        Checking consistency of B and x\n");
    const size_t m=A.row_size();
    const size_t n=A.column_size();

    Matrix<X> A_B(m,m);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        for(size_t i=0; i!=m; ++i) {
            A_B[i][k]=A[i][j];
        }
    }

    array<size_t> p_B(p.begin(),p.begin()+m);

    Matrix<X> I=prod(B,A_B);
    ARIADNE_LOG(9,"          p_B="<<p_B<<" B="<<B<<" A_B="<<A_B<<" B*A_B="<<I<<"\n");
    Matrix<X> Z=I;
    for(size_t i=0; i!=m; ++i) { Z[i][i]-=1; }
    ARIADNE_ASSERT_MSG(norm(Z)<=1e-5,"vt="<<vt<<" p_B="<<p_B<<" B="<<B<<" A_B="<<A_B<<" B*A_B="<<I<<"\n");

    Vector<X> Ax=prod(A,x);
    ARIADNE_LOG(9,"          A="<<A<<" x="<<x<<" b="<<b<<" Ax="<<Ax<<"\n");

    for(size_t k=m; k!=n; ++k) {
        size_t j=p[k];
        ARIADNE_ASSERT_MSG(vt[j]==LOWER || vt[j]==UPPER,
                           "vt["<<j<<"]="<<vt[j]<<"\n  A="<<A<<", b="<<b<<", xl="<<xl<<", xu="<<xu<<", vt="<<vt<<", p="<<p<<", x="<<x<<", Ax="<<Ax);
        X xj = (vt[j]==LOWER ? xl[j] : xu[j]);
        ARIADNE_ASSERT_MSG(x[j]==xj,"x["<<j<<"]="<<x[j]<<" xj="<<xj<<"\n  A="<<A<<", b="<<b<<", xl="<<xl<<", xu="<<xu<<", vt="<<vt<<", p="<<p<<", x="<<x<<", Ax="<<Ax);
    }
    Vector<X> Axmb = Ax-b;
    ARIADNE_ASSERT_MSG(norm(Axmb)<1e-5,"A="<<A<<", b="<<b<<", xl="<<xl<<", xu="<<xu<<", vt="<<vt<<", p="<<p<<", x="<<x<<", Ax="<<Ax);
}




// Compute the cost function for a feasibility step given lower and upper bounds and the values of x.
template<class X>
Vector<X>
compute_c(const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, const Vector<X>& x, size_t m) {
    const size_t n=x.size();
    Vector<X> c(n);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if(possibly(x[j]<=xl[j])) { c[j]=-1; }
        if(possibly(x[j]>=xu[j])) { c[j]=+1; }
    }
    return c;
}

// Compute the variable types from the permutation, taking m basic variables and all non-basic variables lower.
template<class X>
array<Slackness>
compute_vt(const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, const size_t m)
{
    const size_t n=p.size();
    array<Slackness> vt(n);
    for(size_t k=0; k!=m; ++k) {
        vt[p[k]]=BASIS;
    }
    for(size_t k=m; k!=n; ++k) {
        if(xl[p[k]]==-inf<X>()) {
            vt[p[k]] = UPPER;
        } else {
            vt[p[k]] = LOWER;
        }
    }
    return vt;
}


array<size_t>
compute_p(const array<Slackness>& tv)
{
    const size_t n=tv.size();
    array<size_t> p(n);
    size_t k=0;
    for(size_t j=0; j!=n; ++j) {
        if(tv[j]==BASIS) { p[k]=j; ++k; }
    }
    for(size_t j=0; j!=n; ++j) {
        if(tv[j]!=BASIS) { p[k]=j; ++k; }
    }
    return p;
}


// Compute a basis (p_1,\ldots,p_m) for the matrix A
// Throws an error if the matrix A has full row rank
template<class X>
pair< array<size_t>, Matrix<X> >
SimplexSolver<X>::compute_basis(const Matrix<X>& A) const
{
    ARIADNE_LOG(9,"compute_basis(A) with A="<<A<<"\n");
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    ARIADNE_DEBUG_ASSERT(n>=m);

    array<size_t> p(n);
    for(size_t j=0; j!=n; ++j) { p[j]=j; }

    // Factorise into lower and upper triangular matrices L and U
    Matrix<X> L=Matrix<X>::identity(m);
    Matrix<X> U=A;

    for(size_t k=0; k!=m; ++k) {
        // Find a good pivot column j and swap entries of jth and kth columns

        // Look for a column which is the unit vector ek below k
        bool found_unit=false;
        size_t j;
        for(j=k; j!=n; ++j) {
            if(U[k][j]==+1) {
                found_unit=true;
                for(uint i=k+1; i!=m; ++i) {
                    if(U[i][j]!=0) { found_unit=false; break; }
                }
            }
            if(found_unit) { break; }
        }

        // Look for a column with largest U[k][j]
        if(!found_unit) {
            X Ukjmax = abs(U[k][k]);
            size_t jmax=k;
            for(j=k+1; j!=n; ++j) {
                X absUkj=abs(U[k][j]);
                if(absUkj>Ukjmax) {
                    Ukjmax=absUkj;
                    jmax=j;
                }
            }
            j=jmax;
        }

        if (abs(U[k][j]) < SINGULARITY_THRESHOLD) { ARIADNE_THROW(SingularLinearProgram,"compute_basis"," matrix A="<<A<<" is singular or very nearly singular"); }

        if(j!=k) {
            std::swap(p[k],p[j]);
            for(size_t i=0; i!=m; ++i) {
                std::swap(U[i][k],U[i][j]);
            }
        }

        ARIADNE_DEBUG_ASSERT(U[k][k]!=0);

        if(!found_unit) {
            // Update LU factorisation
            X r  = 1/U[k][k];
            for(size_t i=k+1; i!=m; ++i) {
                X s=U[i][k] * r;
                for(size_t j=0; j!=m; ++j) {
                    L[i][j] -= s * L[k][j];
                }
                for(size_t j=k+1; j!=n; ++j) {
                    U[i][j] -= s * U[k][j];
                }
                U[i][k] = 0;
            }
            for(size_t j=0; j!=m; ++j) {
                L[k][j] *= r;
            }
            for(size_t j=k+1; j!=n; ++j) {
                U[k][j] *= r;
            }
            U[k][k] = 1;
        }

    } // end loop on diagonal k

    // Backsubstitute to find inverse of pivot columns

    for(size_t k=m; k!=0; ) {
        --k;
        for(size_t i=0; i!=k; ++i) {
            X s=U[i][k];
            for(size_t j=0; j!=m; ++j) {
                L[i][j] -= s * L[k][j];
            }
            U[i][k] = 0;
        }
    }

    return make_pair(p,L);

}


template<class XX, class X>
Matrix<XX>
compute_B(const Matrix<X>& A, const array<size_t>& p)
{
    const size_t m=A.row_size();
    Matrix<XX> A_B(m,m);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        for(size_t i=0; i!=m; ++i) {
            A_B[i][k]=A[i][j];
        }
    }

    Matrix<XX> B=inverse(A_B);

    return B;
}

template<class X>
Vector<X>
compute_c(const Matrix<X>& A, const array<size_t>& p, const Vector<X>& x)
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    Vector<X> c(n);
    for(size_t k=m; k!=n; ++k) {
        if(x[p[k]]<0) { c[p[k]]=-1; }
    }
    return c;
}


template<class X, class XX> Vector<X>
compute_c(const size_t m, const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, const Vector<XX>& x)
{
    const size_t n=x.size();
    Vector<X> c(n);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if((x[j]<=xl[j])) { c[j]=-1; }
        else if((x[j]>=xu[j])) { c[j]=+1; }
    }
    return c;
}

template<> Vector<Float>
compute_c(const size_t m, const Vector<Float>& xl, const Vector<Float>& xu, const array<size_t>& p, const Vector<Interval>& x)
{
    const size_t n=x.size();
    Vector<Float> c(n);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if(possibly(x[j]<=xl[j])) { c[j]=-1; }
        else if(possibly(x[j]>=xu[j])) { c[j]=+1; }
        if(possibly(x[j]<xl[j]) && possibly(x[j]>xu[j])) {
            ARIADNE_FAIL_MSG("Unhandled case in checking feasibility of linear program. Basic variable x["<<j<<"]="<<x[j]<<" may violate both lower bound "<<xl[j]<<" and upper bound "<<xu[j]<<".");
        }
    }
    return c;
}




template<class X, class XX>
Vector<XX>
compute_x(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
          const array<Slackness>& vt, const array<size_t>& p, const Matrix<XX>& B)
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    ARIADNE_ASSERT_MSG(p.size()==n, "vt="<<vt<<", p="<<p);

    Vector<XX> w(m);
    Vector<XX> x(n);

    // Compute x_N
    for(size_t j=0; j!=n; ++j) {
        if(vt[j]==LOWER) { x[j]=xl[j]; }
        else if(vt[j]==UPPER) { x[j]=xu[j]; }
        else { x[j]=0; }
    }

    // Compute w=b-A_N x_N
    for(size_t i=0; i!=m; ++i) {
        w[i]=b[i];
        for(size_t k=m; k!=n; ++k) {
            size_t j=p[k];
            w[i]-=A[i][j]*x[j];
        }
    }

    // Compute x_B=B w
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        x[j]=0;
        for(size_t i=0; i!=m; ++i) {
            x[j]+=B[k][i]*w[i];
        }
    }

    Vector<XX> Ax=prod(A,x);
    Vector<XX> Axmb=Ax-b;
    ARIADNE_ASSERT_MSG(norm(Axmb)<0.00001,"A="<<A<<", b="<<b<<", xl="<<xl<<", xu="<<xu<<", vt="<<vt<<", p="<<p<<", x="<<x<<", Ax="<<Ax);
    return x;
}


template<class X,class XX>
std::pair<Vector<XX>,Vector<XX> >
compute_wx(const Matrix<X>& A, const Vector<X>& b, const Vector<X>& xl, const Vector<X>& xu, array<Slackness>& vt, const array<size_t>& p, const Matrix<XX>& B)
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();

    Vector<XX> w(m);
    Vector<XX> x(n);

    // Compute x_N
    for(size_t j=0; j!=n; ++j) {
        if(vt[j]==LOWER) { x[j]=xl[j]; }
        else if(vt[j]==UPPER) { x[j]=xu[j]; }
        else { x[j]=0; }
    }
    ARIADNE_LOG(9,"  x_N="<<x);

    // Compute w=b-A_N x_N
    for(size_t i=0; i!=m; ++i) {
        w[i]=b[i];
        for(size_t k=m; k!=n; ++k) {
            size_t j=p[k];
            w[i]-=A[i][j]*x[j];
        }
    }
    ARIADNE_LOG(9," w="<<w);

    // Compute x_B=B w
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        x[j]=0;
        for(size_t i=0; i!=m; ++i) {
            x[j]+=B[k][i]*w[i];
        }
    }

    ARIADNE_LOG(9," x="<<x<<"\n");

    Vector<X> Axmb=prod(A,x)-b;
    ARIADNE_ASSERT(norm(Axmb)<0.00001);
    return make_pair(w,x);
}


template<class X,class XX>
Vector<XX>
compute_y(const Vector<X>& c, const array<size_t>& p, const Matrix<XX>& B)
{
    const size_t m=B.row_size();
    Vector<XX> y(m);
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        for(size_t i=0; i!=m; ++i) {
            y[i]+=c[j]*B[k][i];
        }
    }
    return y;
}

template<class X,class XX,class XXX>
Vector<XX>
compute_z(const Matrix<X>& A, const Vector<XXX>& c, const array<size_t>& p, const Vector<XX>& y)
{
    const double CUTOFF_THRESHOLD=1e-10;
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    Vector<XX> z(n);

    // Shortcut if we can assume c_B - y A_B = 0
    for(size_t k=0; k!=m; ++k) {
        z[p[k]]=0;
    }
    for(size_t k=0; k!=n; ++k) { // Change to k=m,...,n if c_B - y A_B = 0
        size_t j=p[k];
        z[j]=c[j];
        for(size_t i=0; i!=m; ++i) {
            z[j]-=y[i]*A[i][j];
        }
        if(abs(z[j])<CUTOFF_THRESHOLD) {
            //z[j]=0;
        }
    }
    return z;
}

template<class X>
size_t
compute_s(const size_t m, const array<size_t>& p, const Vector<X>& z)
{
    const size_t n=z.size();
    for(size_t k=0; k!=n; ++k) {
        if(z[p[k]]< -PROGRESS_THRESHOLD) { return k; }
    }
    return n;
}



template<class X>
size_t
compute_s(const size_t m, const array<Slackness>& vt, const array<size_t>& p, const Vector<X>& z)
{
    return compute_s_nocycling(m,vt,p,z);
}

// Compute the variable to enter the basis by finding the first which can increase the value function
template<class X>
size_t
compute_s_fast(const size_t m, const array<Slackness>& vt, const array<size_t>& p, const Vector<X>& z)
{
    const size_t n=z.size();
    for(size_t k=m; k!=n; ++k) {
        if( (vt[p[k]]==LOWER && z[p[k]]< -PROGRESS_THRESHOLD)
            || (vt[p[k]]==UPPER && z[p[k]]> +PROGRESS_THRESHOLD) ) { return k; }
    }
    return n;
}

// Compute the variable to enter the basis by giving the one with the highest rate of increase.
template<class X>
size_t
compute_s_best(const size_t m, const array<Slackness>& vt, const array<size_t>& p, const Vector<X>& z)
{
    const size_t n=z.size();
    size_t kmax=n;
    X abszmax=0;
    for(size_t k=m; k!=n; ++k) {
        size_t j=p[k];
        X posz=(vt[j]==LOWER ? -z[j] : z[j]);
        if(posz>abszmax) {
            kmax=k;
            abszmax=posz;
        }
    }
    return kmax;
}

// Compute the variable to enter the basis by using Bland's rule to avoid cycling.
template<class X>
size_t
compute_s_nocycling(const size_t m, const array<Slackness>& vt, const array<size_t>& p, const Vector<X>& z)
{
    const size_t n=z.size();
    for(size_t j=0; j!=n; ++j) {
        if( (vt[j]==LOWER && z[j]< -PROGRESS_THRESHOLD)
                || (vt[j]==UPPER && z[j]> +PROGRESS_THRESHOLD) ) {
            for(size_t k=m; k!=n; ++k) {
                if(p[k]==j) { return k; }
            }
            ARIADNE_ASSERT(false); // Should not reach here
        }
    }
    return n;
}

// Compute the direction in which the basic variables move if non-basic variable is increased.
// given by d=-B*A_s
template<class X, class XX>
Vector<XX>
compute_d(const Matrix<X>& A, const array<size_t>& p, const Matrix<XX>& B, const size_t ks)
{
    const size_t m=A.row_size();
    size_t js=p[ks];
    Vector<XX> d(m);
    for(size_t k=0; k!=m; ++k) {
        for(size_t i=0; i!=m; ++i) {
            d[k]-=B[k][i]*A[i][js];
        }
    }
    return d;
}

template<> Interval inf<Interval>() { return Interval(inf<Float>()); }

template<class X>
pair<size_t,X>
compute_rt(const array<size_t>& p, const Vector<X>& x, const Vector<X>& d)
{
    const size_t m=d.size();
    X t=inf<X>();
    size_t r=m;
    for(size_t k=0; k!=m; ++k) {
        if(d[k] < -CUTOFF_THRESHOLD && x[p[k]] >= CUTOFF_THRESHOLD) {
            X tk=(x[p[k]])/(-d[k]);
            if(r==m || tk<t) {
                t=tk;
                r=k;
            }
        }
    }
    return make_pair(r,t);
}


template<class X>
std::pair<size_t,X>
compute_rt(const Vector<X>& xl, const Vector<X>& xu, const array<Slackness>& vt, const array<size_t>& p, const Vector<X>& x, const Vector<X>& d, const size_t s)
{
    const X inf=Ariadne::inf<X>();

    // Choose variable to take out of basis
    // If the problem is degenerate, choose the variable with smallest index
    const size_t m=d.size();
    const size_t n=x.size();
    size_t r=n;
    X ds=(vt[p[s]]==LOWER ? +1 : -1);
    X t=xu[p[s]]-xl[p[s]];
    if(t<inf) { r=s; }
    X tk=0.0;
    ARIADNE_LOG(7,"   xl="<<xl<<" x="<<x<<" xu="<<xu<<"\n");
    ARIADNE_LOG(7,"   vt="<<vt<<" p="<<p<<" d="<<d<<"\n");
    ARIADNE_LOG(7,"   s="<<s<<" p[s]="<<p[s]<<" vt[p[s]]="<<vt[p[s]]<<" ds="<<ds<<" xl[p[s]]="<<xl[p[s]]<<" xu[p[s]]="<<xu[p[s]]<<" r="<<r<<" t[r]="<<t<<"\n");
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if( d[k]*ds<-CUTOFF_THRESHOLD && x[j]>=xl[j] && xl[j] != -inf) {
            tk=(xl[j]-x[j])/(ds*d[k]);
            //if( r==n || tk<t || (tk==t && p[k]<p[r]) ) { t=tk; r=k; }
            if( tk<t || (tk==t && p[k]<p[r]) ) { t=tk; r=k; }
        } else if( d[k]*ds>CUTOFF_THRESHOLD && x[j]<=xu[j] && xu[j] != inf ) {
            tk=(xu[j]-x[j])/(ds*d[k]);
            //if( r==n || tk<t || (tk==t && p[k]<p[r])) { t=tk; r=k; }
            if( tk<t || (tk==t && p[k]<p[r])) { t=tk; r=k; }
        } else {
            tk=inf;
        }
        ARIADNE_LOG(7,"    k="<<k<<" j=p[k]="<<j<<" xl[j]="<<xl[j]<<" x[j]="<<x[j]<<" xu[j]="<<xu[j]<<" d[k]="<<d[k]<<" t[k]="<<tk<<" r="<<r<<" t[r]="<<t<<"\n");
    }
    t*=ds;

    if(r==n) {
        // Problem is either highly degenerate or optimal do nothing.
        ARIADNE_WARN("SimplexSolver<X>::compute_rt(...): "<<
                     "Cannot find compute variable to exit basis\n"<<
                     "  xl="<<xl<<" x="<<x<<" xu="<<xu<<" vt="<<vt<<" p="<<p<<" d="<<d<<"\n");
    }
    return make_pair(r,t);
}

std::pair<size_t,Interval>
compute_rt(const Vector<Float>& xl, const Vector<Float>& xu, const array<Slackness>& vt, const array<size_t>& p, const Vector<Interval>& x, const Vector<Interval>& d, const size_t s)
{
    typedef Float X;
    typedef Interval XX;
    const X inf=Ariadne::inf<X>();

    // Choose variable to take out of basis
    // If the problem is degenerate, choose the variable with smallest index
    const size_t m=d.size();
    const size_t n=x.size();
    size_t r=n;
    X ds=(vt[p[s]]==LOWER ? +1 : -1);
    XX t=XX(xu[p[s]])-XX(xl[p[s]]);
    if(definitely(t<inf)) { r=s; }
    XX tk=0.0;
    ARIADNE_LOG(7,"   xl="<<xl<<" x="<<x<<" xu="<<xu<<"\n");
    ARIADNE_LOG(7,"   vt="<<vt<<" p="<<p<<" d="<<d<<"\n");
    ARIADNE_LOG(7,"   s="<<s<<" p[s]="<<p[s]<<" vt[p[s]]="<<vt[p[s]]<<" ds="<<ds<<" xl[p[s]]="<<xl[p[s]]<<" xu[p[s]]="<<xu[p[s]]<<" r="<<r<<" t[r]="<<t<<"\n");
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if( definitely(d[k]*ds<0.0) && definitely(x[j]>=xl[j]) && xl[j] != -inf) {
            tk=(xl[j]-x[j])/(ds*d[k]);
            //if( r==n || tk<t || (tk==t && p[k]<p[r]) ) { t=tk; r=k; }
            if( tk<t || (tk==t && p[k]<p[r]) ) { t=tk; r=k; }
        } else if( definitely(d[k]*ds>0.0) && definitely(x[j]<=xu[j]) && xu[j] != inf ) {
            tk=(xu[j]-x[j])/(ds*d[k]);
            //if( r==n || tk<t || (tk==t && p[k]<p[r])) { t=tk; r=k; }
            if( tk<t || (tk==t && p[k]<p[r])) { t=tk; r=k; }
        } else {
            tk=inf;
        }
        ARIADNE_LOG(7,"    k="<<k<<" j=p[k]="<<j<<" xl[j]="<<xl[j]<<" x[j]="<<x[j]<<" xu[j]="<<xu[j]<<" d[k]="<<d[k]<<" t[k]="<<tk<<" r="<<r<<" t[r]="<<t<<"\n");
    }
    t*=ds;

    if(r==n) {
        // Problem is either highly degenerate or optimal do nothing.
        ARIADNE_WARN("SimplexSolver<X>::compute_rt(...): "<<
                     "Cannot find compute variable to exit basis\n"<<
                     "  xl="<<xl<<" x="<<x<<" xu="<<xu<<" vt="<<vt<<" p="<<p<<" d="<<d<<"\n");
    }
    return make_pair(r,t);
}


template<class X>
void
update_B(Matrix<X>& B, const Vector<X>& d, const size_t r)
{
    const size_t m=d.size();
    X dr=d[r];
    X drr=1/dr;
    Vector<X> e(m); e[r]=1;
    Vector<X> Br(m); for(uint j=0; j!=m; ++j) { Br[j]=B[r][j]; }
    for(uint i=0; i!=m; ++i) {
        for(uint j=0; j!=m; ++j) {
            B[i][j]-=(d[i]+e[i])*Br[j]*drr;
        }
    }
    return;
}


template<class X>
void
update_x(const array<size_t>& p, Vector<X>& x, const size_t s, const Vector<X>& d, const size_t r, const X& t)
{
    const size_t m=d.size();
    for(size_t i=0; i!=m; ++i) {
        x[p[i]]+=t*d[i];
    }
    x[p[r]] = 0.0;
    x[p[s]] = t;
    return;
}


template<class X>
void
update_x(const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, Vector<X>& x, const size_t s, const Vector<X>& d, const size_t r, const X& t)
{
    // Update x when variable p[s] becomes basic and variable p[r] becomes non-basic
    // The variable p[s] moves by t; the variables p[i] i<m by t*d[i]
    // The variable p[r] becomes an upper or lower variable depending on t*d[r]
    const size_t m=d.size();
    const size_t n=x.size();
    ARIADNE_ASSERT(r<m);
    ARIADNE_ASSERT(s>=m);
    ARIADNE_ASSERT(s<n);
    for(size_t i=0; i!=m; ++i) {
        x[p[i]]+=t*d[i];
    }
    x[p[s]] += t;
    if(t*d[r]<0) { x[p[r]]=xl[p[r]]; }
    else if(t*d[r]>0) { x[p[r]]=xu[p[r]]; }
    return;
}

template<class X>
void
update_x(const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, Vector<X>& x, const size_t s, const Vector<X>& d, const X& t)
{
    // Update basis when a variable changes between lower and upper
    // The constant t determines how much the variable p[s] moves
    ARIADNE_ASSERT_MSG(s>=d.size(),"x="<<x<<" d="<<d<<" s="<<s<<"\n");
    ARIADNE_ASSERT_MSG(s<x.size(),"x="<<x<<" d="<<d<<" s="<<s<<"\n");
    const size_t m=d.size();
    for(size_t i=0; i!=m; ++i) {
        x[p[i]]+=t*d[i];
    }

    if(t>0) { x[p[s]]=xu[p[s]]; }
    else if(t<0) { x[p[s]]=xl[p[s]]; }
}


template<class X>
void
update_y(const Vector<X>& xl, const Vector<X>& xu, const array<size_t>& p, Vector<X>& y, const size_t s, const Vector<X>& d, const X& t)
{
    ARIADNE_NOT_IMPLEMENTED;
}





template<class X>
size_t lpenter(const Matrix<X>& A, const Vector<X>& c, const array<Slackness>& vt, const array<size_t>& p, const Matrix<X>& B)
{
    const size_t m=A.row_size();

    Vector<X> y=compute_y(c,p,B);
    Vector<X> z=compute_z(A,c,p,y);

    size_t s=compute_s(m,vt,p,z);
    ARIADNE_LOG(5,"  vt="<<vt<<" y="<<y<<" z="<<z<<" s="<<s<<" p[s]="<<p[s]<<"\n");
    return s;
}


template<class X>
tribool
SimplexSolver<X>::validated_feasible(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b) const
{
    ARIADNE_LOG(4,"A="<<A<<" b="<<b<<"\n");
    ARIADNE_LOG(4,"xl="<<xl<<" xu="<<xu<<"\n");

    array<size_t> p(A.column_size());
    array<Slackness> vt(A.column_size());
    Matrix<X> B(A.row_size(),A.row_size());
    make_lpair(p,B)=this->compute_basis(A);
    vt=compute_vt(xl,xu,p,A.row_size());

    bool done = false;
    while(!done) {
        done=this->validated_feasibility_step(xl,xu,A,b,vt,p);
    }
    return this->verify_feasibility(xl,xu,A,b,vt);
}

template<class X>
bool
SimplexSolver<X>::validated_feasibility_step(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                                             array<Slackness>& vt, array<size_t>& p) const
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    static const X inf = Ariadne::inf<X>();

    typedef Interval XX;

    ARIADNE_LOG(9,"vt="<<vt<<" p="<<p<<"\n");
    Matrix<XX> B=compute_B<XX>(A,p);
    ARIADNE_LOG(9," B="<<B<<"\n");
    Vector<XX> x=Ariadne::compute_x(xl,xu,A,b,vt,p,B);
    ARIADNE_LOG(9," x="<<x<<"\n");

    tribool feasible=true;

    Vector<X> c(n);
    Vector<X> relaxed_xl(xl);
    Vector<X> relaxed_xu(xu);
    for(uint i=0; i!=m; ++i) {
        size_t j=p[i];
        if(possibly(x[p[i]]<=xl[p[i]])) { c[j]=-1; relaxed_xl[j]=-inf; feasible=indeterminate; }
        if(possibly(x[p[i]]>=xu[p[i]])) { c[j]=+1; relaxed_xu[j]=+inf; feasible=indeterminate; }
    }
    ARIADNE_LOG(9," c="<<c<<"\n");
    if(definitely(feasible)) { return true; }

    const Vector<XX> y=compute_y(c,p,B);
    ARIADNE_LOG(9," y="<<y<<"\n");

    const Vector<XX> z=compute_z(A,c,p,y);
    ARIADNE_LOG(9," z="<<z<<"\n");

    size_t s = n;
    feasible=false;
    for(size_t k=m; k!=n; ++k) {
        size_t j=p[k];
        if(vt[j]==LOWER) { if(possibly(z[j]<=0)) { feasible=indeterminate; if(definitely(z[j]<0)) { s=k; break; } } }
        if(vt[j]==UPPER) { if(possibly(z[j]>=0)) { feasible=indeterminate; if(definitely(z[j]>0)) { s=k; break; } } }
    }
    ARIADNE_LOG(9," s="<<s<<"\n");
    if(definitely(!feasible)) { return true; }
    if(s==n) { ARIADNE_LOG(9," Cannot find variable to exit basis; no improvement can be made\n"); return true; }

    Vector<XX> d=compute_d(A,p,B,s);
    ARIADNE_LOG(9," d="<<d<<"\n");

    // Compute distance t along d in which to move,
    // and the variable p[r] to leave the basis
    // The bounds on t are given by xl <= x + t * d <= xu
    // Note that t is negative if an upper variable enters the basis
    size_t r; XX t;
    make_lpair(r,t)=compute_rt(xl,xu,vt,p,x,d,s);
    if(r==n) {
        ARIADNE_LOG(3,"   Cannot find variable to enter basis; no improvement can be made\n");
        return true;
    }

    ARIADNE_LOG(5,"  s="<<s<<" p[s]="<<p[s]<<" r="<<r<<" p[r]="<<p[r]<<" d="<<d<<" t="<<t<<"\n");

    if(r==s) {
        // Update variable type
        if(vt[p[s]]==LOWER) { vt[p[s]]=UPPER; }
        else { vt[p[s]]=LOWER; }
    } else {
        // Variable p[r] should leave basis, and variable p[s] enter
        ARIADNE_ASSERT(r<m);

        // Update pivots and variable types
        vt[p[s]] = BASIS;
        if(d[r]*t>0) {
            vt[p[r]] = UPPER;
        } else if(d[r]*t<0) {
            vt[p[r]] = LOWER;
        } else {
            size_t pr=p[r];
            ARIADNE_ASSERT(x[pr]==xl[pr] || x[pr]==xu[pr]);
            if(x[pr]==xl[pr]) { vt[pr]=LOWER; } else { vt[pr]=UPPER; }
        }

        std::swap(p[r],p[s]);
    }

    return false;

}



template<class X>
size_t
SimplexSolver<X>::lpstep(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                         array<Slackness>& vt, array<size_t>& p, Matrix<X>& B, Vector<X>& x, size_t s) const
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();

    ARIADNE_ASSERT(s<=n);
    ARIADNE_ASSERT(vt[p[s]]!=BASIS);

    // Compute direction d in which to move the current basic variables
    // as the variable entering the basis changes by +1
    Vector<X> d=compute_d(A,p,B,s);

    // Compute distance t along d in which to move,
    // and the variable p[r] to leave the basis
    // The bounds on t are given by xl <= x + t * d <= xu
    // Note that t is negative if an upper variable enters the basis
    size_t r; X t;
    make_lpair(r,t)=compute_rt(xl,xu,vt,p,x,d,s);
    if(r==n) {
        ARIADNE_LOG(3,"   Cannot find variable to enter basis; no improvement can be made\n");
        return r;
    }

    ARIADNE_LOG(5,"  s="<<s<<" p[s]="<<p[s]<<" r="<<r<<" p[r]="<<p[r]<<" d="<<d<<" t="<<t<<"\n");

    if(r==s) {
        Slackness nvts=(vt[p[s]]==LOWER ? UPPER : LOWER);
        ARIADNE_LOG(5,"   Changing non-basic variable x["<<p[s]<<"]=x[p["<<s<<"]] from type "<<vt[p[s]]<<" to type "<<nvts<<"\n");
    } else {
        ARIADNE_LOG(5,"   Swapping non-basic variable x["<<p[s]<<"]=x[p["<<s<<"]] with basic variable x["<<p[r]<<"]=x[p["<<r<<"]]\n");
    }

    if(r==s) {
        // Constraint is due to bounds on x_s
        // No change in basic variables or inverse basis matrix
        update_x(xl,xu,p,x,s,d,t);

        // Update variable type
        if(vt[p[s]]==LOWER) { vt[p[s]]=UPPER; }
        else { vt[p[s]]=LOWER; }
    } else {
        // Variable p[r] should leave basis, and variable p[s] enter
        ARIADNE_ASSERT(r<m);

        update_B(B,d,r);
        update_x(xl,xu,p,x,s,d,r,t);

        // Update pivots and variable types
        vt[p[s]] = BASIS;
        if(d[r]*t>0) {
            vt[p[r]] = UPPER;
        } else if(d[r]*t<0) {
            vt[p[r]] = LOWER;
        } else {
            size_t pr=p[r];
            ARIADNE_ASSERT(x[pr]==xl[pr] || x[pr]==xu[pr]);
            if(x[pr]==xl[pr]) { vt[pr]=LOWER; } else { vt[pr]=UPPER; }
        }

        std::swap(p[r],p[s]);
    }

    const double ERROR_TOLERANCE = std::numeric_limits<float>::epsilon();

    // Recompute B and x if it appears that there are problems with numerical degeneracy
    bool possible_degeneracy=false;
    for(uint i=0; i!=m; ++i) {
        if(xl[p[i]]>x[p[i]] || x[p[i]]>xu[p[i]]) {
            possible_degeneracy=true;
            break;
        }
    }
    B=compute_B<X>(A,p);
    x=Ariadne::compute_x<X>(xl,xu,A,b, vt,p,B);
    for(uint i=0; i!=m; ++i) {
        if(x[p[i]]<xl[p[i]]) {
            ARIADNE_ASSERT(x[p[i]]>xl[p[i]]-ERROR_TOLERANCE);
            x[p[i]]=xl[p[i]];
        } else if(x[p[i]]>xu[p[i]]) {
            ARIADNE_ASSERT(x[p[i]]<xu[p[i]]+ERROR_TOLERANCE);
            x[p[i]]=xu[p[i]];
        }
    }

    ARIADNE_LOG(7,"      vt="<<vt<<"\n      p="<<p<<"\n");
    ARIADNE_LOG(7,"      B="<<B<<"\n      x="<<x<<"\n");

    this->consistency_check(xl,xu,A,b, vt,p,B,x);

    return r;
}

template<class X>
bool
SimplexSolver<X>::lpstep(const Vector<X>& c, const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                         array<Slackness>& vt, array<size_t>& p, Matrix<X>& B, Vector<X>& x) const
{
    ARIADNE_LOG(9,"  lpstep(A,b,c,xl,xu,vt,p,V,x)\n    A="<<A<<" b="<<b<<" c="<<c<<"\n    p="<<p<<" B="<<B<<"\n    vt="<<vt<<" xl="<<xl<<" x="<<x<<" xu="<<xu<<"\n");

    const size_t n=A.column_size();
    size_t s=lpenter(A,c,vt,p,B);
    if(s==n) { return true; }
    lpstep(xl,xu,A,b,vt,p,B,x,s);
    return false;
}



template<class X>
Vector<X>
SimplexSolver<X>::compute_x(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                            const array<Slackness>& vt) const
{
    array<uint> p=compute_p(vt);
    Matrix<X> B = Ariadne::compute_B<X>(A,p);
    return Ariadne::compute_x(xl,xu,A,b, vt,p,B);
}



template<class X>
tribool
SimplexSolver<X>::_feasible(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                            array<Slackness>& vt, array<size_t>& p, Matrix<X>& B, Vector<X>& x) const
{
    ARIADNE_LOG(5,"\nInitial A="<<A<<" b="<<b<<"; xl="<<xl<<" xu="<<xu<<"\n  vt="<<vt<<"\n");
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    static const X inf = Ariadne::inf<X>();

    Vector<X> cc(n);
    Vector<X> ll(xl);
    Vector<X> uu(xu);

    // It seems that using this threshold does not work...
    static const double ROBUST_FEASIBILITY_THRESHOLD = std::numeric_limits<double>::epsilon() * 0;

    bool infeasible=false;
    for(size_t j=0; j!=n; ++j) {
        // If x[j] is (almost) infeasible by way of being to low, relax constraint x[j]>=xl[j] to x[j]>=-inf.
        if(x[j]<xl[j]) { cc[j]=-1; ll[j]=-inf; infeasible=true; }
        else if(x[j]>xu[j]) { cc[j]=+1; uu[j]=+inf; infeasible=true; }
        else { cc[j]=0; }
    }
    ARIADNE_LOG(9,"    vt="<<vt<<" x="<<x<<" cc="<<cc<<"\n");

    static const int MAX_STEPS=1024;
    int steps=0;
    while(infeasible) {

        bool done=lpstep(cc,ll,uu,A,b, vt,p,B,x);
        ARIADNE_LOG(9,"  Done changing basis\n");
        ARIADNE_LOG(9,"    p="<<p<<" B="<<B<<"\n");
        ARIADNE_LOG(9,"    vt="<<vt<<" x="<<x<<"\n");

        if(done) {
            ARIADNE_LOG(9,"  Cannot put infeasible variables into basis.");
            Vector<X> y=compute_y(cc,p,B);
            Vector<X> yA=prod(y,A);
            X yb=dot(y,b);
            ARIADNE_LOG(5,"\nCertificate of infeasibility:\n y="<<y<<"\n yA="<<yA<<" yb="<<yb<<"\n");
            return false;
        }

        infeasible=false;
        for(size_t j=0; j!=n; ++j) {
            if(vt[j]==LOWER) { ARIADNE_ASSERT(x[j]==xl[j]); }
            if(vt[j]==UPPER) { ARIADNE_ASSERT(x[j]==xu[j]); }
            if(x[j]<xl[j]+ROBUST_FEASIBILITY_THRESHOLD) { cc[j]=-1; ll[j]=-inf; infeasible=true; }
            else if(x[j]>xu[j]-ROBUST_FEASIBILITY_THRESHOLD) { cc[j]=+1; uu[j]=+inf; infeasible=true; }
            else { cc[j]=0; ll[j]=xl[j]; uu[j]=xu[j]; }
        }
        ARIADNE_LOG(9,"\n    vt="<<vt<<" x="<<x<<" cc="<<cc<<"\n");

        ++steps;
        if(steps>=MAX_STEPS) {
            if(verbosity>0) {
                ARIADNE_WARN("WARNING: Maximum number of steps reached in constrained feasibility problem. "
                             <<"A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" cc="<<cc<<" ll="<<ll<<" uu="<<uu<<" vt="<<vt
                             <<" x="<<x<<" y="<<compute_y(cc,p,B)<<" Ay="<<Vector<X>(prod(compute_y(cc,p,B),A))<<"\n");
            }
            throw DegenerateFeasibilityProblemException();
        }
    }

    ARIADNE_LOG(9,"  Checking solution\n");

    // Check solution
    for(size_t i=0; i!=n; ++i) {
        ARIADNE_ASSERT_MSG(x[i]>=xl[i]-BOUNDS_TOLERANCE, "A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<" B="<<B<<" x="<<x );
        ARIADNE_ASSERT_MSG(x[i]<=xu[i]+BOUNDS_TOLERANCE, "A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<" B="<<B<<" x="<<x );
    }
    Vector<X> Ax=prod(A,x);
    for(size_t i=0; i!=m; ++i) {
        ARIADNE_ASSERT(abs(Ax[i]-b[i])<0.0001);
    }

    ARIADNE_LOG(5,"\nFeasible point x="<<x<<"; xl="<<xl<<" xu="<<xu<<"\n Ax="<<Vector<X>(prod(A,x))<<" b="<<b<<"\n");

    return true;
}



// Check for feasibility of Ax=b xl<=b<=xu
template<class X>
tribool
SimplexSolver<X>::feasible(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b) const
{
    ARIADNE_LOG(2,"feasible(xl,xu,A,b)\n");
    ARIADNE_LOG(3,"    A="<<A<<"\n    b="<<b<<"\n    xl="<<xl<<"\n    xu="<<xu<<"\n");
    ARIADNE_ASSERT(b.size()==A.row_size());
    ARIADNE_ASSERT(xl.size()==A.column_size());
    ARIADNE_ASSERT(xu.size()==A.column_size());

    const size_t m=A.row_size();

    array<size_t> p;
    Matrix<X> B;
    make_lpair(p,B)=compute_basis(A);

    array<Slackness> vt=compute_vt(xl,xu,p,m);

    ARIADNE_LOG(9,"    p="<<p<<" B="<<B<<"  (BA="<<Matrix<X>(prod(B,A))<<")\n");

    Vector<X> x=Ariadne::compute_x(xl,xu,A,b,vt,p,B);
    Vector<X> y(m);

    return this->hotstarted_feasible(xl,xu,A,b,vt,p,B,x,y);
}



template<class X>
tribool
SimplexSolver<X>::hotstarted_feasible(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                                      array<Slackness>& vt, array<size_t>& p, Matrix<X>& B, Vector<X>& x, Vector<X>& y) const
{
    ARIADNE_LOG(5,"A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<"\n");
    ARIADNE_LOG(5,"vt="<<vt<<" p="<<p<<"\n");

    const size_t m=A.row_size();
    //const size_t n=A.column_size();
    if(vt.size()==0) {
        make_lpair(p,B)=compute_basis(A);
        vt=compute_vt(xl,xu,p,m);
    }
    if(p.size()==0) {
        p=compute_p(vt);
        B=compute_B<X>(A,p);
    }
    consistency_check(A,p,B);

    x=Ariadne::compute_x(xl,xu,A,b,vt,p,B);
    this->consistency_check(xl,xu,A,b,vt,p,B,x);

    tribool fs = this->_feasible(xl,xu,A,b,vt,p,B,x);

    ARIADNE_LOG(7,"vt="<<vt<<" p="<<p<<" fs="<<fs<<"\n");
    Vector<X> c=compute_c(xl,xu,p,x,m);
    y=compute_y(c,p,B);
    Vector<X> z=compute_z(A,c,p,y);
    ARIADNE_LOG(7,"x="<<x<<" c="<<c<<" y="<<y<<" z="<<z<<"\n");

    tribool vfs = this->verify_feasibility(xl,xu,A,b,vt);
    if(!indeterminate(vfs) && vfs!=fs) {
        if(verbosity>0) {
            ARIADNE_WARN("Approximate feasibility algorithm for\n  A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<"\nyielded basic variables "<<vt<<
                         " and result "<<fs<<", but validation code gave "<<vfs<<".\n");
        }
    }
    //FIXME: Return correct value
    // return vfs;
    return fs;
}

template<class X> struct RigorousNumericsTraits { typedef X Type; };
template<> struct RigorousNumericsTraits<Float> { typedef Interval Type; };

// A point x is strictly feasible for the basis B with lower variables L and upper variables U if
//   x_B = A_B^{-1} (b - A_L x_L - A_U x_U) is definitely within (x_B), the open interval (x_BL,x_BU).
// To prove infeasibility, choose a vector c (typically c_L=c_U=0, (c_B)_i = +1 if x_i>u_i and -1 if x_i<u_i)
// and set dual vector y = c_B A_B^{-1} .
//  y (b - A_L x_L - A_U x_U) > 0
//  z = c - y A  satisfies z_U < 0 and z_L > 0; by construction z_B = 0.
template<class X> tribool
SimplexSolver<X>::verify_feasibility(const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b, const array<Slackness>& vt) const
{
    ARIADNE_LOG(4,"verify_feasibility(Vector xl, Vector xu, Matrix A, Vector b, VariableTypeArray vt)\n");
    ARIADNE_LOG(5,"A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<"\n");
    const array<size_t> p=compute_p(vt);

    typedef Interval XX;
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    ARIADNE_ASSERT(b.size()==m);
    ARIADNE_ASSERT(xl.size()==n);
    ARIADNE_ASSERT(xu.size()==n);
    ARIADNE_ASSERT(vt.size()==n);

    // Ensure singleton constraints for x are non-basic
    for(size_t j=0; j!=n; ++j) {
        if(xl[j]==xu[j]) { ARIADNE_ASSERT(!vt[j]==BASIS);}
    }

    {
        const Matrix<X> B=compute_B<X>(A,p);
        const Vector<X> x=Ariadne::compute_x(xl,xu,A,b,vt,p,B);
        const Vector<X> c=compute_c(m,xl,xu,p,x);
        const Vector<X> y=compute_y(c,p,B);
        const Vector<X> z=compute_z(A,c,p,y);
        ARIADNE_LOG(7,"x="<<x<<" c="<<c<<" y="<<y<<" z="<<z<<"\n");
    }


    const Matrix<XX> B=compute_B<XX>(A,p);
    ARIADNE_LOG(9," B="<<B<<"; B*A="<<midpoint(prod(B,A))<<"\n");

    const Vector<XX> x=Ariadne::compute_x(xl,xu,A,b,vt,p,B);
    ARIADNE_LOG(9," x="<<x<<"; A*x="<<Vector<XX>(prod(A,x))<<"\n");


    const Vector<X> c=compute_c(m,xl,xu,p,x);
    ARIADNE_LOG(9," c="<<c<<"\n");

    const Vector<XX> y=compute_y(c,p,B);
    ARIADNE_LOG(9," y="<<y<<"\n");

    const Vector<XX> z=compute_z(A,c,p,y);
    ARIADNE_LOG(9," z="<<z<<"\n");

    ARIADNE_LOG(5,"x="<<x<<" c="<<c<<" y="<<y<<" z="<<z<<"\n");

    tribool fs=true;
    for(size_t k=0; k!=m; ++k) {
        size_t j=p[k];
        if(possibly(x[j]<=xl[j]) || possibly(x[j]>=xu[j])) {
            ARIADNE_LOG(9," k="<<k<<" j="<<j<<" xl[j]="<<xl[j]<<" x[j]="<<x[j]<<" xu[j]="<<xu[j]<<"\n");
            fs=indeterminate;
            if(definitely(x[j]<xl[j]) || definitely(x[j]>xu[j])) {
                fs=false;
                break;
            }
        }
    }

    if(fs==true) {
        return fs;
    }

    // The basis is optimal for min c x if z_L >=0  and z_U <= 0.
    // We have definite infeasibility if z_L > 0  and z_U < 0
    // If z_j < 0 for j lower, or z_j > 0 for j upper, then the simplex algorithm has not terminated correctly.

    for(size_t k=m; k!=n; ++k) {
        size_t j=p[k];
        if(vt[j]==LOWER && possibly(z[j]<=0)) { fs=indeterminate; break; }
        if(vt[j]==UPPER && possibly(z[j]>=0)) { fs=indeterminate; break; }
    }

    if(fs==false) {
        return fs;
    }

    // TODO: Code below is too enthusiastic about declaring an error.
    // For some degenerate, there may be no strictly feasible basic solution,
    // but it is easy to find a strictly feasible non-basic solution.
    // Should make feasibility testing more robust by testing the possibility of
    // obtaining strict feasibility by a partial step.

    return fs;
}





template<class X>
Vector<X>
SimplexSolver<X>::minimise(const Vector<X>& c, const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b) const
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    ARIADNE_ASSERT(b.size()==m);
    ARIADNE_ASSERT(c.size()==n);
    ARIADNE_ASSERT(xl.size()==n);
    ARIADNE_ASSERT(xu.size()==n);

    array<Slackness> vt(n);
    array<size_t> p(n);
    Matrix<X> B(m,m);
    Vector<X> x(n);

    make_lpair(p,B)=compute_basis(A);
    for(size_t k=0; k!=m; ++k) { vt[p[k]]=BASIS; }
    for(size_t k=m; k!=n; ++k) { vt[p[k]]=LOWER; }

    return hotstarted_minimise(c,xl,xu,A,b,vt,p,B);

}

template<class X>
Vector<X>
SimplexSolver<X>::hotstarted_minimise(const Vector<X>& c, const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                                      array<Slackness>& vt) const
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();
    ARIADNE_ASSERT(b.size()==m);
    ARIADNE_ASSERT(c.size()==n);
    ARIADNE_ASSERT(xl.size()==n);
    ARIADNE_ASSERT(xu.size()==n);
    ARIADNE_ASSERT(vt.size()==n);
    ARIADNE_ASSERT(static_cast<size_t>(std::count(vt.begin(),vt.end(),BASIS))==m);

    array<size_t> p=compute_p(vt);
    Matrix<X> B=compute_B<X>(A,p);

    return hotstarted_minimise(c,xl,xu,A,b,vt,p,B);

}

template<class X>
Vector<X>
SimplexSolver<X>::hotstarted_minimise(const Vector<X>& c, const Vector<X>& xl, const Vector<X>& xu, const Matrix<X>& A, const Vector<X>& b,
                                      array<Slackness>& vt, array<size_t>& p, Matrix<X>& B) const
{
    const size_t m=A.row_size();
    const size_t n=A.column_size();

    if(p.size()==m) { p=extend_p(p,n); }

    ARIADNE_ASSERT(b.size()==m);
    ARIADNE_ASSERT(c.size()==n);
    ARIADNE_ASSERT(xl.size()==n);
    ARIADNE_ASSERT(xu.size()==n);
    ARIADNE_ASSERT(vt.size()==n);
    ARIADNE_ASSERT(p.size()==n);
    ARIADNE_ASSERT(B.row_size()==m);
    ARIADNE_ASSERT(B.column_size()==m);

    consistency_check(vt,p);
    this->consistency_check(A,p,B);

    Vector<X> x(n);
    x=Ariadne::compute_x(xl,xu,A,b, vt,p,B);
    ARIADNE_LOG(3,"Initial A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<" p="<<p<<" x="<<x<<" Ax="<<A*x<<"\n");

    this->_feasible(xl,xu,A,b, vt,p,B,x);
    ARIADNE_LOG(3,"Feasible A="<<A<<" b="<<b<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<" p="<<p<<" x="<<x<<" Ax="<<A*x<<"\n");

    bool done=false;
    const int MAX_STEPS=1024;
    int steps=0;
    while(not done) {
        done=lpstep(c,xl,xu,A,b, vt,p,B,x);
        ++steps;
        ARIADNE_ASSERT_MSG(steps<MAX_STEPS,"Maximum number of steps reached for linear programming problem.");
    }
    ARIADNE_LOG(3,"Optimal A="<<A<<" b="<<b<<" c="<<c<<" xl="<<xl<<" xu="<<xu<<" vt="<<vt<<" p="<<p<<" x="<<x<<" Ax="<<A*x<<" cx="<<dot(x,x)<<"\n");

    return x;
}






template class SimplexSolver<Float>;

#ifdef HAVE_GMPXX_H
inline Interval operator+(const Interval& ivl, const Rational& q) { return ivl+Interval(q); }
inline Interval operator+(const Rational& q, const Interval& ivl) { return Interval(q)+ivl; }
inline Interval operator-(const Interval& ivl, const Rational& q) { return ivl-Interval(q); }
inline Interval operator-(const Rational& q, const Interval& ivl) { return Interval(q)-ivl; }
inline Interval operator*(const Interval& ivl, const Rational& q) { return ivl*Interval(q); }
inline Interval operator*(const Rational& q, const Interval& ivl) { return Interval(q)-ivl; }
template class SimplexSolver<Rational>;
#endif // HAVE_GMPXX_H


} // namespace Ariadne

