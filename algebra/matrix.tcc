/***************************************************************************
 *            matrix.tcc
 *
 *  Copyright 2005-13  Alberto Casagrande, Pieter Collins
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

#include "algebra/vector.h"

namespace Ariadne {

template<class X> Matrix<X>::Matrix(SizeType m, SizeType n)
    : _zero(), _rs(m), _cs(n), _ary(m*n) {
}

template<class X> Matrix<X>::Matrix(SizeType m, SizeType n, const X& x)
    : _zero(0), _rs(m), _cs(n), _ary(m*n,x) {
}

template<class X> Matrix<X> Matrix<X>::zero(SizeType m, SizeType n) {
    Matrix<X> A(m,n,X(0));
    return std::move(A);
}

template<class X> Matrix<X> Matrix<X>::identity(SizeType n) {
    Matrix<X> A(n,n,X(0));
    for(SizeType i=0; i!=n; ++i) { A.at(i,i)=1u; }
    return std::move(A);
}


template<class X> SizeType Matrix<X>::row_size() const {
    return _rs;
}

template<class X> SizeType Matrix<X>::column_size() const {
    return _cs;
}

template<class X> Void Matrix<X>::resize(SizeType m, SizeType n) {
    if(m*n != _rs*_cs) { _ary.resize(m*n); } _rs=m; _cs=n;
}

template<class X> X Matrix<X>::zero_element() const {
    return _zero;
}

template<class X> Matrix<X>::Matrix(InitializerList<InitializerList<X>> lst) : _rs(lst.size()), _cs(lst.begin()->size()), _ary(_rs*_cs) {
    typename std::initializer_list<std::initializer_list<X>>::const_iterator row_iter=lst.begin();
    for(SizeType i=0; i!=this->row_size(); ++i, ++row_iter) {
        ARIADNE_PRECONDITION(row_iter->size()==this->column_size());
        typename std::initializer_list<X>::const_iterator col_iter=row_iter->begin();
        for(SizeType j=0; j!=this->column_size(); ++j, ++col_iter) {
            this->at(i,j)=*col_iter;
        }
    }
}

template<class X> OutputStream& Matrix<X>::write(OutputStream& os) const {
    const Matrix<X>& A=*this;
    if(A.row_size()==0 || A.column_size()==0) { os << "["; }
    for(SizeType i=0; i!=A.row_size(); ++i) {
        for(SizeType j=0; j!=A.column_size(); ++j) {
            os << (j==0 ? (i==0 ? "[" : "; ") : ",") << A.at(i,j); } }
    return os << "]";
}

#ifdef ARIADNE_OMIT
template<class X> InputStream& Matrix<X>::read(InputStream& is) {
    Matrix<X>& A=*this;
    char c;
    is >> c;
    is.putback(c);
    if(c=='[') {
        is >> c;
        /* Representation as a literal [a11,a12,...,a1n; a21,a22,...a2n; ... ; am1,am2,...,amn] */
        std::vector< std::vector<X> > v;
        X x;
        c=';';
        while(is && c==';') {
            v.push_back(std::vector<X>());
            c=',';
            while(is && c==',') {
                is >> x;
                v.back().push_back(x);
                is >> c;
            }
        }
        if(is) {
            A=Matrix<X>(v.size(),v.front().size());
            for(SizeType i=0; i!=A.row_size(); ++i) {
                if(v[i].size()!=A.column_size()) {
                    // TOOD: Add exception
                    assert(false);
                    //ARIADNE_THROW(InvalidInput,"Matrix::read(istream&)","row[0].size()="<<v[0].size()<<", row["<<i<<"].size()="<<v[i].size());
                }
                for(SizeType j=0; j!=A.column_size(); ++j) {
                    A[i][j]=v[i][j];
                }
            }
        }
    }
    else {
        // TOOD: Add exception
        assert(false);
        //ARIADNE_THROW(InvalidInput,"Matrix::read(istream&)"," separator c="<<c);
    }
    return is;
}
#endif

template<class X> decltype(mag(declval<X>())) sup_norm(const Matrix<X>& A)
{
    typedef decltype(mag(declval<X>())+mag(declval<X>())) R;
    R result=0u;
    for(SizeType i=0; i!=A.row_size(); ++i) {
        R row_sum=0u;
        for(SizeType j=0; j!=A.column_size(); ++j) {
            row_sum+=mag(A[i][j]);
        }
        // NOTE: The arguments must be this way round to propagate a nan row_sum
        result=max(row_sum,result);
    }
    return result;
}

template<class X> decltype(declval<X>()+mag(declval<X>())) log_norm(Matrix<X> const& A)
{
    ARIADNE_PRECONDITION(A.row_size()==A.column_size());
    typedef decltype(declval<X>()+mag(declval<X>())) R;
    R r(-inf);
    for(SizeType i=0; i!=A.row_size(); ++i) {
        R t=A[i][i];
        for(SizeType j=0; j!=A.column_size(); ++j) {
            if(j!=i) {
                t+=mag(A[i][j]);
            }
        }
        r=max(r,t);
    }
    return r;
}


} // namespace Ariadne
