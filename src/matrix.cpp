#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mkl.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

//using namespace std;

class Matrix {

public:
    Matrix() = default;
    Matrix(size_t nrow, size_t ncol)
            : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
    }
    Matrix(size_t nrow, size_t ncol, double b, double k, double r, double step, double dx, double dy)
            : m_nrow(nrow), m_ncol(ncol), m_b(b), m_k(k), m_r(r), m_step(step), m_dx(dx), m_dy(dy)
    {
        reset_buffer(nrow, ncol);
        cal_max_dt();
    }
    ~Matrix()
    {
        reset_buffer(0, 0);
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const &vec)
            : m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix(Matrix const & other)
            : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_b(other.m_b),
             m_k(other.m_k), m_r(other.m_r), m_step(other.m_step),
            m_dx(other.m_dx), m_dy(other.m_dy)
    {

        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    Matrix(Matrix && other)
            : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_b(other.m_b),
              m_k(other.m_k), m_r(other.m_r), m_step(other.m_step),
            m_dx(other.m_dx), m_dy(other.m_dy)
    {
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
//        std::swap(m_d, other.m_d);
        std::swap(m_b, other.m_b);
        std::swap(m_k, other.m_k);
        std::swap(m_r, other.m_r);
        std::swap(m_step, other.m_step);
        std::swap(m_dx, other.m_dx);
        std::swap(m_dy, other.m_dy);
        std::swap(m_buffer, other.m_buffer);
    }

    Matrix & operator=(std::vector<double> const & vec)
    {
        if (size() != vec.size())
        {
            throw std::out_of_range("number of elements mismatch");
        }

        size_t k = 0;
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = vec[k];
                ++k;
            }
        }

        return *this;
    }

    Matrix & operator=(Matrix const & other)
    {
        if (this->true_equal(other)) { return *this; }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
//            this->set_d(other.d());
            this->set_b(other.b());
            this->set_k(other.k());
            this->set_r(other.r());
            this->set_step(other.step());
            this->set_dx(other.dx());
            this->set_dy(other.dy());
        }
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
//        this->set_d(other.d());
        this->set_b(other.b());
        this->set_k(other.k());
        this->set_r(other.r());
        this->set_step(other.step());
        this->set_dx(other.dx());
        this->set_dy(other.dy());
        return *this;
    }

    Matrix & operator=(Matrix && other)
    {
        if (this->true_equal(other)) { return *this; }
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
//        std::swap(m_d, other.m_d);
        std::swap(m_b, other.m_b);
        std::swap(m_k, other.m_k);
        std::swap(m_r, other.m_r);
        std::swap(m_step, other.m_step);
        std::swap(m_dx, other.m_dx);
        std::swap(m_dy, other.m_dy);
        std::swap(m_buffer, other.m_buffer);
        return *this;
    }
    // only check the matrix value equal or not
    bool operator==(Matrix &right) {

        if (m_nrow != right.nrow() && m_ncol != right.ncol()) {
            return false;
        }
        const double *right_buf = right.data();
        size_t len = m_nrow * m_ncol;
        for (size_t i = 0; i < len; ++i) {
            if (m_buffer[i] != right_buf[i])
                return false;
        }
        return true;
    }
    // check both value and constants setting
    bool const true_equal(Matrix const &right) {

        if (m_nrow != right.nrow() && m_ncol != right.ncol()
            && m_b != right.b() && m_k != right.k()
           && m_r != right.r() && m_step != right.step()) {
            return false;
        }
        const double *right_buf = right.data();
        size_t len = m_nrow * m_ncol;
        for (size_t i = 0; i < len; ++i) {
            if (m_buffer[i] != right_buf[i])
                return false;
        }
        return true;
    }
    double   operator() (size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[index(row, col)]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
//    double d() const { return m_d; }
    double b() const { return m_b; }
    double k() const { return m_k; }
    double r() const { return m_r; }
    double t() const { return m_t; }
    double step() const { return m_step; }
    double dx() const { return m_dx; }
    double dy() const { return m_dy; }
    double max_dt() const { return m_max_dt; }

//    void set_d(double d){ m_d = d;}
    void set_b(double b){ m_b = b;}
    void set_k(double k){ m_k = k;}
    void set_r(double r){ m_r = r;}
    void set_t(double t){ m_t = t;}
    void set_step(double step){ m_step = step;}
    void set_dx(double dx){ m_dx = dx;}
    void set_dy(double dy){ m_dy = dy;}

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const { return std::vector<double>(m_buffer, m_buffer+size()); }

    double * data() const { return m_buffer; }
    Matrix get_block(Matrix const & mat, size_t tsize, size_t bs_i, size_t bs_j, size_t it, size_t jt, bool col_major=false){
        if(!col_major) {
            for (size_t i = 0; i < bs_i; i++) {
                for (size_t j = 0; j < bs_j; j++) {
                    (*this)(i, j) = mat(it * tsize + i, jt * tsize + j);
                }
            }
        }else{
            for (size_t j=0; j<bs_j; j++){
                for (size_t i=0; i<bs_i; i++){
                    (*this)(j, i) = mat(it * tsize + i, jt * tsize + j);
                }
            }
        }
        return *this;
    }

    Matrix &operator+=(Matrix & other){
        // The first time
        if (m_nrow==0 && m_ncol==0){
            *this = other;
            return *this;
        }
        if (this->nrow() != other.nrow() && this->ncol() != other.ncol()){
            throw std::out_of_range("matrix shape is wrong");
        }
        for (size_t i=0; i<other.nrow(); i++){
            for (size_t j=0; j<other.ncol(); j++){
                (*this)(i,j) += other(i,j);
            }
        }
        return *this;
    }
    Matrix &operator+=(Matrix && other){
        // The first time
        if (m_nrow==0 && m_ncol==0){
            *this = other;
            return *this;
        }
        if (this->nrow() != other.nrow() && this->ncol() != other.ncol()){
            throw std::out_of_range("matrix shape is wrong");
        }
        for (size_t i=0; i<other.nrow(); i++){
            for (size_t j=0; j<other.ncol(); j++){
                (*this)(i,j) += other(i,j);
            }
        }
        return *this;
    }




    // helper function
    double cal_slop(size_t x, size_t y){
        return m_b*(laplacian_2(x, y) + 2*m_k*laplacian(x, y) + pow(m_k, 2)*edge_protect(x, y))
                - m_r*edge_protect(x, y) + pow(edge_protect(x, y), 3); // Defect generate function
//        return m_d * laplacian(x, y); // spread function
    }
    double laplacian(size_t x, size_t y){
        return ((edge_protect(x+1, y) + edge_protect(x-1, y) - 2*edge_protect(x, y))
                + (edge_protect(x, y+1) + edge_protect(x, y-1) - 2*edge_protect(x, y)));
    }
    double laplacian_2(size_t x, size_t y){
        // d/dx^4 + 2*d/dx^2dy^2 + d/dy^4
        return (edge_protect(x+4,y)-2*edge_protect(x+3,y)+2*edge_protect(x+1,y)-2*edge_protect(x,y)+2*edge_protect(x-1,y)-2*edge_protect(x-3,y)+edge_protect(x-4,y))/dx()/dx()/dx()/dx()
        + 2*(edge_protect(x+1,y+1)+edge_protect(x+1,y-1)-2*edge_protect(x+1,y) + edge_protect(x-1,y+1)+edge_protect(x-1,y-1)-2*edge_protect(x-1,y) - 2*edge_protect(x,y+1)-2*edge_protect(x,y-1)+4*edge_protect(x,y))/dx()/dx()/dy()/dy()
        + (edge_protect(x,y+4)-2*edge_protect(x,y+3)+2*edge_protect(x,y+1)-2*edge_protect(x,y)+2*edge_protect(x,y-1)-2*edge_protect(x,y-3)+edge_protect(x,y-4))/dy()/dy()/dy()/dy();
    }
    double edge_protect(size_t x, size_t y){
        size_t fixed_x = x;
        size_t fixed_y = y;
        if (x < 0)
            fixed_x = 0;
        if (x >= nrow())
            fixed_x = nrow()-1;
        if (y < 0)
            fixed_y =0;
        if (y >= ncol())
            fixed_y = ncol()-1;
        return (*this)(fixed_x, fixed_y);
    }
    void save_block(Matrix &mat, size_t m, size_t n){
        for (size_t i=m; i<m+mat.nrow(); i++){
            for (size_t j=n; j<n+mat.ncol(); j++){
                (*this)(i,j) = mat(i-m, j-n);
            }
        }
    }
    void set_all_0(){
        if( nrow()!=0 && ncol()!=0) {
            for (size_t i = 0; i < nrow(); i++) {
                for (size_t j = 0; j < ncol(); j++) {
                    (*this)(i, j) = 0;
                }
            }
        }
    }
    // TODO: resize the matrix to generate the defect
    // for now, this function only serve as memory reallocation
    void resize(size_t row, size_t col){
//        if(nrow()!=row || ncol()!=col){
//            if (((std::max(row, nrow()) % std::min(row,nrow())) != 0) || (std::max(col,ncol()) % std::min(col,ncol())) != 0)
//                throw "new size is not the integral multiple of old size!";

//            double ratio_x = (double)row / (double)nrow();
//            double ratio_y = (double)col / (double)ncol();
            //TO-DO: assign value to new (x, y)
            m_nrow = row;
            m_ncol = col;
            reset_buffer(nrow(), ncol());
//        }
    }
    double cal_max_dt(){
        // find the max allowed dt, x0.3 for safety
        m_max_dt = 0.3 * m_dx * m_dx;
        return m_max_dt;
    }
    // nd_array
    py::array_t<double> nd_array() const {
        auto capsule = py::capsule(m_buffer, [](void *){});
        return py::array_t<double>({m_nrow, m_ncol}, m_buffer, capsule);
    }
private:

    size_t index(size_t row, size_t col) const
    {
        return row * m_ncol + col;
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement]; }
        else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double m_b = 0;
    double m_k = 0;
    double m_r = 0;
    double m_step = 0;  // changing speed
//    double m_d = 0; // spread constant: must < 0.5
    double m_max_dt = 0;  // system allowed minimum dt
    double m_dx = 0;      // used in calculate laplacian
    double m_dy = 0;
    double m_t = 0; // global time of iteration
    double * m_buffer = nullptr;

};

std::ostream & operator << (std::ostream & ostr, Matrix const & mat)
{
    for (size_t i=0; i<mat.nrow(); ++i)
    {
        ostr << std::endl << " ";
        for (size_t j=0; j<mat.ncol(); ++j)
        {
            ostr << " " << std::setw(2) << mat(i, j);
        }
    }

    ostr << std::endl << " data: ";
    for (size_t i=0; i<mat.size(); ++i)
    {
        ostr << " " << std::setw(2) << mat.data()[i];
    }

    return ostr;
}
std::ostream & operator << (std::ostream & ostr, std::vector<double> const & vec)
{
    for (size_t i=0; i<vec.size(); ++i)
    {
        std::cout << " " << vec[i];
    }

    return ostr;
}

/*
 * Naive matrix matrix multiplication.
 */
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
                "the number of first matrix column "
                "differs from that of second matrix row");
    }

    Matrix ret(mat1.nrow(), mat2.ncol());

    for (size_t i=0; i<ret.nrow(); ++i)
    {
        for (size_t k=0; k<ret.ncol(); ++k)
        {
            double v = 0.0;
            for (size_t j=0; j<mat1.ncol(); ++j)
            {
                v += mat1(i,j) * mat2(j,k);
            }
            ret(i,k) = v;
        }
        //printf("[%zu,%zu]: %f", i, 13, mat1(i,13));

    }

    return ret;
}

Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
                "the number of first matrix column "
                "differs from that of second matrix row");
    }

    Matrix ret(mat1.nrow(), mat2.ncol());

    size_t m = mat1.nrow();
    size_t n = mat2.ncol();
    size_t k = mat1.ncol();
    double alpha = 1.0;
    double beta = 0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, mat1.data(), k, mat2.data(), n, beta, ret.data(), n);

    return ret;
}


Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t tsize){
    if (mat1.ncol() != mat2.nrow()){
        throw std::out_of_range(
                "the number of first matrix column "
                "differs from that of second matrix row");
    }
    size_t min_size = std::min({mat1.nrow(), mat1.ncol(), mat2.ncol()});
    if (tsize > min_size)
        throw std::invalid_argument("tile size too large!");
    if (tsize == 0)
        return multiply_naive(mat1, mat2);
    // total size of each dimension
    size_t sum_m = mat1.nrow();
    size_t sum_k = mat1.ncol();
    size_t sum_n = mat2.ncol();
    // total tile numbers of each dimension
    size_t tile_m = (sum_m + tsize - 1) / tsize;
    size_t tile_k = (sum_k + tsize - 1) / tsize;
    size_t tile_n = (sum_n + tsize - 1) / tsize;

    Matrix empty_block;
    Matrix block_A(tsize, tsize);
    Matrix block_B(tsize, tsize);
    Matrix block_C(tsize, tsize);
    Matrix result(sum_m, sum_n);
    result.set_all_0();
    for (size_t mt=0; mt<tile_m; mt++){
        // block size = tsize or remain(at last tile)
        size_t bs_m = std::min(tsize, sum_m - mt*tsize);
        for (size_t nt=0; nt<tile_n; nt++){
            size_t bs_n = std::min(tsize, sum_n - nt*tsize);
            // create empty block C to contain block result
            block_C.resize(bs_m, bs_n);
            block_C.set_all_0();
            for (size_t kt=0; kt<tile_k; kt++) {
                size_t bs_k = std::min(tsize, sum_k - kt*tsize);
                block_A.resize(bs_m, bs_k);
                block_B.resize(bs_n, bs_k);

                block_A.get_block(mat1, tsize, bs_m, bs_k, mt, kt);
//                cout << "bs_m: " << bs_m << " bs_k: " << bs_k << " mt: " << mt << " kt: " << kt << endl;
//                std::cout << "block_A: " << block_A << std::endl;
                block_B.get_block(mat2, tsize, bs_k, bs_n, kt, nt, true);
//                std::cout << "block_B: " << block_B << std::endl;
                // calculate block_C with cache friendly way
//                block_C += multiply_naive(block_A, block_B);
                for (size_t m=0; m<bs_m; m++){
                    for (size_t n=0; n<bs_n; n++){
                        double v = 0;
                        for (size_t k=0; k<bs_k; k++)
                            v += block_A(m,k) * block_B(n,k);
                        block_C(m,n) += v;
                    }
                }
//                std::cout << "block_C: " << block_C << std::endl;
            }
//            std::cout << "---------------------save block_C--------------------" << std::endl;
            result.save_block(block_C, mt*tsize, nt*tsize);
        }
    }
    return result;
}

// iterate simulation

Matrix iterate(Matrix &mat, double dt){
    mat.set_t(mat.t() + dt);
    Matrix ret = mat;
    mat.cal_max_dt();
    bool forward = (dt>=0) ? true : false;
    double slop;
    int num_dt = abs(dt/mat.max_dt()); // How many max_dt are contain in dt
//    double dt_remain = dt;
    double dt_remain = dt - num_dt*mat.max_dt();
    for (int k=0; k<num_dt;k++) { // each step is max_dt, run num_dt steps
        for (size_t i = 0; i < mat.nrow(); i++) {
            for (size_t j = 0; j < mat.ncol(); j++) {
                slop = mat.cal_slop(i, j);
                if (forward)
                    ret(i, j) -= mat.max_dt() * mat.step() * slop;
                else
                    ret(i, j) += mat.max_dt() * mat.step() * slop;
            }
        }
    }
    // deal with the remain dt
    for (size_t i = 0; i < mat.nrow(); i++) {
        for (size_t j = 0; j < mat.ncol(); j++) {
            slop = mat.cal_slop(i, j);
            if (forward)
                ret(i, j) -= dt_remain * mat.step() * slop;
            else
                ret(i, j) += dt_remain * mat.step() * slop;
        }
    }
    mat = ret;
    return mat;
}


bool assertEqual(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.nrow() != mat2.nrow())
        return false;
    if (mat1.ncol() != mat2.ncol())
        return false;
    for (size_t i=0; i<mat1.nrow();++i)
    {
        for (size_t j=0; j<mat1.ncol();++j)
        {
            if (mat1(i,j) != mat2(i,j))
                return false;
        }
    }
    return true;
}



/*
int main(int argc, char ** argv)
{
    const size_t n = 3;
    int status;

    std::cout << ">>> Solve Ax=b (row major)" << std::endl;
    Matrix mat(n, n);
    mat(0,0) = 3; mat(0,1) = 5; mat(0,2) = 2;
    mat(1,0) = 2; mat(1,1) = 1; mat(1,2) = 3;
    mat(2,0) = 4; mat(2,1) = 3; mat(2,2) = 2;
    Matrix b(n, 2);
    b(0,0) = 57; b(0,1) = 23;
    b(1,0) = 22; b(1,1) = 12;
    b(2,0) = 41; b(2,1) = 84;
    std::vector<int> ipiv(n);

    std::cout << "A:" << mat << std::endl;
    std::cout << "b:" << b << std::endl;

    Matrix result1(mat.nrow(), b.ncol());
    Matrix result2(mat.nrow(), b.ncol());
    result1 = multiply_naive(mat, b);
    std::cout << "result1:" << result1 << std::endl;
    result2 = multiply_tile(mat, b, 2);
    std::cout << "result2:" << result2 << std::endl;

    return 0;
}
 */
PYBIND11_MODULE(_matrix, m){
m.doc() = "pybin11 plugin to calculate the matrix matrix multiply";
py::class_<Matrix>(m, "Matrix")
.def(py::init<size_t, size_t>())
.def(py::init<size_t, size_t, double, double, double, double, double, double>())
.def_property_readonly("nrow", &Matrix::nrow)
.def_property_readonly("ncol", &Matrix::ncol)
.def("__getitem__", [](Matrix &self, std::pair<size_t, size_t> index)
{ return self(index.first, index.second); })
.def("__setitem__", [](Matrix &self, std::pair<size_t, size_t> index, double value)
{ self(index.first, index.second) = value;})
.def("__eq__", [](Matrix & mat1, Matrix & mat2){ return assertEqual(mat1, mat2); })
.def("__ne__", [](Matrix & mat1, Matrix & mat2){ return !assertEqual(mat1, mat2); })
.def_property("array", &Matrix::nd_array, nullptr )
.def("set_all_0", &Matrix::set_all_0, "set all values in matrix to zero")
.def("cal_max_dt", &Matrix::cal_max_dt, "calculate minimum allowed dt")
.def("cal_slop", &Matrix::cal_slop, "calculate the slop of (x, y)")
//.def_property_readonly("d", &Matrix::d)
.def_property_readonly("b", &Matrix::b)
.def_property_readonly("k", &Matrix::k)
.def_property_readonly("r", &Matrix::r)
.def_property_readonly("t", &Matrix::t)
.def_property_readonly("step", &Matrix::step)
.def_property_readonly("dx", &Matrix::dx)
.def_property_readonly("dy", &Matrix::dy)
.def_property_readonly("max_dt", &Matrix::max_dt)
//.def("set_d", &Matrix::set_d)
.def("set_b", &Matrix::set_b)
.def("set_k", &Matrix::set_k)
.def("set_r", &Matrix::set_r)
.def("set_step", &Matrix::set_step)
.def("set_dx", &Matrix::set_dx)
.def("set_dy", &Matrix::set_dy);

m.def("iterate", &iterate, "iterate forward or backward with dt");
m.def("multiply_tile", &multiply_tile, "tiled matrix-matrix multiply");
m.def("multiply_naive", &multiply_naive, "Naive matrix-matrix multiply");
m.def("multiply_mkl", &multiply_mkl, "mkl matrix-matrix multiply");
}