#include "utils.h"


ND_Shape::ND_Shape(const int *bases, int nvar) {
    this->nvar = nvar;
    this->bases = new int[nvar];
    this->size = 1;
    for (int i = 0; i < nvar; ++i) {
        this->bases[i] = bases[i];
        this->size *= bases[i];
    }
    this->nd_indices = new int*[this->size];
    for (int i = 0; i < this->size; ++i) {
        this->nd_indices[i] = new int[nvar];
        int rem = i;
        for (int j = nvar - 1; j >= 0; --j) {
            this->nd_indices[i][j] = rem % this->bases[j];
            rem /= this->bases[j];
        }
    }
    this->reverse_map = new int[this->size];
    int *nd_index = new int[nvar];
    for (int i = 0; i < this->size; ++i) {
        for (int j = 0; j < nvar; ++j)
            nd_index[j] = (bases[j] - this->nd_indices[i][j]) % bases[j];
        this->reverse_map[i] = this->to_linear(nd_index);
    }
    delete[] nd_index;
    // fast operation
    this->array1 = fftw_alloc_real(this->size);
    this->array2 = fftw_alloc_real(this->size);
    this->array_ = fftw_alloc_real(this->size);
    this->farray1 = fftw_alloc_complex(this->size);
    this->farray2 = fftw_alloc_complex(this->size);
    this->farray_ = fftw_alloc_complex(this->size);
    this->plan1 = fftw_plan_dft_r2c(nvar, bases, array1, farray1, FFTW_MEASURE);
    this->plan2 = fftw_plan_dft_r2c(nvar, bases, array2, farray2, FFTW_MEASURE);
    this->plan_ = fftw_plan_dft_c2r(nvar, bases, farray_, array_, FFTW_MEASURE);
}

ND_Shape::~ND_Shape() {
    delete[] this->bases;
    for (int i = 0; i < this->size; ++i)
        delete[] this->nd_indices[i];
    delete[] this->nd_indices;
    fftw_destroy_plan(this->plan1);
    fftw_destroy_plan(this->plan2);
    fftw_destroy_plan(this->plan_);
    fftw_free(this->array1);
    fftw_free(this->array2);
    fftw_free(this->array_);
    fftw_free(this->farray1);
    fftw_free(this->farray2);
    fftw_free(this->farray_);
}

void ND_Shape::set_partial(int var, int value, double *data) const {
    double tau = 0.0;
    for (int i = 0; i < this->size; ++i)
        if (this->nd_indices[i][var] == value && data[i] != 0) {
            data[i] = 1.0;
            tau += 1.0;
        } else {
            data[i] = 0.0;
        }
    if (tau == 0)
        std::cout << "found tau=0 in set_partial()" << std::endl;
    for (int i = 0; i < this->size; ++i)
        data[i] /= tau;
}

int ND_Shape::to_linear(const int *index) const {
    int k = 0;
    for (int i = 0; i < this->nvar; ++i)
        k = k * this->bases[i] + index[i];
    return k;
}

void ND_Shape::nrmcomb(const double *from1, const double *from2, double *to) const {
    double tau = 0.0;
    for (int i = 0; i < this->size; ++i) {
        to[i] = from1[i] * from2[i];
        tau += to[i];
    }
    if (tau == 0)
        std::cout << "found tau=0 in nrmcomb()" << std::endl;
    for (int i = 0; i < this->size; ++i)
        to[i] /= tau;
}

void ND_Shape::circonv(const double *from1, const double *from2, double *to) const {
    for (int i = 0; i < this->size; ++i) {
        this->array1[i] = from1[i];
        this->array2[i] = from2[i];
    }
    fftw_execute(this->plan1);
    fftw_execute(this->plan2);
    // multiplication of complex numbers
    for (int i = 0; i < this->size; ++i) {
        double a = this->farray1[i][0], b = this->farray1[i][1];
        double c = this->farray2[i][0], d = this->farray2[i][1];
        this->farray_[i][0] = a * c - b * d;
        this->farray_[i][1] = a * d + b * c;
    }
    fftw_execute(this->plan_);
    for (int i = 0; i < this->size; ++i) {
        double p = this->array_[i] / this->size;
        to[i] = (p <= 0) ? 1e-12 : p;
    }
}

Edge::Edge(int branch, int size, const ND_Shape *shape) {
    this->branch = branch;
    this->size = size;
    this->shape = shape;
    this->from = nullptr;
    this->data = new double*[size];
    for (int i = 0; i < this->size; ++i) {
        this->data[i] = new double[shape->get_size()];
        shape->set_uniform(this->data[i]);
    }
}

Edge::Edge(const Edge *edge) {
    this->branch = edge->branch;
    this->size = edge->size;
    this->shape = edge->shape;
    this->from = edge->from;
    this->data = new double*[this->size];
    for (int i = 0; i < this->size; ++i) {
        this->data[i] = new double[this->shape->get_size()];
        this->shape->copy(edge->data[i], this->data[i]);
    }
}

Edge::~Edge() {
    for (int i = 0; i < this->size; ++i)
        delete[] this->data[i];
    delete this->data;
}

void Edge::clear_data() {
    for (int i = 0; i < this->size; ++i)
        this->shape->set_uniform(this->data[i]);
}

void Edge::copy_from(const Edge *edge) {
    this->branch = edge->branch;
    this->size = edge->size;
    this->shape = edge->shape;
    this->from = edge->from;
    for (int i = 0; i < this->size; ++i)
        this->shape->copy(edge->data[i], this->data[i]);
}

void Edge::set_probs(const double *probs) {
    for (int i = 0; i < this->size; ++i)
        for (int j = 0; j < this->shape->get_size(); ++j)
            this->data[i][j] = probs[i * this->shape->get_size() + j];
}

void Edge::combine_with(const Edge *edge) {
    for (int i = 0; i < this->size; ++i)
        this->shape->nrmcomb(this->data[i], edge->data[i], this->data[i]);
}

Vertex::Vertex(int branch) {
    this->branch = branch;
    this->parent = nullptr;
    this->left = nullptr;
    this->right = nullptr;
}

Vertex::Vertex(const Vertex *vertex) {
    this->copy_from(vertex);
}

void Vertex::copy_from(const Vertex *vertex) {
    this->branch = vertex->branch;
    this->parent = vertex->parent;
    this->left = vertex->left;
    this->right = vertex->right;
}

void Vertex::calc_parent(Edge *result) {
    const ND_Shape *shape = this->left->shape;
    int child_size = this->left->size;
    for (int i = 0; i < child_size; ++i) {
        shape->circonv(this->left->data[i], this->right->data[i], result->data[i]);
        shape->copy(this->right->data[i], result->data[i + child_size]);
    }
}

void Vertex::calc_left(Edge *result) {
    const ND_Shape *shape = this->left->shape;
    int child_size = this->left->size;
    double *temp = new double[shape->get_size()];
    for (int i = 0; i < child_size; ++i) {
        shape->nrmcomb(this->parent->data[i + child_size], this->right->data[i], result->data[i]);
        shape->reverse(result->data[i], temp);
        shape->circonv(this->parent->data[i], temp, result->data[i]);
    }
}

void Vertex::calc_right(Edge *result) {
    const ND_Shape *shape = this->left->shape;
    int child_size = this->left->size;
    double *temp = new double[shape->get_size()];
    for (int i = 0; i < child_size; ++i) {
        shape->reverse(this->left->data[i], result->data[i]);
        shape->circonv(this->parent->data[i], result->data[i], temp);
        shape->nrmcomb(this->parent->data[i + child_size], temp, result->data[i]);
    }
}
