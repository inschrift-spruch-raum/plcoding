#include "utils.h"


void Edge::relation_get(const Edge *from) {
    this->root = from->root;
    this->lbro = from->lbro;
    this->rbro = from->rbro;
    this->lchl = from->lchl;
    this->rchl = from->rchl;
}

JointProb::JointProb(const int *shape, int ndim) {
    this->shape = new int[ndim];
    this->ndim = ndim;
    this->size = 1;
    for (int i = 0; i < ndim; ++i) {
        this->shape[i] = shape[i];
        this->size *= shape[i];
    }
    this->multi_index = new int*[this->size];
    for (int i = 0; i < this->size; ++i) {
        this->multi_index[i] = new int[ndim];
        int rem = i;
        for (int j = ndim - 1; j >= 0; --j) {
            this->multi_index[i][j] = rem % this->shape[j];
            rem /= this->shape[j];
        }
    }
    this->temp_index = new int[ndim];
    // fast operation
    this->array1 = fftw_alloc_real(this->size);
    this->array2 = fftw_alloc_real(this->size);
    this->array_ = fftw_alloc_real(this->size);
    this->farray1 = fftw_alloc_complex(this->size);
    this->farray2 = fftw_alloc_complex(this->size);
    this->farray_ = fftw_alloc_complex(this->size);
    this->plan1 = fftw_plan_dft_r2c(ndim, shape, array1, farray1, FFTW_MEASURE);
    this->plan2 = fftw_plan_dft_r2c(ndim, shape, array2, farray2, FFTW_MEASURE);
    this->plan_ = fftw_plan_dft_c2r(ndim, shape, farray_, array_, FFTW_MEASURE);
}

JointProb::~JointProb() {
    delete[] this->shape;
    for (int i = 0; i < this->size; ++i) {
        delete[] this->multi_index[i];
    }
    delete[] this->multi_index;
    delete[] this->temp_index;
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

int JointProb::twrd_linear(const int *index) {
    int k = 0;
    for (int i = 0; i < this->ndim; ++i) {
        k = k * this->shape[i] + index[i];
    }
    return k;
}

void JointProb::datacpy(const double *from, double *twrd) {
    for (int i = 0; i < this->size; ++i) {
        twrd[i] = from[i];
    }
}

void JointProb::inverse(const double *from, double *twrd) {
    for (int i = 0; i < this->size; ++i) {
        for (int j = 0; j < this->ndim; ++j) {
            this->temp_index[j] = (this->shape[j] - this->multi_index[i][j]) % this->shape[j];
        }
        twrd[i] = from[this->twrd_linear(this->temp_index)];
    }
}

void JointProb::nrmcomb(const double *from1, const double *from2, double *twrd) {
    double tau = 0.0;
    for (int i = 0; i < this->size; ++i) {
        twrd[i] = from1[i] * from2[i];
        tau += twrd[i];
    }
    for (int i = 0; i < this->size; ++i) {
        twrd[i] /= tau;
    }
}

void JointProb::circonv(const double *from1, const double *from2, double *twrd) {
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
        twrd[i] = this->array_[i] / this->size;
    }
}

Walker::Walker(int code_len, JointProb *jprob) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->jprob = jprob;
    this->jpsize = jprob->get_size();
    // allocate memory
    this->probs = new double[(this->code_lvl + 1) * this->code_len * this->jpsize];
    this->tree = new Edge[2 * this->code_len - 1];
    this->tmp = new double[this->jpsize];
    // reset the walker
    this->reset();
}

Walker::~Walker() {
    delete[] this->probs;
    delete[] this->tree;
    delete[] this->tmp;
}

void Walker::reset() {
    this->branch_now = 0;
    // clear probabilities
    int offset = this->code_len * this->jpsize;
    for (int i = 0; i < this->code_lvl * this->code_len * this->jpsize; ++i) {
        this->probs[i + offset] = 1.0 / this->jpsize;
    }
    // clear the decoding tree
    for (int i = 0; i <= this->code_lvl; ++i) {
        int edge_num  = (1 << i);
        int edge_size = this->code_len / edge_num;
        for (int j = 0; j < edge_num; ++j) {
            // edges pointing to local memory
            int branch = (1 << i) - 1 + j;
            Edge *edge = this->tree + branch;
            edge->size = edge_size;
            edge->data = this->probs + (i * this->code_len + j * edge_size) * this->jpsize;
            // relabel the neighbor relationships
            if (i > 0) {
                edge->root = this->tree + (branch - 1) / 2;
                if (branch % 2 == 1) {
                    edge->rbro = this->tree + (branch + 1);
                } else {
                    edge->lbro = this->tree + (branch - 1);
                }
            }
            if (i < this->code_lvl) {
                edge->lchl = this->tree + (branch * 2 + 1);
                edge->rchl = this->tree + (branch * 2 + 2);
            }
        }
    }
}

void Walker::set_priors(const double *priors) {
    for (int i = 0; i < this->code_len * this->jpsize; ++i) {
        this->probs[i] = priors[i];
    }
}

void Walker::step(int branch_new, Edge *edge_new) {
    if (this->branch_now == branch_new) return;
    Edge *edge_now = this->tree + this->branch_now;
    // case 1: from children to parent
    if ((this->branch_now - 1) / 2 == branch_new) {
        edge_new->relation_get(edge_now->root);
        if (this->branch_now % 2 == 1) {
            this->update_root(edge_now, edge_now->rbro, edge_new);
            edge_now->root = edge_new;
            edge_new->lchl = edge_now;
        } else {
            this->update_root(edge_now->lbro, edge_now, edge_new);
            edge_now->root = edge_new;
            edge_new->rchl = edge_now;
        }
    }
    // case 2: from parent to children
    else if (this->branch_now == (branch_new - 1) / 2) {
        if (branch_new % 2 == 1) {
            edge_new->relation_get(edge_now->lchl);
            this->update_lchl(edge_new, edge_now->rchl, edge_now);
            edge_now->lchl = edge_new;
            edge_new->root = edge_now;
        } else {
            edge_new->relation_get(edge_now->rchl);
            this->update_rchl(edge_now->lchl, edge_new, edge_now);
            edge_now->rchl = edge_new;
            edge_new->root = edge_now;
        }
    }
    // case 3: to brother
    else if (branch_new - 1 == this->branch_now) {
        edge_new->relation_get(edge_now->rbro);
        this->update_rchl(edge_now, edge_new, edge_now->root);
        edge_now->rbro = edge_new;
        edge_new->lbro = edge_now;
    }
    else if (this->branch_now - 1 == branch_new) {
        edge_new->relation_get(edge_now->lbro);
        this->update_lchl(edge_new, edge_now, edge_now->root);
        edge_now->lbro = edge_new;
        edge_new->rbro = edge_now;
    } else {
        throw std::runtime_error("Unreasonable branch number for 'step', not adjacent!");
    }
    // update state
    this->branch_now = branch_new;
}

const Edge *Walker::get_next(int branch_next) {
    Edge *edge_now = this->tree + this->branch_now;
    if ((this->branch_now - 1) / 2 == branch_next) {
        return edge_now->root;
    } else if (this->branch_now == (branch_next - 1) / 2) {
        return (branch_next % 2 == 1) ? edge_now->lchl : edge_now->rchl;
    } else if ((branch_next % 2 == 0) && (branch_next - 1 == this->branch_now)) {
        return edge_now->rbro;
    } else if ((branch_next % 2 == 1) && (branch_next + 1 == this->branch_now)) {
        return edge_now->lbro;
    } else {
        throw std::runtime_error("Unreasonable branch number for 'get_next', not adjacent!");
    }
}

void Walker::update_root(const Edge *lchl, const Edge *rchl, Edge *root)
{
    int offset = lchl->size;
    double *x1 = root->data;
    double *x2 = root->data + offset * this->jpsize;
    double *u1 = lchl->data;
    double *u2 = rchl->data;
    for (int i = 0; i < offset; ++i) {
        this->jprob->circonv(u1, u2, x1);
        this->jprob->datacpy(u2, x2);
        x1 += this->jpsize;
        x2 += this->jpsize;
        u1 += this->jpsize;
        u2 += this->jpsize;
    }
}

void Walker::update_lchl(Edge *lchl, const Edge *rchl, const Edge *root) {
    int offset = lchl->size;
    double *x1 = root->data;
    double *x2 = root->data + offset * this->jpsize;
    double *u1 = lchl->data;
    double *u2 = rchl->data;
    for (int i = 0; i < offset; ++i) {
        this->jprob->nrmcomb(u2, x2, u1);
        this->jprob->inverse(u1, this->tmp);
        this->jprob->circonv(x1, this->tmp, u1);
        x1 += this->jpsize;
        x2 += this->jpsize;
        u1 += this->jpsize;
        u2 += this->jpsize;
    }
}

void Walker::update_rchl(const Edge *lchl, Edge *rchl, const Edge *root) {
    int offset = lchl->size;
    double *x1 = root->data;
    double *x2 = root->data + offset * this->jpsize;
    double *u1 = lchl->data;
    double *u2 = rchl->data;
    for (int i = 0; i < offset; ++i) {
        this->jprob->inverse(u1, u2);
        this->jprob->circonv(x1, u2, this->tmp);
        this->jprob->nrmcomb(x2, this->tmp, u2);
        x1 += this->jpsize;
        x2 += this->jpsize;
        u1 += this->jpsize;
        u2 += this->jpsize;
    }
}
