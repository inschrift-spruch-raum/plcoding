#include "utils.h"


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

int JointProb::to_linear(const int *index) {
    int k = 0;
    for (int i = 0; i < this->ndim; ++i) {
        k = k * this->shape[i] + index[i];
    }
    return k;
}

void JointProb::datacpy(const double *from, double *to) {
    for (int i = 0; i < this->size; ++i) {
        to[i] = from[i];
    }
}

void JointProb::inverse(const double *from, double *to) {
    for (int i = 0; i < this->size; ++i) {
        for (int j = 0; j < this->ndim; ++j) {
            this->temp_index[j] = (this->shape[j] - this->multi_index[i][j]) % this->shape[j];
        }
        to[i] = from[this->to_linear(this->temp_index)];
    }
}

void JointProb::nrmcomb(const double *from1, const double *from2, double *to) {
    for (int i = 0; i < this->size; ++i) {
        to[i] = from1[i] * from2[i];
    }
    this->normalize(to);
}

void JointProb::circonv(const double *from1, const double *from2, double *to) {
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
        to[i] = this->array_[i] / this->size;
    }
}

void JointProb::normalize(double *target) {
    double tau = 0.0;
    for (int i = 0; i < this->size; ++i) {
        tau += target[i];
    }
    for (int i = 0; i < this->size; ++i) {
        target[i] /= tau;
    }
}

Edge::Edge(int size, JointProb *jprob) {
    if (size < 1) {
        throw std::runtime_error("Edge size must be >= 1!");
    }
    this->size = size;
    this->jprob = jprob;
    this->data = new double[jprob->get_size() * size];
}

Edge::~Edge() {
    delete[] this->data;
}

void Edge::copy_ptrs(const Edge *from) {
    this->root = from->root;
    this->lbro = from->lbro;
    this->rbro = from->rbro;
    this->lchl = from->lchl;
    this->rchl = from->rchl;
}

void Edge::copy_data(const Edge *from) {
    this->copy_data(from->data);
}

void Edge::copy_data(const double *pt) {
    int data_size = this->size * this->jprob->get_size();
    for (int i = 0; i < data_size; ++i) {
        this->data[i] = pt[i];
    }
}

void Edge::set_uniform() {
    int data_size = this->size * this->jprob->get_size();
    double p_value = 1.0 / this->jprob->get_size();
    for (int i = 0; i < data_size; ++i) {
        this->data[i] = p_value;
    }
}

// this method provides an elegant way to obtain the corresponding pointer attribute
Edge *&Edge::get_ptr(stype type) {
    if (type == left_to_right) {
        return this->rbro;
    } else if (type == right_to_left) {
        return this->lbro;
    } else if (type == parent_to_left) {
        return this->lchl;
    } else if (type == parent_to_right) {
        return this->rchl;
    } else {
        return this->root;
    }
}

double Edge::partially_judge_from(const Edge *from, int var, int value) {
    if (this->size != 1) {
        throw std::runtime_error("Invalid partially judged edge!");
    }
    int jpsize = this->jprob->get_size();
    // transfer data and set zero at corresponding indices
    this->copy_data(from->data);
    for (int i = 0; i < jpsize; ++i) {
        if (this->jprob->from_linear(i)[var] != value) {
            this->data[i] = 0.0;
        }
    }
    this->jprob->normalize(this->data);
    // calculate the marginal likelihood
    double likelihood = 0.0;
    for (int i = 0; i < jpsize; ++i) {
        if (this->jprob->from_linear(i)[var] == value) {
            likelihood += this->data[i];
        }
    }
    // set the distribution to be partially uniform
    for (int i = 0; i < jpsize; ++i) {
        if (this->data[i] != 0.0) {
            this->data[i] = 1;
        }
    }
    return std::log(likelihood);
}

void Edge::check_params(int var, int value) {
    if (var < 0 || var >= this->jprob->get_ndim()) {
        throw std::runtime_error("Invalid variable!");
    }
    if (value < 0 || value >= this->jprob->get_shape(var)) {
        throw std::runtime_error("Invalid value!");
    }
}

DecodingWalker::DecodingWalker(int code_len, JointProb *jprob) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->jprob = jprob;
    // note that the decoding tree has an additional root edge
    int tree_size = 2 * this->code_len - 1;
    this->edges = new Edge*[tree_size];
    for (int i = 0; i < tree_size; ++i) {
        int edge_size = this->edge_size_at(this->depth_of(i));
        this->edges[i] = new Edge(edge_size, jprob);
    }
    // allocate memories for the lazy update operation
    this->buffers = new Edge*[this->code_lvl + 1];
    for (int i = 0; i <= this->code_lvl; ++i) {
        int edge_size = this->edge_size_at(i);
        this->buffers[i] = new Edge(edge_size, jprob);
    }
    // allocate memories for update operations
    this->tmp_data = new double[jprob->get_size()];
    this->reset();
}

DecodingWalker::~DecodingWalker() {
    // free the decoding tree
    int tree_size = 2 * this->code_len - 1;
    for (int i = 0; i < tree_size; ++i) {
        delete this->edges[i];
    }
    delete[] this->edges;
    // free buffers
    for (int i = 0; i <= this->code_lvl; ++i) {
        delete this->buffers[i];
    }
    delete[] this->buffers;
}

void DecodingWalker::reset() {
    this->branch_now = 0;
    // reset the decoding tree
    int tree_size = 2 * this->code_len - 1;
    // reset the probabilities to uniform
    for (int i = 1; i < tree_size; ++i) {
        this->edges[i]->set_uniform();
    }
    for (int i = 0; i < tree_size; ++i) {
        // set parents for edges
        if (this->depth_of(i) >= 1) {
            this->edges[i]->root = this->edges[(i - 1) / 2];
            if (i % 2 == 1) {
                this->edges[i]->rbro = this->edges[i + 1];
            } else {
                this->edges[i]->lbro = this->edges[i - 1];
            }
        }
        // set children for edges
        if (this->depth_of(i) < this->code_lvl) {
            this->edges[i]->lchl = this->edges[i * 2 + 1];
            this->edges[i]->rchl = this->edges[i * 2 + 2];
        }
    }
}

void DecodingWalker::set_priors(const double *priors) {
    this->edges[0]->copy_data(priors);
}

// maintain an unchanged tree view
void DecodingWalker::lazy_step(int branch_to) {
    Edge *edge_now = this->edges[this->branch_now];
    Edge *buffer = this->buffers[this->depth_of(branch_to)];
    stype type1 = this->step_type(this->branch_now, branch_to);
    stype type2 = this->step_type(branch_to, this->branch_now);
    Edge *edge_old = edge_now->get_ptr(type1);
    this->calc_edge(edge_now, buffer, type1);
    buffer->copy_ptrs(edge_old);
    buffer->get_ptr(type2) = edge_now;
    // if the next edge is in the bottom layer, nrmcomb the result with previous data
    if (this->depth_of(branch_to) == this->code_lvl) {
        this->jprob->nrmcomb(edge_old->data, buffer->data, buffer->data);
    }
    this->branch_now = branch_to;
}

// update the buffer data to the real edge
void DecodingWalker::flush_buffer() {
    // transfer data and pointers
    Edge *buffer = this->buffers[this->depth_of(this->branch_now)];
    Edge *edge_now = this->edges[this->branch_now];
    edge_now->copy_data(buffer);
    edge_now->copy_ptrs(buffer);
}

// calculate the data part of <edge_to> based on the given <edge_from> and their step <type>
void DecodingWalker::calc_edge(Edge *edge_from, Edge *edge_to, stype type) {
    if (type == left_to_parent) {
        this->calc_root(edge_from, edge_from->rbro, edge_to);
    } else if (type == right_to_parent) {
        this->calc_root(edge_from->lbro, edge_from, edge_to);
    } else if (type == parent_to_left) {
        this->calc_lchl(edge_to, edge_from->rchl, edge_from);
    } else if (type == parent_to_right) {
        this->calc_rchl(edge_from->lchl, edge_to, edge_from);
    } else if (type == left_to_right) {
        this->calc_rchl(edge_from, edge_to, edge_from->root);
    } else if (type == right_to_left) {
        this->calc_lchl(edge_to, edge_from, edge_from->root);
    } else {
        throw std::runtime_error("Invalid parameters for calc_edge()!");
    }
}

void DecodingWalker::calc_root(const Edge *lchl, const Edge *rchl, Edge *root) {
    int pnum = lchl->size;
    int plen = this->jprob->get_size();
    double *x1 = root->data, *x2 = x1 + pnum * plen, *u1 = lchl->data, *u2 = rchl->data;
    for (int i = 0; i < pnum; ++i) {
        this->jprob->circonv(u1, u2, x1);
        this->jprob->datacpy(u2, x2);
        x1 += plen; x2 += plen; u1 += plen; u2 += plen;
    }
}

void DecodingWalker::calc_lchl(Edge *lchl, const Edge *rchl, const Edge *root) {
    int pnum = lchl->size;
    int plen = this->jprob->get_size();
    double *x1 = root->data, *x2 = x1 + pnum * plen, *u1 = lchl->data, *u2 = rchl->data;
    for (int i = 0; i < pnum; ++i) {
        this->jprob->nrmcomb(u2, x2, u1);
        this->jprob->inverse(u1, this->tmp_data);
        this->jprob->circonv(x1, this->tmp_data, u1);
        x1 += plen; x2 += plen; u1 += plen; u2 += plen;
    }
}

void DecodingWalker::calc_rchl(const Edge *lchl, Edge *rchl, const Edge *root) {
    int pnum = lchl->size;
    int plen = this->jprob->get_size();
    double *x1 = root->data, *x2 = x1 + pnum * plen, *u1 = lchl->data, *u2 = rchl->data;
    for (int i = 0; i < pnum; ++i) {
        this->jprob->inverse(u1, u2);
        this->jprob->circonv(x1, u2, this->tmp_data);
        this->jprob->nrmcomb(x2, this->tmp_data, u2);
        x1 += plen; x2 += plen; u1 += plen; u2 += plen;
    }
}

// get the step type between two branches
stype DecodingWalker::step_type(int branch_from, int branch_to) {
    int depth_from = this->depth_of(branch_from);
    int depth_to = this->depth_of(branch_to);
    if (((branch_from - 1) / 2 == branch_to) && (depth_from == depth_to + 1)) {
        // case 1: from children to parent
        return (branch_from % 2 == 1) ? left_to_parent : right_to_parent;
    } else if ((branch_from == (branch_to - 1) / 2) && (depth_from == depth_to - 1)) {
        // case 2: from parent to children
        return (branch_to % 2 == 1) ? parent_to_left : parent_to_right;
    } else if ((depth_from == depth_to) && (branch_from == branch_to - 1)) {
        // case 3.1: from brother to brother
        return left_to_right;
    } else if ((depth_from == depth_to) && (branch_from - 1 == branch_to)) {
        // case 3.2: from brother to brother
        return right_to_left;
    } else {
        throw std::runtime_error("Invalid stype, not adjacent!");
    }
}

void DecodingWalker::check_params(int var, int index, int value) {
    this->edges[0]->check_params(var, value);
    if (index < 0 || index >= this->code_len) {
        throw std::runtime_error("Invalid index!");
    }
}

// get the path between any two branches on the decoding tree
void DecodingWalker::get_path(int branch_from, int branch_to, int *path, int path_len) {
    for (int i = 0; i < path_len; ++i) {
        path[i] = -1;
    }
    int up = 0, dn = 0;
    // two branches are not necessarily on the same level
    while (branch_from != branch_to) {
        while (branch_from > branch_to) {
            path[up] = branch_from;
            branch_from = (branch_from - 1) / 2;
            up += 1;
        }
        while (branch_from < branch_to) {
            path[path_len - 1 - dn] = branch_to;
            branch_to = (branch_to - 1) / 2;
            dn += 1;
        }
    }
}

// lead the walkers going through the path of the decoding tree
void DecodingWalker::walk_to(DecodingWalker **walkers, int walker_num, int branch_to) {
    int branch_from = walkers[0]->branch_now;
    int path_len = walkers[0]->code_lvl * 2 + 1;
    int *path = new int[path_len];
    DecodingWalker::get_path(branch_from, branch_to, path, path_len);
    for (int i = 0; i < path_len; ++i) {
        int branch_next = path[i];
        if (branch_next == -1) continue;
        for (int j = 0; j < walker_num; ++j) {
            walkers[j]->lazy_step(branch_next);
        }
        for (int j = 0; j < walker_num; ++j) {
            walkers[j]->flush_buffer();
        }
    }
    delete[] path;
}
