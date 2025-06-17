#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include <cmath>
#include <iostream>


namespace py = pybind11;


// edges of the decoding tree
class Edge {
public:
    int size;
    double *data;
    Edge *root, *lbro, *rbro, *lchl, *rchl;
public:
    void relation_get(const Edge *from);
};


// multi-dimensional joint distribution
class JointProb {
private:
    int *shape;
    int ndim, size;
    int **multi_index;
    int *temp_index;
    double *array1, *array2, *array_;
    fftw_complex *farray1, *farray2, *farray_;
    fftw_plan plan1, plan2, plan_;
public:
    JointProb(const int *shape, int ndim);
    ~JointProb();
    const int *get_shape() { return this->shape; }
    int get_shape(int dim) { return this->shape[dim]; }
    int get_ndim() { return this->ndim; }
    int get_size() { return this->size; }
    const int *from_linear(int i) { return this->multi_index[i]; }
    int twrd_linear(const int *index);
    void datacpy(const double *from, double *twrd);
    void inverse(const double *from, double *twrd);
    void nrmcomb(const double *from1, const double *from2, double *twrd);
    void circonv(const double *from1, const double *from2, double *twrd);
};


// a walker on the Baysian network of polar transform
class Walker {
private:
    int code_len;
    int code_lvl;
    JointProb *jprob;
    int jpsize;
    double *probs;
    int branch_now;
    double *tmp;
    Edge *tree;
public:
    Walker(int code_len, JointProb *jprob);
    ~Walker();
    void reset();
    void set_priors(const double *priors);
    void step(int branch_new, Edge *edge_new);
    int get_branch() { return this->branch_now; }
    Edge *get_edge(int branch) { return this->tree + branch; }
    const Edge *get_next(int branch_next);
private:
    void update_root(const Edge *lchl, const Edge *rchl, Edge *root);
    void update_lchl(Edge *lchl, const Edge *rchl, const Edge *root);
    void update_rchl(const Edge *lchl, Edge *rchl, const Edge *root);
};
