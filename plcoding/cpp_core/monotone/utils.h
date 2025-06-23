#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include <cmath>
#include <iostream>
#include <algorithm>


namespace py = pybind11;


// multi-dimensional joint distribution unit
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
    int to_linear(const int *index);
    void datacpy(const double *from, double *to);
    void inverse(const double *from, double *to);
    void nrmcomb(const double *from1, const double *from2, double *to);
    void circonv(const double *from1, const double *from2, double *to);
    void normalize(double *target);
};


// the step type between two branches
enum stype {parent_to_left, parent_to_right, left_to_right, right_to_left, left_to_parent, right_to_parent};


// edges of the decoding tree
class Edge {
public:
    int size;
    JointProb *jprob;
    double *data;
    Edge *root, *lbro, *rbro, *lchl, *rchl;
public:
    Edge(int size, JointProb *jprob);
    ~Edge();
    void copy_ptrs(const Edge *from);
    void copy_data(const Edge *from);
    void copy_data(const double *pt);
    void set_uniform();
    Edge *&get_ptr(stype type);
    double partially_judge_from(const Edge *from, int var, int value);
    void check_params(int var, int value = 0);
};


// decoding tree and the walker
class DecodingWalker {
public:
    int code_len;
    int code_lvl;
    JointProb *jprob;
    Edge **edges;
    int branch_now;
    Edge **buffers;
    double *tmp_data;
public: // external interface functions
    DecodingWalker(int code_len, JointProb *jprob);
    ~DecodingWalker();
    void reset();
    void set_priors(const double *priors);
    void lazy_step(int branch_to);
    void flush_buffer();
    Edge *get_head() { return this->edges[this->branch_now]; }
public: // internal utility functions
    int edge_size_at(int level) { return (1 << (this->code_lvl - level)); }
    int depth_of(int branch) { return std::ceil(std::log2(branch + 2)) - 1; }
    void calc_edge(Edge *edge_from, Edge *edge_to, stype type);
    void calc_root(const Edge *lchl, const Edge *rchl, Edge *root);
    void calc_lchl(Edge *lchl, const Edge *rchl, const Edge *root);
    void calc_rchl(const Edge *lchl, Edge *rchl, const Edge *root);
    stype step_type(int branch_from, int branch_to);
public: // external utility functions
    void check_params(int var, int index, int value = 0);
    static void get_path(int branch_from, int branch_to, int *path, int path_len);
    static void walk_to(DecodingWalker **walkers, int walker_num, int branch_to);
};
