#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>


namespace py = pybind11;
class NDimShape;
class JointProb;
class Edge;
class Node;
class Tree;
class Walker;


// multi-dimension shape operator
class NDimShape {
private:
    int *shape;
    int ndim, size;
    int **nd_indices;
    double *array1, *array2, *array_;
    fftw_complex *farray1, *farray2, *farray_;
    fftw_plan plan1, plan2, plan_;
public:
    NDimShape(const int *shape, int ndim);
    ~NDimShape();
    const int *get_shape() { return this->shape; }
    int get_ndim() { return this->ndim; }
    int get_size() { return this->size; }
    const int *from_linear(int i) { return this->nd_indices[i]; }
    int to_linear(const int *index);
    void nrmcomb(const double *from1, const double *from2, double *to);
    void circonv(const double *from1, const double *from2, double *to);
    void normalize(double *target);
};

// joint probability distribution
class JointProb {
private:
    int *values;
    double *data;
    NDimShape *shape;
public:
    JointProb(NDimShape *shape);
    ~JointProb();
    void reset();
    void calc_marginal(int var, double *output);
    void set_uniform();
    void decision_upon(int var, int value);
    void set_prior(const double *prior);
    void sum(JointProb *input1, JointProb *input2);
    void copy_from(JointProb *input);
    void combine(JointProb *input1, JointProb *input2);
    void reverse(JointProb *input);
    void subtract(JointProb *input1, JointProb *input2);
    void print(bool _not_last = true);
};

// directed edges of the decoding tree
class Edge {
private:
    int size;
    JointProb **probs;
    NDimShape *shape;
    Node *node_from;
public:
    Edge(int size, NDimShape *shape);
    ~Edge();
    void reset();
    int get_size() { return this->size; }
    JointProb *get_prob(int index = 0) { return this->probs[index]; }
    Node *get_node_from() { return this->node_from; }
    void set_node_from(Node *node) { this->node_from = node; }
    void set_uniform();
    void set_probs(const double *probs);
    void copy_from(const Edge *input);
    void combine_with(const Edge *input);
    void print(bool _not_last = true);
public:
    static void update_root(Edge *root, Edge *lchl, Edge *rchl);
    static void update_lchl(Edge *root, Edge *lchl, Edge *rchl);
    static void update_rchl(Edge *root, Edge *lchl, Edge *rchl);
};

// nodes of the decoding tree
// type: 0 root, 1 left child, 2 right child, -1 others
class Node {
private:
    int branch;
    Edge *edge_root;
    Edge *edge_lchl;
    Edge *edge_rchl;
public:
    Node(int branch);
    void reset();
    int get_branch() { return this->branch; }
    void copy_ptrs(const Node *input);
    Edge *get_ptr(int type);
    void set_ptr(int type, Edge *edge);
    void update_edge(int type);
public:
    static int eval_relation(int branch_from, int branch_to);
};

// decoding tree
class Tree {
private:
    int n_level;
    int edge_num;
    Edge **edges;
    int node_num;
    Node **nodes;
public:
    Tree(int n_level, NDimShape *shape);
    ~Tree();
    void reset();
    void set_root(const double *probs);
    Edge *get_edge(int branch) { return this->edges[branch]; }
    Node *get_node(int branch) { return this->nodes[branch]; }
    int edge_size(int depth) { return 1 << (this->n_level - 1 - depth); }
    void print();
public:
    static int get_depth(int branch) { return std::ceil(std::log2(branch + 2)) - 1; }
};

// walker
class Walker {
private:
    int code_len;
    int code_lvl;
    Tree *tree;
    Edge **edge_buffers;
    Node *node_buffer;
public:
    Node *head;
public:
    Walker(int code_len, NDimShape *shape);
    ~Walker();
    void reset();
    void set_priors(const double *priors);
    void lazy_step(int branch_to);
    void flush_to(int branch_to);
    Edge *get_edge_real(int branch) { return this->tree->get_edge(branch); }
    Edge *get_edge_buffer(int branch) { return this->edge_buffers[Tree::get_depth(branch)]; }
    Node *get_node_buffer() { return node_buffer; }
    void print_tree();
public:
    static int *get_path(int branch_from, int branch_to);
    static void walk_to(Walker **walker, int n_walker, int branch_to);
};
