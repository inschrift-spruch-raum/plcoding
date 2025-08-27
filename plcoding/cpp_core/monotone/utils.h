#pragma once
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>


enum StepType {to_parent, to_left, to_right, invalid};
class ND_Shape;
class Edge;
class Vertex;


inline StepType get_type(int from, int to) {
    if (from < 0 || to < 0 || from == to) return invalid;
    if (from != 0 && to == (from - 1) / 2) return to_parent;
    if (to == 2 * from + 1) return to_left;
    if (to == 2 * from + 2) return to_right;
    return invalid;
}


class ND_Shape {
private:    
    int nvar;
    int *bases;
    int size;
    int **nd_indices;
    int *reverse_map;
    double *array1, *array2, *array_;
    fftw_complex *farray1, *farray2, *farray_;
    fftw_plan plan1, plan2, plan_;
public:
    ND_Shape(const int *bases, int nvar);
    ~ND_Shape();
    int get_size() const { return this->size; }
    void set_uniform(double *data) const { for (int i = 0; i < this->size; ++i) data[i] = 1.0 / this->size; }
    void set_partial(int var, int value, double *data) const;
    int to_linear(const int *index) const;
    void nrmcomb(const double *from1, const double *from2, double *to) const;
    void circonv(const double *from1, const double *from2, double *to) const;
    void reverse(const double *from, double *to) const { for (int i = 0; i < this->size; ++i) to[this->reverse_map[i]] = from[i]; }
    void copy(const double *from, double *to) const { for (int i = 0; i < this->size; ++i) to[i] = from[i]; }
};


class Edge {
public:
    int branch;
    int size;
    const ND_Shape *shape;
    Vertex *from;
    double **data;
public:
    Edge(int branch, int size, const ND_Shape *shape);
    Edge(const Edge *edge);
    ~Edge();
    void clear_data();
    void copy_from(const Edge *edge);
    void set_probs(const double *probs);
    void combine_with(const Edge *edge);
};


class Vertex {
public:
    int branch;
    Edge *parent;
    Edge *left;
    Edge *right;
public:
    Vertex(int branch);
    Vertex(const Vertex *vertex);
    void copy_from(const Vertex *vertex);
    void calc_parent(Edge *result);
    void calc_left(Edge *result);
    void calc_right(Edge *result);
};
