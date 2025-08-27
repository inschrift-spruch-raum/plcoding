#include "utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using input_int = py::array_t<int, py::array::c_style | py::array::forcecast>;
using input_double = py::array_t<double, py::array::c_style | py::array::forcecast>;


class DecMemory {
private:
    int code_len;
    const ND_Shape *shape;
    Edge **edges;
    Vertex **vertices;
public:
    DecMemory(int code_len, const ND_Shape *shape);
    ~DecMemory();
    void reset();
    Edge *get_edge(int branch) { return this->edges[branch]; }
    Edge *get_edge(const Edge *like) { return this->edges[like->branch]; }
    Vertex *get_vertex(int branch) { return this->vertices[branch]; }
    Vertex *get_vertex(const Vertex *like) {return this->vertices[like->branch]; }
};


class DecHead {
private:
    Edge    *target_e,  *buffer_e,  *memory_e;
    Vertex  *target_v,  *buffer_v,  *memory_v;
public:
    Vertex *head;
public:
    DecHead(Vertex *vertex) { this->head = vertex; }
    void lazy_step(StepType type, StepType type_inv, DecMemory *mem);
    void lazy_fork(DecHead *dhead, StepType type, const double *decision, DecMemory *mem);
    void lazy_leaf(StepType type, DecMemory *mem);
    void flush();
    void calc_root(double *mem);
};


class ListIterator {
private:
    int code_len;
    int list_len;
    ND_Shape *shape;
    DecMemory **memories;
    DecHead **dheads;
    int active_num;
private:
    void walk_to(int branch_to);
public:
    ListIterator(int code_len, int list_len, input_int bases);
    ~ListIterator();
    void reset();
    void set_priors(input_double priors) { this->memories[0]->get_edge(0)->set_probs(priors.data()); }
    py::array_t<double> get_probs(int var, int index);
    py::array_t<double> get_roots();
    void set_values(int var, int index, input_int values, input_int from);
};
