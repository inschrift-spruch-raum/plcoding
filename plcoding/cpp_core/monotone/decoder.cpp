#include "decoder.h"


DecMemory::DecMemory(int code_len, const ND_Shape *shape) {
    this->code_len = code_len;
    this->shape = shape;
    // allocate memory
    int num_e = this->code_len * 2 - 1;
    int num_v = this->code_len - 1;
    this->edges = new Edge*[num_e];
    for (int i = 0; i < num_e; ++i) {
        int level = std::log2(i + 1);
        int size = this->code_len / (1 << level);
        this->edges[i] = new Edge(i, size, this->shape);
    }
    this->vertices = new Vertex*[num_v];
    for (int i = 0; i < num_v; ++i)
        this->vertices[i] = new Vertex(i);
    // initialize the structure
    this->edges[0]->from = nullptr;
    for (int i = 1; i < num_v; ++i)
        this->edges[i]->from = this->vertices[i];
    for (int i = num_v; i < num_e; ++i)
        this->edges[i]->from = nullptr;
    for (int i = 0; i < num_v; ++i) {
        this->vertices[i]->parent = this->edges[i];
        this->vertices[i]->left   = this->edges[i * 2 + 1];
        this->vertices[i]->right  = this->edges[i * 2 + 2];
    }
}

DecMemory::~DecMemory() {
    int num_e = this->code_len - 1;
    int num_v = this->code_len / 2 - 1;
    for (int i = 0; i < num_e; ++i)
        delete this->edges[i];
    delete[] this->edges;
    for (int i = 0; i < num_v; ++i)
        delete this->vertices[i];
    delete[] this->vertices;
}

void DecMemory::reset() {
    int num_e = this->code_len * 2 - 1;
    int num_v = this->code_len - 1;
    // initialized the corresponding relationship
    this->edges[0]->from = nullptr;
    for (int i = 1; i < num_v; ++i)
        this->edges[i]->from = this->vertices[i];
    for (int i = num_v; i < num_e; ++i)
        this->edges[i]->from = nullptr;
    for (int i = 0; i < num_v; ++i) {
        this->vertices[i]->parent = this->edges[i];
        this->vertices[i]->left   = this->edges[i * 2 + 1];
        this->vertices[i]->right  = this->edges[i * 2 + 2];
    }
    // clear all probability data except for the root edge
    for (int i = 1; i < num_e; ++i)
        this->edges[i]->clear_data();
}

void DecHead::lazy_step(StepType type, StepType type_inv, DecMemory *mem) {
    if (type == to_parent) {
        // obtain edges
        this->target_e = this->head->parent;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // obtain vertices
        this->target_v = this->target_e->from;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // pre-assign proper relationship
        if (type_inv == to_left) {
            this->buffer_v->left  = this->memory_e;
        } else {
            this->buffer_v->right = this->memory_e;
        }
        this->buffer_e->from = this->head;
        // compute the corresponding probability
        this->head->calc_parent(this->buffer_e);
    } else if (type == to_left) {
        // obtain edges
        this->target_e = this->head->left;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // obtain vertices
        this->target_v = this->target_e->from;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // pre-assign proper relationship
        this->buffer_v->parent = this->memory_e;
        this->buffer_e->from = this->head;
        this->head->calc_left(this->buffer_e);
    } else if (type == to_right) {
        // obtain edges
        this->target_e = this->head->right;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // obtain vertices
        this->target_v = this->target_e->from;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // pre-assign proper relationship and compute probabilities
        this->buffer_v->parent = this->memory_e;
        this->buffer_e->from = this->head;
        this->head->calc_right(this->buffer_e);
    } else {
        throw std::runtime_error("Invalid StepType in lazy_step()!");
    }
}

void DecHead::lazy_fork(DecHead *dhead, StepType type, const double *decision, DecMemory *mem) {
    if (type == to_left) {
        // obtain vertices
        this->target_v = dhead->head;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // obtain edges
        this->target_e = this->target_v->left;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // pre-assign proper relationship and compute probabilities
        this->buffer_v->left = this->memory_e;
        this->buffer_e->shape->copy(decision, this->buffer_e->data[0]);
    } else if (type == to_right) {
        // obtain vertices
        this->target_v = dhead->head;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // obtain edges
        this->target_e = this->target_v->right;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // pre-assign proper relationship and compute probabilities
        this->buffer_v->right = this->memory_e;
        this->buffer_e->shape->copy(decision, this->buffer_e->data[0]);
    }
}

void DecHead::lazy_leaf(StepType type, DecMemory *mem) {
    if (type == to_left) {
        // obtain vertices
        this->target_v = this->head;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // obtain edges
        this->target_e = this->target_v->left;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // pre-assign proper relationship
        this->buffer_v->left = this->memory_e;
        this->target_v->calc_left(this->buffer_e);
        this->buffer_e->combine_with(this->target_e);
    } else if (type == to_right) {
        // obtain vertices
        this->target_v = this->head;
        this->buffer_v = new Vertex(this->target_v);
        this->memory_v = mem->get_vertex(this->target_v);
        // obtain edges
        this->target_e = this->target_v->right;
        this->buffer_e = new Edge(this->target_e);
        this->memory_e = mem->get_edge(this->target_e);
        // pre-assign proper relationship
        this->buffer_v->right = this->memory_e;
        this->target_v->calc_right(this->buffer_e);
        this->buffer_e->combine_with(this->target_e);
    } else {
        throw std::runtime_error("Invalid StepType in calc_leaf()!");
    }
}

void DecHead::flush() {
    this->head = this->memory_v;
    this->memory_e->copy_from(this->buffer_e);
    this->memory_v->copy_from(this->buffer_v);
    delete this->buffer_e;
    delete this->buffer_v;
}

void DecHead::calc_root(double *mem) {
    Edge *temp = new Edge(this->head->parent);
    this->head->calc_parent(temp);
    for (int i = 0; i < temp->size; ++i)
        temp->shape->copy(temp->data[i], mem + i * temp->shape->get_size());
    delete temp;
}

ListIterator::ListIterator(int code_len, int list_len, input_int bases)
{
    int nvar = (int)bases.size();
    const int *bases_ptr = bases.data();
    this->code_len = code_len;
    this->list_len = list_len;
    this->shape = new ND_Shape(bases_ptr, nvar);
    this->memories = new DecMemory*[this->list_len];
    this->dheads = new DecHead*[this->list_len];
    for (int i = 0; i < this->list_len; ++i) {
        this->memories[i] = new DecMemory(code_len, this->shape);
        this->dheads[i] = new DecHead(this->memories[0]->get_vertex(0));
    }
    this->active_num = 1;
}

ListIterator::~ListIterator() {
    delete this->shape;
    for (int i = 0; i < this->list_len; ++i)
        delete this->memories[i];
    delete[] this->memories;
    for (int i = 0; i < this->active_num; ++i)
        delete this->dheads[i];
    delete[] this->dheads;
}
void ListIterator::reset() {
    this->memories[0]->reset();
    this->dheads[0]->head = this->memories[0]->get_vertex(0);
    this->active_num = 1;
}

void ListIterator::walk_to(int branch_to) {
    // generate the path
    int beg = this->dheads[0]->head->branch;
    int end = branch_to;
    int path_len = std::log2(this->code_len) * 2 - 1;
    int *path = new int[path_len];
    for (int i = 0; i < path_len; ++i)
        path[i] = -1;
    int i = 0, j = path_len - 1;
    while (beg != end) {
        while (beg > end) { path[i++] = beg; beg = (beg - 1) / 2; }
        while (beg < end) { path[j--] = end; end = (end - 1) / 2; }
    }
    path[j] = end;
    // walk along the path
    int now = -1;
    for (int i = 0; i < path_len; ++i) {
        if (path[i] == -1) continue;
        if (now != -1) {
            for (int j = 0; j < this->active_num; ++j) {
                StepType type = get_type(now, path[i]);
                StepType type_inv = get_type(path[i], now);
                this->dheads[j]->lazy_step(type, type_inv, this->memories[j]);
            }
            for (int j = 0; j < this->active_num; ++j)
                this->dheads[j]->flush();
        }
        now = path[i];
    }
    delete[] path;
}

py::array_t<double> ListIterator::get_probs(int var, int index) {
    this->walk_to((index + this->code_len - 2) / 2);
    // calculate the leaf edge 
    py::array_t<double> results({this->active_num, this->shape->get_size()});
    double *results_ptr = results.mutable_data();
    if (index % 2 == 0) {
        for (int i = 0; i < this->active_num; ++i)
            this->dheads[i]->lazy_leaf(to_left, this->memories[i]);
        for (int i = 0; i < this->active_num; ++i)
            this->dheads[i]->flush();
        for (int i = 0; i < this->active_num; ++i)
            this->shape->copy(this->dheads[i]->head->left->data[0], results_ptr + i * this->shape->get_size());
    } else {
        for (int i = 0; i < this->active_num; ++i)
            this->dheads[i]->lazy_leaf(to_right, this->memories[i]);
        for (int i = 0; i < this->active_num; ++i)
            this->dheads[i]->flush();
        for (int i = 0; i < this->active_num; ++i)
            this->shape->copy(this->dheads[i]->head->right->data[0], results_ptr + i * this->shape->get_size());
    }
    return results;
}

py::array_t<double> ListIterator::get_roots() {
    this->walk_to(0);
    // calculate the root edge 
    int step_size = this->shape->get_size() * this->code_len;
    py::array_t<double> results({this->active_num, step_size});
    double *results_ptr = results.mutable_data();
    for (int i = 0; i < this->active_num; ++i)
        this->dheads[i]->calc_root(results_ptr + i * step_size);
    return results;
}

void ListIterator::set_values(int var, int index, input_int values, input_int from) {
    int nfork = (int)values.size();
    const int* values_ptr = values.data();
    const int* from_ptr = from.data();
    double *temp = new double[this->shape->get_size()];
    for (int i = 0; i < nfork; ++i) {
        DecHead *dhead_from = this->dheads[from_ptr[i]];
        DecHead *dhead_to   = this->dheads[i];
        if (index % 2 == 0) {
            this->shape->copy(dhead_from->head->left->data[0], temp);
            this->shape->set_partial(var, values_ptr[i], temp);
            dhead_to->lazy_fork(dhead_from, to_left, temp, this->memories[i]);
        } else {
            this->shape->copy(dhead_from->head->right->data[0], temp);
            this->shape->set_partial(var, values_ptr[i], temp);
            dhead_to->lazy_fork(dhead_from, to_right, temp, this->memories[i]);
        }
    }
    /*int n = this->code_len;
    Edge **ptrs = new Edge*[n * nfork];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < nfork; ++j)
            ptrs[i * nfork + j] = this->memories[from.data()[j]]->get_edge(i);
    delete[] ptrs;*/
    for (int i = 0; i < nfork; ++i)
        this->dheads[i]->flush();
    this->active_num = nfork;
    delete[] temp;
}

PYBIND11_MODULE(monotone, m) {
    py::class_<ListIterator>(m, "ListIterator")
        .def(py::init<int, int, input_int>(), py::arg("code_len"), py::arg("list_len"), py::arg("bases"))
        .def("reset", &ListIterator::reset)
        .def("set_priors", &ListIterator::set_priors)
        .def("get_probs", &ListIterator::get_probs)
        .def("get_roots", &ListIterator::get_roots)
        .def("set_values", &ListIterator::set_values);
}
