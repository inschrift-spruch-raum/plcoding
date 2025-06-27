#include "decoders.h"


MCPolarIterator::MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> shape) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    // get the shape of joint probability from python
    this->shape = new NDimShape(shape.data(), shape.size());
    this->walker = new Walker(code_len, this->shape);
    this->reset();
}

MCPolarIterator::~MCPolarIterator() {
    delete this->shape;
    delete this->walker;
}

void MCPolarIterator::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->shape->get_size())
        throw std::runtime_error("Priors shape must be (N, q)!");
    this->walker->set_priors(priors.data());
}

py::array_t<double> MCPolarIterator::get_prob(int var, int index) {
    // walking along the path
    int branch_to = this->root_of_leaf(index);
    Walker::walk_to(&(this->walker), 1, branch_to);
    int type = (index % 2 == 0) ? 1 : 2;
    // calculate the joint distribution
    int edge_branch = this->branch_of_leaf(index);
    Edge *edge_buffer = this->walker->get_edge_buffer(edge_branch);
    Edge *edge_prev = this->walker->head->get_ptr(type);
    Node *node_buffer = this->walker->get_node_buffer();
    node_buffer->copy_ptrs(this->walker->head);
    node_buffer->set_ptr(type, edge_buffer);
    node_buffer->update_edge(type);
    edge_buffer->combine_with(edge_prev);
    // calculate the marginal distribution
    py::array_t<double> marginal(this->shape->get_shape()[var]);
    edge_buffer->get_prob()->calc_marginal(var, marginal.mutable_data());
    return marginal;
}

void MCPolarIterator::set_value(int var, int index, int value) {
    int type = (index % 2 == 0) ? 1 : 2;
    this->walker->head->get_ptr(type)->get_prob()->decision_upon(var, value);
}

PYBIND11_MODULE(monotone, m) {
    py::class_<MCPolarIterator>(m, "MCPolarIterator")
        .def(py::init<int, py::array_t<int, py::array::c_style | py::array::forcecast>>(), py::arg("code_len"), py::arg("bases"))
        .def("reset", &MCPolarIterator::reset)
        .def("set_priors", &MCPolarIterator::set_priors)
        .def("get_prob", &MCPolarIterator::get_prob)
        .def("set_value", &MCPolarIterator::set_value);
}
