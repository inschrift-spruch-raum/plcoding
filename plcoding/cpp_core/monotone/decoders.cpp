#include "decoders.h"


MCPolarIterator::MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> bases) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    // get the shape of joint probability from python
    this->bases = new NDimShape(bases.data(), bases.size());
    this->walker = new Walker(code_len, this->bases);
    this->reset();
}

MCPolarIterator::~MCPolarIterator() {
    delete this->bases;
    delete this->walker;
}

void MCPolarIterator::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->bases->get_size())
        throw std::runtime_error("Priors shape must be (N, q)!");
    this->walker->set_priors(priors.data());
}

py::array_t<double> MCPolarIterator::get_prob(int var, int index) {
    // walking along the path
    int branch_to = this->walker->root_of_leaf(index);
    Walker::walk_to(&(this->walker), 1, branch_to);
    // calculate the marginal distribution
    auto marginal = py::array_t<double>(this->bases->get_base(var));
    this->walker->calc_leaf(index)->calc_marginal(var, marginal.mutable_data());
    return marginal;
}

void MCPolarIterator::set_value(int var, int index, int value) {
    int type = (index % 2 == 0) ? 1 : 2;
    this->walker->head->get_ptr(type)->get_prob()->decision_upon(var, value);
}

MCListDecoder::MCListDecoder(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> bases, int list_size) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->bases = new NDimShape(bases.data(), bases.size());
    // list parameters
    this->list_size = list_size;
    this->walkers = new Walker*[list_size];
    for (int i = 0; i < list_size; ++i)
        this->walkers[i] = new Walker(code_len, this->bases);
    this->likelihoods = new double[list_size];
    this->reset();
}

MCListDecoder::~MCListDecoder() {
    delete this->bases;
    for (int i = 0; i < this->list_size; ++i)
        delete this->walkers[i];
    delete[] this->walkers;
    delete[] this->likelihoods;
}

void MCListDecoder::reset() {
    this->active_num = 1;
    this->walkers[0]->reset();
    this->likelihoods[0] = 0.0;
}

void MCListDecoder::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->bases->get_size())
        throw std::runtime_error("Priors shape must be (N, q)!");
    this->walkers[0]->set_priors(priors.data());
}

void MCListDecoder::explore_at(int var, int index) {
    int branch_to = this->walkers[0]->root_of_leaf(index);
    Walker::walk_to(this->walkers, this->active_num, branch_to);
    // explore all possible values at the specified var_index
    int n_row = this->active_num;
    int n_col = this->bases->get_base(var);
    int hold_num = n_row * n_col;
    double *ll_holds = new double[hold_num];
    for (int i = 0; i < n_row; ++i) {
        double *ll_hold = ll_holds + i * n_col;
        this->walkers[i]->calc_leaf(index)->calc_marginal(var, ll_hold);
        for (int j = 0; j < n_col; ++j) {
            ll_hold[j] = std::log(ll_hold[j]);
            ll_hold[j] += this->likelihoods[i];
        }
    }
    // process walkers with the likelihoods of the first-L large, one by one
    int edge_branch = this->walkers[0]->branch_of_leaf(index);
    int type = (index % 2 == 0) ? 1 : 2;
    this->active_num = (hold_num < this->list_size) ? hold_num : this->list_size;
    double threshold = MCListDecoder::get_kth(ll_holds, hold_num, this->active_num);
    for (int i = 0, j = 0; i < hold_num && j < this->active_num; ++i)  {
        if (ll_holds[i] >= threshold) {
            // get relation (row) and decision value (col)
            int from  = i / n_col;
            int value = i % n_col;
            // obtain walkers
            Walker *walker_from = this->walkers[from];
            Walker *walker_to   = this->walkers[j];
            // update the edge buffer of the new walker
            Edge *edge_from   = walker_from->head->get_ptr(type);
            Edge *edge_buffer = walker_to->get_edge_buffer(edge_branch);
            edge_buffer->copy_from(edge_from);
            edge_buffer->get_prob()->decision_upon(var, value);
            // update the node buffer of the new walker
            Node *node_buffer = walker_to->get_node_buffer();
            node_buffer->copy_from(walker_from->head);
            node_buffer->set_ptr(type, walker_to->get_edge_real(edge_branch));
            // store the cumulative likelihoods
            this->likelihoods[j] = ll_holds[i];
            j += 1;
        }
    }
    delete[] ll_holds;
    // copy the buffer data into real memory
    for (int i = 0; i < this->active_num; ++i) {
        Walker *walker = this->walkers[i];
        // copy edge
        Edge *edge_real = walker->get_edge_real(edge_branch);
        edge_real->copy_from(walker->get_edge_buffer(edge_branch));
        // copy node
        Node *node_real = walker->get_node_real(branch_to);
        node_real->copy_from(walker->get_node_buffer());
        walker->head = node_real;
    }
}

void MCListDecoder::freeze_with(int var, int index, int value) {
    int branch_to = this->walkers[0]->root_of_leaf(index);
    Walker::walk_to(this->walkers, this->active_num, branch_to);
    int edge_branch = this->walkers[0]->branch_of_leaf(index);
    int type = (index % 2 == 0) ? 1 : 2;
    for (int i = 0; i < this->active_num; ++i) {
        double p = this->walkers[i]->calc_leaf(index)->calc_marginal(var, value);
        this->likelihoods[i] += std::log(p);
    }
    for (int i = 0; i < this->active_num; ++i) {
        Walker *walker = this->walkers[i];
        Edge *edge_real = walker->get_edge_real(edge_branch);
        Edge *edge_buffer = walker->get_edge_buffer(edge_branch);
        edge_real->copy_from(edge_buffer);
        edge_real->get_prob()->decision_upon(var, value);
        walker->head->set_ptr(type, edge_real);
    }
}

// get the decoding results
py::array_t<int> MCListDecoder::get_results() {
    Walker::walk_to(this->walkers, this->active_num, 0);
    int result_len = this->code_len * this->bases->get_ndim();
    auto results = py::array_t<int>(this->active_num * result_len);
    int *ptr = results.mutable_data();
    for (int i = 0; i < this->active_num; ++i) {
        Walker *walker = this->walkers[i];
        Node *node_buffer = walker->get_node_buffer();
        Edge *edge_buffer = walker->get_edge_buffer(0);
        node_buffer->copy_from(walker->head);
        node_buffer->set_ptr(0, edge_buffer);
        node_buffer->update_edge(0);
        for (int j = 0; j < edge_buffer->get_size(); ++j) {
            edge_buffer->get_prob(j)->get_value(ptr);
            ptr += this->bases->get_ndim();
        }
    }
    return results;
}

py::array_t<double> MCListDecoder::get_likelihoods() {
    auto results = py::array_t<double>(this->active_num);
    double *ptr = results.mutable_data();
    for (int i = 0; i < this->active_num; ++i)
        ptr[i] = this->likelihoods[i];
    return results;
}

// get the k-th max element in the given array
double MCListDecoder::get_kth(const double *arr, int len, int k) {
    double *copy = new double[len];
    for (int i = 0; i < len; ++i)
        copy[i] = arr[i];
    std::nth_element(copy, copy + len - k, copy + len);
    double result = copy[len - k];
    delete[] copy;
    return result;
}

PYBIND11_MODULE(monotone, m) {
    py::class_<MCPolarIterator>(m, "MCPolarIterator")
        .def(py::init<int, py::array_t<int, py::array::c_style | py::array::forcecast>>(), py::arg("code_len"), py::arg("bases"))
        .def("reset", &MCPolarIterator::reset)
        .def("set_priors", &MCPolarIterator::set_priors)
        .def("get_prob", &MCPolarIterator::get_prob)
        .def("set_value", &MCPolarIterator::set_value);
    py::class_<MCListDecoder>(m, "MCListDecoder")
        .def(py::init<int, py::array_t<int, py::array::c_style | py::array::forcecast>, int>(), py::arg("code_len"), py::arg("bases"), py::arg("list_size"))
        .def("reset", &MCListDecoder::reset)
        .def("set_priors", &MCListDecoder::set_priors)
        .def("explore_at", &MCListDecoder::explore_at)
        .def("freeze_with", &MCListDecoder::freeze_with)
        .def("get_results", &MCListDecoder::get_results)
        .def("get_likelihoods", &MCListDecoder::get_likelihoods);
}
