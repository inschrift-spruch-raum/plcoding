#include "decoders.h"


MCPolarIterator::MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> shape) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    // get the shape of joint probability from python
    this->jprob = new JointProb(shape.data(), shape.size());
    this->walker = new DecodingWalker(code_len, this->jprob);
    this->reset();
}

MCPolarIterator::~MCPolarIterator() {
    delete this->jprob;
    delete this->walker;
}

void MCPolarIterator::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->jprob->get_size()) {
        throw std::runtime_error("Priors shape must be (N, q)!");
    }
    this->walker->set_priors(priors.data());
}

py::array_t<double> MCPolarIterator::get_prob(int var, int index) {
    this->walker->check_params(var, index);
    // walking along the path
    int branch_to = index + this->code_len - 1;
    DecodingWalker::walk_to(&(this->walker), 1, branch_to);
    // return the marginal distribution
    int array_len = this->jprob->get_shape(var);
    py::array_t<double> result(array_len);
    double *result_data = result.mutable_data();
    for (int i = 0; i < array_len; ++i) {
        result_data[i] = 0;
    }
    double *head_data = this->walker->get_head()->data;
    for (int i = 0; i < this->jprob->get_size(); ++i) {
        int value = this->jprob->from_linear(i)[var];
        result_data[value] += head_data[i];
    }
    return result;
}

void MCPolarIterator::set_value(int var, int index, int value) {
    this->walker->check_params(var, index, value);
    Edge *head = this->walker->get_head();
    head->partially_judge_from(head, var, value);
}

MCListDecoder::MCListDecoder(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> shape, int list_size) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->jprob = new JointProb(shape.data(), shape.size());
    // list parameters
    this->list_size = list_size;
    this->walkers = new DecodingWalker*[list_size];
    for (int i = 0; i < list_size; ++i) {
        this->walkers[i] = new DecodingWalker(code_len, this->jprob);
    }
    this->likelihoods = new double[list_size];
    this->reset();
}

MCListDecoder::~MCListDecoder() {
    for (int i = 0; i < this->list_size; ++i) {
        delete this->walkers[i];
    }
    delete[] this->walkers;
    delete[] this->likelihoods;
}

void MCListDecoder::reset() {
    this->active_num = 1;
    this->walkers[0]->reset();
    this->likelihoods[0] = 0.0;
}

void MCListDecoder::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->jprob->get_size()) {
        throw std::runtime_error("Priors shape must be (N, q)!");
    }
    this->walkers[0]->set_priors(priors.data());
}

void MCListDecoder::explore_at(int var, int index) {
    this->walkers[0]->check_params(var, index);
    // walking along the path
    int branch_to = index + this->code_len - 1;
    DecodingWalker::walk_to(this->walkers, this->active_num, branch_to);
    // explore all possible values at the specified var_index
    int n_row = this->active_num;
    int n_col = this->jprob->get_shape(var);
    int hold_num = n_row * n_col;
    double *ll_holds = new double[hold_num]();
    for (int i = 0; i < n_row; ++i) {
        // the increment of newly added likelihood
        int offset = i * n_col;
        double *head_data = this->walkers[i]->get_head()->data;
        for (int j = 0; j < this->jprob->get_size(); ++j) {
            int value = this->jprob->from_linear(j)[var];
            ll_holds[offset + value] += head_data[j];
        }
        // and historical cumulative likelihood
        for (int j = 0; j < n_col; ++j) {
            double &ll = ll_holds[offset + j];
            ll = std::log(ll);
            ll += this->likelihoods[i];
        }
    }
    // process walkers with the likelihoods of the first-L large, one by one
    this->active_num = 0;
    double threshold = MCListDecoder::topk_in(ll_holds, hold_num, this->list_size - 1);
    for (int i = 0; i < hold_num; ++i)  {
        if (ll_holds[i] >= threshold) {
            // get relation (row) and value (col)
            int row = i / n_col;
            int col = i % n_col;
            DecodingWalker *walker_from = this->walkers[row];
            Edge *edge_from = walker_from->get_head();
            DecodingWalker *walker_next = this->walkers[this->active_num];
            walker_next->branch_now = branch_to;
            Edge *buffer_next = walker_next->buffers[this->code_lvl];
            // generate a new edge and store it into the buffer
            buffer_next->partially_judge_from(edge_from, var, col);
            buffer_next->copy_ptrs(edge_from);
            this->likelihoods[this->active_num] = ll_holds[i];
            this->active_num += 1;
            if (this->active_num >= this->list_size) break;
        }
    }
    for (int i = 0; i < this->active_num; ++i) {
        this->walkers[i]->flush_buffer();
    }
    delete[] ll_holds;
}

void MCListDecoder::freeze_with(int var, int index, int value) {
    this->walkers[0]->check_params(var, index);
    // walking along the path
    int branch_to = index + this->code_len - 1;
    DecodingWalker::walk_to(this->walkers, this->active_num, branch_to);
    // set variable value for each head edge
    for (int i = 0; i < this->active_num; ++i) {
        Edge *head = this->walkers[i]->get_head();
        this->likelihoods[i] += head->partially_judge_from(head, var, value);
    }
}

// get the list of the decoding results
py::array_t<double> MCListDecoder::get_results() {
    // force all active walkers to execute to branch=2
    DecodingWalker::walk_to(this->walkers, this->active_num, 0);
    // allocate memory for python output
    int data_size = this->code_len * this->jprob->get_size();
    int totl_size = this->active_num * data_size;
    auto results = py::array_t<double>(totl_size);
    // force all active walkers to lazy_step to branch=0
    for (int i = 0; i < this->active_num; ++i) {
        this->walkers[i]->lazy_step(0);
        // copy the deterministic distributions of buffers into the pre-allocated space
        Edge *buffer = this->walkers[i]->buffers[0];
        double *offset_ptr = results.mutable_data() + i * data_size;
        for (int j = 0; j < data_size; ++j) {
            offset_ptr[j] = buffer->data[j];
        }
    }
    return results;
}

py::array_t<double> MCListDecoder::get_likelihoods() {
    auto results = py::array_t<double>(this->active_num);
    double *ptr = results.mutable_data();
    for (int i = 0; i < this->active_num; ++i) {
        ptr[i] = this->likelihoods[i];
    }
    return results;
}

double MCListDecoder::topk_in(const double *arr, int arr_len, int k)
{
    double *copy = new double[arr_len];
    for (int i = 0; i < arr_len; ++i) {
        copy[i] = arr[i];
    }
    double result;
    if (k < 0) {
        result = *std::max_element(copy, copy + arr_len);
    } else if (k >= arr_len) {
        result = *std::min_element(copy, copy + arr_len);
    } else {
        std::nth_element(copy, copy + (arr_len - 1 - k), copy + arr_len);
        result = copy[arr_len - 1 - k];
    }
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
