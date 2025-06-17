#include "utils.h"


// monotone cancellation decoder
class MCPolarIterator {
private:
    int code_len;
    int code_lvl;
    int *path;
    JointProb *jprob;
    int jpsize;
    Walker *walker;
    double *tmp;
public:
    MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> shape);
    ~MCPolarIterator();
    void reset() { this->walker->reset(); }
    void set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors);
    py::array_t<double> get_prob(int var, int index);
    void set_prob(int var, int index, int value);
private:
    void get_path(int branch_from, int branch_twrd);
};

MCPolarIterator::MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> shape) {
    this->code_lvl = std::ceil(std::log2(code_len));
    this->code_len = (1 << this->code_lvl);
    this->path = new int[this->code_lvl * 2];
    // get the shape of joint probability from python
    this->jprob = new JointProb(shape.data(), shape.size());
    this->jpsize = this->jprob->get_size();
    this->walker = new Walker(code_len, this->jprob);
    // temp variable
    this->tmp = new double[this->jpsize];
}

MCPolarIterator::~MCPolarIterator() {
    delete[] this->path;
    delete this->jprob;
    delete this->walker;
    delete[] this->tmp;
}

void MCPolarIterator::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors) {
    if (priors.ndim() != 2 || priors.shape()[0] != this->code_len || priors.shape()[1] != this->jpsize)
        throw std::runtime_error("Priors shape must be (N, q)!");
    this->walker->set_priors(priors.data());
}

py::array_t<double> MCPolarIterator::get_prob(int var, int index) {
    // the in-place computation is performed until just before the last edge
    int branch_dest = index + this->code_len - 1;
    this->get_path(this->walker->get_branch(), branch_dest);
    for (int i = 0; i < this->code_lvl * 2 - 1; ++i) {
        int branch_next = this->path[i];
        if (branch_next != -1) {
            this->walker->step(branch_next, this->walker->get_edge(branch_next));
        }
    }
    // retain the history information
    double *data_prev = this->walker->get_next(branch_dest)->data;
    for (int i = 0; i < this->jpsize; ++i) {
        this->tmp[i] = (data_prev[i] == 0) ? 0 : 1;
    }
    Edge *edge_dest = this->walker->get_edge(branch_dest);
    this->walker->step(branch_dest, edge_dest);
    this->jprob->nrmcomb(edge_dest->data, this->tmp, edge_dest->data);
    // return the marginal probability distribution
    int array_len = this->jprob->get_shape(var);
    py::array_t<double> result(array_len);
    double *result_data = result.mutable_data();
    for (int i = 0; i < array_len; ++i) {
        result_data[i] = 0;
    }
    for (int i = 0; i < this->jpsize; ++i) {
        int value = this->jprob->from_linear(i)[var];
        result_data[value] += edge_dest->data[i];
    }
    return result;
}

void MCPolarIterator::set_prob(int var, int index, int value) {
    if (this->walker->get_branch() != index - 1 + this->code_len) {
        throw std::runtime_error("Invalid index, you can only operate on the current decoding position");
    }
    double *data_ptr = this->walker->get_edge(this->walker->get_branch())->data;
    for (int i = 0; i < this->jpsize; ++i) {
        bool flag = (this->jprob->from_linear(i)[var] == value) && (data_ptr[i] != 0);
        data_ptr[i] = flag ? 1 : 0;
    }
}

// compute the path between any two edges on the decoding tree
void MCPolarIterator::get_path(int branch_from, int branch_twrd) {
    int up = 0, dn = 2 * this->code_lvl - 1;
    // two branches are not necessarily on the same level
    while (branch_from != branch_twrd) {
        while (branch_from > branch_twrd) {
            this->path[up++] = branch_from;
            branch_from = (branch_from - 1) / 2;
        }
        while (branch_from < branch_twrd) {
            this->path[dn--] = branch_twrd;
            branch_twrd = (branch_twrd - 1) / 2;
        }
    }
    // their common root node is not saved
    for (int i = up; i <= dn; ++i) {
        this->path[i] = -1;
    }
}


PYBIND11_MODULE(monotone, m) {
    py::class_<MCPolarIterator>(m, "MCPolarIterator")
        .def(py::init<int, py::array_t<int, py::array::c_style | py::array::forcecast>>(), py::arg("code_len"), py::arg("bases"))
        .def("set_priors", &MCPolarIterator::set_priors)
        .def("reset", &MCPolarIterator::reset)
        .def("get_prob", &MCPolarIterator::get_prob)
        .def("set_prob", &MCPolarIterator::set_prob);
}
