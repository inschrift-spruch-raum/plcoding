#include "utils.h"


class PolarIterator {
private:
    int n, N, q;
    double* edges;
    bool* states;
    int *randmap, *lookups;
    double* tmp;
    FFTW3Wrapper* fftw;
public:
    PolarIterator(int code_len, int prob_base);
    ~PolarIterator();
    void set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> input);
    void reset();
    py::array_t<double> get_prob(int index);
    void set_value(int index, int value);
    py::array_t<int> transform_2x(py::array_t<int, py::array::c_style | py::array::forcecast> input);
    py::array_t<int> transform_2u(py::array_t<int, py::array::c_style | py::array::forcecast> input);
private:
    double* get_edge(int level, int edge_index, bool desired_state);
};

PolarIterator::PolarIterator(int code_len, int prob_base) {
    this->n = std::ceil(std::log2(code_len));
    this->N = (1 << this->n);
    this->q = prob_base;
    this->edges = new double[(this->n + 1) * this->N * this->q];
    this->states = new bool[2 * this->N - 1];
    this->randmap = new int[this->q];
    this->lookups = new int[this->q];
    gen_randmap(this->q, this->randmap);
    get_inverse(this->q, this->randmap, this->lookups);
    this->tmp = new double[this->q];
    this->fftw = new FFTW3Wrapper(this->q);
    this->reset();
}

PolarIterator::~PolarIterator() {
    delete[] this->edges;
    delete[] this->states;
    delete[] this->randmap;
    delete[] this->lookups;
    delete[] this->tmp;
    delete this->fftw;
}

void PolarIterator::set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info info = input.request();
    if (info.ndim != 2 || info.shape[0] != this->N || info.shape[1] != this->q)
        throw std::runtime_error("Priors shape must be (N, q)!");
    double* data_ptr = static_cast<double*>(info.ptr);
    std::copy(data_ptr, data_ptr + this->N * this->q, this->edges);
}

py::array_t<double> PolarIterator::get_prob(int index) {
    double* data = this->get_edge(this->n, index, false);
    py::array_t<double> result(this->q);
    py::buffer_info buf = result.request();
    double* res_pt = static_cast<double*>(buf.ptr);
    std::copy(data, data + this->q, res_pt);
    return result;
}

void PolarIterator::set_value(int index, int value) {
    if (value < 0 || value >= this->q)
        throw std::runtime_error("The value should be in Z_q!");
    // you cannot just change an edge without updating the Bayesian network
    double* edge = this->get_edge(this->n, index, false);
    for (int i = 0; i < this->q; ++i)
        edge[i] = 0;
    edge[value] = 1;
    // this edge is updated
    this->states[this->N - 1 + index] = true;
}

py::array_t<int> PolarIterator::transform_2x(py::array_t<int, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info info = input.request();
    int* data_ptr = static_cast<int*>(info.ptr);
    if (info.ndim != 1 || info.shape[0] != this->N)
        throw std::runtime_error("Invalid input size.");
    // initialization
    int* xs = new int[this->N];
    int* us = new int[this->N];
    std::copy(data_ptr, data_ptr + this->N, us);
    // recursive calculation
    int size = 2;
    while (size <= this->N) {
        for (int i = 0; i < this->N / size; ++i) {
            for (int j = 0; j < size / 2; ++j) {
                int index1 = i * size + j;
                int index2 = index1 + size / 2;
                xs[index1] = (us[index1] + us[index2]) % this->q;
                xs[index2] = this->randmap[us[index2]];
            }
        }
        data_ptr = xs; xs = us; us = data_ptr;
        size *= 2;
    }
    // ending process
    delete[] xs;
    auto output = py::array_t<int>({this->N}, {sizeof(int)}, us, py::capsule(us, [](void* p) { delete[] reinterpret_cast<int*>(p); }));
    return output;
}

py::array_t<int> PolarIterator::transform_2u(py::array_t<int, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info info = input.request();
    int* data_ptr = static_cast<int*>(info.ptr);
    if (info.ndim != 1 || info.shape[0] != this->N)
        throw std::runtime_error("Invalid input size.");
    // initialization
    int* xs = new int[this->N];
    int* us = new int[this->N];
    std::copy(data_ptr, data_ptr + this->N, xs);
    // recursive calculation
    int size = this->N;
    while (size > 1) {
        for (int i = 0; i < this->N / size; ++i) {
            for (int j = 0; j < size / 2; ++j) {
                int index1 = i * size + j;
                int index2 = index1 + size / 2;
                us[index2] = this->lookups[xs[index2]];
                us[index1] = (xs[index1] - us[index2] + this->q) % this->q;
            }
        }
        data_ptr = xs; xs = us; us = data_ptr;
        size /= 2;
    }
    // ending process
    delete[] us;
    auto output = py::array_t<int>({this->N}, {sizeof(int)}, xs, py::capsule(xs, [](void* p) { delete[] reinterpret_cast<int*>(p); }));
    return output;
}

void PolarIterator::reset() {
    for (int i = this->N * this->q; i < (this->n + 1) * this->N * this->q; ++i)
        this->edges[i] = 1.0 / this->q;
    this->states[0] = false;
    for (int i = 1; i < 2 * this->N - 1; ++i)
        this->states[i] = true;
}

double* PolarIterator::get_edge(int level, int edge_index, bool desired_state) {
    if (level < 0 || level > this->n)
        throw std::runtime_error("Invalid tree level!");
    if (edge_index < 0 || edge_index >= (1 << level))
        throw std::runtime_error("Invalid edge index!");
    const int edge_size = this->N / (1 << level);
    double* edge_now = this->edges + (level * this->N + edge_index * edge_size) * this->q;
    bool&  state_now = this->states[(1 << level) - 1 + edge_index];
    // retrieve from the previous calculation
    if (state_now == desired_state)
        return edge_now;
    // recalculation
    state_now = desired_state;
    if (desired_state) {
        double* x1 = edge_now;
        double* x2 = x1 + (edge_size / 2) * this->q;
        double* u1 = get_edge(level + 1, edge_index * 2, true);
        double* u2 = get_edge(level + 1, edge_index * 2 + 1, true);
        for (int i = 0; i < edge_size / 2; ++i) {
            // (x1, x2) <- (u1, u2)
            fftw->circonv(u1, u2, x1);
            for (int j = 0; j < this->q; ++j)
                x2[this->randmap[j]] = u2[j];
            normalize(this->q, x2);
            x1 += this->q; x2 += this->q;
            u1 += this->q; u2 += this->q;
        }
    } else {
        double* x1 = get_edge(level - 1, edge_index / 2, false);
        double* x2 = x1 + edge_size * this->q;
        if (edge_index % 2 == 0) {
            double* u1 = edge_now;
            double* u2 = get_edge(level, edge_index + 1, true);
            for (int i = 0; i < edge_size; ++i) {
                // (x1, x2) (u1) <- (u2)
                for (int j = 0; j < this->q; ++j)
                    this->tmp[(this->q - j) % this->q] = x2[this->randmap[j]] * u2[j];
                normalize(this->q, this->tmp);
                fftw->circonv(x1, this->tmp, u1);
                x1 += this->q; x2 += this->q;
                u1 += this->q; u2 += this->q;
            }
        } else {
            double* u1 = get_edge(level, edge_index - 1, true);
            double* u2 = edge_now;
            for (int i = 0; i < edge_size; ++i) {
                // (x1, x2) (u1) -> (u2)
                for (int j = 0; j < this->q; ++j)
                    this->tmp[j] = u1[(this->q - j) % this->q];
                fftw->circonv(x1, this->tmp, u2);
                for (int j = 0; j < this->q; ++j)
                    u2[j] *= x2[this->randmap[j]];
                normalize(this->q, u2);
                x1 += this->q; x2 += this->q;
                u1 += this->q; u2 += this->q;
            }
        }
    }
    return edge_now;
}

PYBIND11_MODULE(classics, m) {
    py::class_<PolarIterator>(m, "PolarIterator")
        .def(py::init<int, int>(), py::arg("code_len"), py::arg("prob_base"))
        .def("set_priors", &PolarIterator::set_priors)
        .def("reset", &PolarIterator::reset)
        .def("get_prob", &PolarIterator::get_prob)
        .def("set_value", &PolarIterator::set_value)
        .def("transform_2x", &PolarIterator::transform_2x)
        .def("transform_2u", &PolarIterator::transform_2u);
}
