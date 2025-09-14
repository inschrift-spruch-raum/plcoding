#include "utils.h"
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace py = pybind11;

class PolarIterator {
private:
    std::size_t n, N, q;
    double* edges;
    bool* states;
    std::vector<int> randmap;
    int *lookups;
    double* tmp;
    FFTW3Wrapper fftw;
public:
    PolarIterator(std::size_t code_len, std::size_t prob_base);
    ~PolarIterator();
    void set_priors(const py::array_t<double, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input);
    void reset();
    py::array_t<double> get_prob(std::size_t index);
    void set_value(std::size_t index, std::size_t value);
    py::array_t<int> transform_2x(const py::array_t<int, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input);
    py::array_t<int> transform_2u(const py::array_t<int, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input);
private:
    double* get_edge(std::size_t level, std::size_t edge_index, bool desired_state);
};

PolarIterator::PolarIterator(std::size_t code_len, std::size_t prob_base):
    n(std::ceil(std::log2(code_len))),
    N(1U << this->n),
    q(prob_base),
    edges(new double[(this->n + 1) * this->N * this->q]),
    states(new bool[2 * this->N - 1]),
    randmap(this->q),
    lookups(new int[this->q]),
    tmp(new double[this->q]),
    fftw(this->q) {
    gen_randmap(this->q, this->randmap.data());
    get_inverse(this->q, this->randmap.data(), this->lookups);

    this->reset();
}

PolarIterator::~PolarIterator() {
    delete[] this->edges;
    delete[] this->states;
    delete[] this->lookups;
    delete[] this->tmp;
}

void PolarIterator::set_priors(const py::array_t<double, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input) {
    py::buffer_info info = input.request();
    if (info.ndim != 2 || static_cast<std::size_t>(info.shape[0]) != this->N || static_cast<std::size_t>(info.shape[1]) != this->q) {
        throw std::runtime_error("Priors shape must be (N, q)!");
    }
    auto* data_ptr = static_cast<double*>(info.ptr);
    std::copy_n(data_ptr, this->N * this->q, this->edges);
}

py::array_t<double> PolarIterator::get_prob(std::size_t index) {
    double* data = this->get_edge(this->n, index, false);
    py::array_t<double> result(static_cast<Py_ssize_t>(this->q));
    py::buffer_info buf = result.request();
    auto* res_pt = static_cast<double*>(buf.ptr);
    std::copy_n(data, this->q, res_pt);
    return result;
}

void PolarIterator::set_value(std::size_t index, std::size_t value) {
    if (value < 0 || value >= this->q) {
        throw std::runtime_error("The value should be in Z_q!");
    }
    // you cannot just change an edge without updating the Bayesian network
    double* edge = this->get_edge(this->n, index, false);
    for (std::size_t i = 0; i < this->q; ++i) {
        edge[i] = 0;
    }
    edge[value] = 1;
    // this edge is updated
    this->states[this->N - 1 + index] = true;
}

py::array_t<int> PolarIterator::transform_2x(const py::array_t<int, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input) {
    py::buffer_info info = input.request();
    if (info.ndim != 1 || static_cast<std::size_t>(info.shape[0]) != this->N) {
        throw std::runtime_error("Invalid input size.");
    }
    // initialization
    std::vector<int> xs(this->N);
    std::vector<int> us(this->N);
    std::copy_n(static_cast<int*>(info.ptr), this->N, us.data());
    // recursive calculation
    std::size_t size = 2;
    while (size <= this->N) {
        for (std::size_t i = 0; i < this->N / size; ++i) {
            for (std::size_t j = 0; j < size / 2; ++j) {
                std::size_t index1 = i * size + j;
                std::size_t index2 = index1 + size / 2;
                xs[index1] = static_cast<int>((us[index1] + us[index2]) % this->q);
                xs[index2] = this->randmap[us[index2]];
            }
        }
        std::swap(xs, us);
        size *= 2;
    }
    // ending process
    auto output = py::array_t<int>(static_cast<Py_ssize_t>(this->N));
    py::buffer_info buf = output.request();
    std::ranges::move(us, static_cast<int*>(buf.ptr));
    return output;
}

py::array_t<int> PolarIterator::transform_2u(const py::array_t<int, static_cast<std::size_t>(py::array::c_style) | static_cast<std::size_t>(py::array::forcecast)>& input) {
    py::buffer_info info = input.request();
    int* data_ptr = static_cast<int*>(info.ptr);
    if (info.ndim != 1 || static_cast<std::size_t>(info.shape[0]) != this->N) {
        throw std::runtime_error("Invalid input size.");
    }
    // initialization
    std::vector<int> xs(this->N);
    std::vector<int> us(this->N);
    std::copy_n(data_ptr, this->N, xs.data());
    // recursive calculation
    std::size_t size = this->N;
    while (size > 1) {
        for (std::size_t i = 0; i < this->N / size; ++i) {
            for (std::size_t j = 0; j < size / 2; ++j) {
                std::size_t index1 = i * size + j;
                std::size_t index2 = index1 + size / 2;
                us[index2] = this->lookups[xs[index2]];
                us[index1] = static_cast<int>((xs[index1] - us[index2] + this->q) % this->q);
            }
        }
        std::swap(xs, us);
        size /= 2;
    }
    // ending process
    auto output = py::array_t<int>(static_cast<Py_ssize_t>(this->N));
    py::buffer_info buf = output.request();
    std::ranges::move(xs, static_cast<int*>(buf.ptr));
    return output;
}

void PolarIterator::reset() {
    for (std::size_t i = this->N * this->q; i < (this->n + 1) * this->N * this->q; ++i) {
        this->edges[i] = 1.0 / this->q;
    }
    this->states[0] = false;
    for (std::size_t i = 1; i < 2 * this->N - 1; ++i) {
        this->states[i] = true;
    }
}

double* PolarIterator::get_edge(std::size_t level, std::size_t edge_index, bool desired_state) {
    if (level < 0 || level > this->n) {
        throw std::runtime_error("Invalid tree level!");
    }
    if (edge_index < 0 || edge_index >= (1U << level)) {
        throw std::runtime_error("Invalid edge index!");
    }
    const std::size_t edge_size = this->N / (1U << level);
    double* edge_now = this->edges + static_cast<ptrdiff_t>((level * this->N + edge_index * edge_size) * this->q);
    bool&  state_now = this->states[(1U << level) - 1 + edge_index];
    // retrieve from the previous calculation
    if (state_now == desired_state) {
        return edge_now;
    }
    // recalculation
    state_now = desired_state;
    if (desired_state) {
        double* x1 = edge_now;
        double* x2 = x1 + static_cast<ptrdiff_t>((edge_size / 2) * this->q);
        double* u1 = get_edge(level + 1, edge_index * 2, true);
        double* u2 = get_edge(level + 1, edge_index * 2 + 1, true);
        for (std::size_t i = 0; i < edge_size / 2; ++i) {
            // (x1, x2) <- (u1, u2)
            fftw.circonv(u1, u2, std::span<double>(x1, this->q));
            for (std::size_t j = 0; j < this->q; ++j) {
                x2[this->randmap[j]] = u2[j];
            }
            normalize(this->q, x2);
            x1 += this->q; x2 += this->q;
            u1 += this->q; u2 += this->q;
        }
    } else {
        double* x1 = get_edge(level - 1, edge_index / 2, false);
        double* x2 = x1 + static_cast<ptrdiff_t>(edge_size * this->q);
        if (edge_index % 2 == 0) {
            double* u1 = edge_now;
            double* u2 = get_edge(level, edge_index + 1, true);
            for (std::size_t i = 0; i < edge_size; ++i) {
                // (x1, x2) (u1) <- (u2)
                for (std::size_t j = 0; j < this->q; ++j) {
                    this->tmp[(this->q - j) % this->q] = x2[this->randmap[j]] * u2[j];
                }
                normalize(this->q, this->tmp);
                fftw.circonv(x1, this->tmp, std::span<double>(u1, this->q));
                x1 += this->q; x2 += this->q;
                u1 += this->q; u2 += this->q;
            }
        } else {
            double* u1 = get_edge(level, edge_index - 1, true);
            double* u2 = edge_now;
            for (std::size_t i = 0; i < edge_size; ++i) {
                // (x1, x2) (u1) -> (u2)
                for (std::size_t j = 0; j < this->q; ++j) {
                    this->tmp[j] = u1[(this->q - j) % this->q];
                }
                fftw.circonv(x1, this->tmp, std::span<double>(u2, this->q));
                for (std::size_t j = 0; j < this->q; ++j) {
                    u2[j] *= x2[this->randmap[j]];
                }
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
