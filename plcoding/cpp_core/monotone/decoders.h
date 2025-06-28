#include "utils.h"


// monotone cancellation decoder
class MCPolarIterator {
private:
    int code_len;
    int code_lvl;
    NDimShape *bases;
    Walker *walker;
public:
    MCPolarIterator(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> bases);
    ~MCPolarIterator();
    void reset() { this->walker->reset(); }
    void set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors);
    py::array_t<double> get_prob(int var, int index);
    void set_value(int var, int index, int value);
};


// monotone cancellation list decoder
class MCListDecoder {
private:
    int code_len;
    int code_lvl;
    NDimShape *bases;
    int list_size;
    int active_num;
    Walker **walkers;
    double *likelihoods;
public:
    MCListDecoder(int code_len, py::array_t<int, py::array::c_style | py::array::forcecast> bases, int list_size);
    ~MCListDecoder();
    void reset();
    void set_priors(py::array_t<double, py::array::c_style | py::array::forcecast> priors);
    void explore_at(int var, int index);
    void freeze_with(int var, int index, int value);
    py::array_t<int> get_results();
    py::array_t<double> get_likelihoods();
private:
    static double get_kth(const double *arr, int len, int k);
};