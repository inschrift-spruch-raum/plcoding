#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include <cmath>


typedef unsigned int uint32_t;
namespace py = pybind11;


const uint32_t randa = 1103515245;
const uint32_t randc = 12345;


void gen_randmap(int q, int* randmap);
void get_inverse(int q, int* randmap, int* lookups);
void normalize(int q, double* vector);


class FFTW3Wrapper {
private:
    int q;
    double *a_time, *b_time, *c_time;
    fftw_complex *a_freq, *b_freq, *c_freq;
    fftw_plan a_plan, b_plan, c_plan;
public:
    FFTW3Wrapper(int seq_len);
    ~FFTW3Wrapper();
    void circonv(const double* in1, const double* in2, double* out);
};
