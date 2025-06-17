#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>


namespace py = pybind11;


// random seed-pair
const unsigned int randa = 1103515245;
const unsigned int randc = 12345;


// generate a random permutation for Z_q
inline void gen_randmap(int q, int* randmap) {
    for (int i = 0; i < q; ++i)
        randmap[i] = i;
    unsigned int seed = q;
    int temp;
    for (int i = q - 1; i > 0; i--) {
        seed = seed * randa + randc;
        int j = seed % (i + 1);
        temp = randmap[i];
        randmap[i] =randmap[j];
        randmap[j] = temp;
    }
}


// fast probability distribution polarization
py::tuple prob_polarize(py::array_t<double, py::array::c_style | py::array::forcecast> pmf, py::array_t<int, py::array::c_style | py::array::forcecast> sym) {
    // initialization
    auto pmf_buf = pmf.request();
    int L = pmf_buf.shape[0];
    int q = pmf_buf.shape[1];
    double* seqs  = new double[L * q];
    double* seq_  = new double[L * q];
    int* bols = new int[L];
    int* randmap = new int[q];
    gen_randmap(q, randmap);
    // initialization of fftw3
    int N = L / 2;
    int f = q / 2 + 1;
    fftw_complex* feqs = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * 2 * N * f);
    double scale = 1.0 / q;
    int n[] = {q};
    fftw_plan plan_fwd = fftw_plan_many_dft_r2c(
        1, n, 2 * N,
        seqs, NULL, 1, q,
        feqs, NULL, 1, f,
        FFTW_MEASURE
    );
    fftw_plan plan_bwd = fftw_plan_many_dft_c2r(
        1, n, N,
        feqs, NULL, 1, 2 * f,
        seq_, NULL, 1, 2 * q,
        FFTW_MEASURE
    );
    // recursive computation
    std::memcpy(seqs, pmf.data(), sizeof(double) * L * q);
    std::memcpy(bols, sym.data(), sizeof(int) * L);
    int group_size = L;
    while (group_size > 1) {
        // 1. pairwise-circonv
        fftw_execute(plan_fwd);
        for (int i = 0; i < N; i++) {
            fftw_complex* A = feqs + (2 * i + 0) * f;
            fftw_complex* B = feqs + (2 * i + 1) * f;
            for (int j = 0; j < f; j++) {
                double ar = A[j][0], ai = A[j][1];
                double br = B[j][0], bi = B[j][1];
                A[j][0] = ar * br - ai * bi;
                A[j][1] = ar * bi + ai * br;
            }
        }
        fftw_execute(plan_bwd);
        for (int i = 0; i < N; i++) {
            double* out = seq_ + (2 * i) * q;
            for (int j = 0; j < q; ++j) {
                out[j] *= scale;
            }
        }
        for (int i = 0; i < N; i++) {
            bols[2 * i] = (bols[2 * i] + bols[2 * i + 1]) % q;
        }
        // 2. pairwise-nrmcomb
        for (int i = 0; i < N; i++) {
            double* a = seqs + (2 * i + 0) * q;
            double* b = seqs + (2 * i + 1) * q;
            double* c = seq_ + (2 * i + 1) * q;
            for (int j = 0; j < q; j++) {
                int k = (bols[2 * i] - j + q) % q;
                c[randmap[j]] = a[k] * b[j];
            }
            double tau = 0.0;
            for (int j = 0; j < q; j++) tau += c[j];
            for (int j = 0; j < q; j++) c[j] /= tau;
            bols[2 * i + 1] = randmap[bols[2 * i + 1]];
        }
        // 3. shuffle-permutation
        int half = group_size / 2;
        int* bols_tmp = new int[group_size];
        for (int base = 0; base < L; base += group_size) {
            for (int i = 0; i < half; i++) {
                std::memcpy(seqs + (base + i) * q, seq_ + (base + 2 * i) * q, sizeof(double) * q);
                bols_tmp[i] = bols[base + 2 * i];
                std::memcpy(seqs + (base + half + i) * q, seq_ + (base + 2 * i + 1) * q, sizeof(double) * q);
                bols_tmp[half + i] = bols[base + 2 * i + 1];
            }
            std::memcpy(bols + base, bols_tmp, sizeof(int) * group_size);
        }
        delete[] bols_tmp;
        group_size /= 2;
    }
    // end
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(feqs);
    delete[] seq_;
    delete[] randmap;
    auto seqs_ndarray = py::array_t<double>(
        {L, q},
        {sizeof(double) * q, sizeof(double)},
        seqs,
        py::capsule(seqs, [](void* p) { delete[] reinterpret_cast<double*>(p); })
    );
    auto bols_ndarray = py::array_t<int>(
        {L},
        {sizeof(int)},
        bols,
        py::capsule(bols, [](void* p) { delete[] reinterpret_cast<int*>(p); })
    );
    return py::make_tuple(seqs_ndarray, bols_ndarray);
}

PYBIND11_MODULE(functions, m) {
    m.def("prob_polarize", &prob_polarize, "Polarize given symbols along with their distributions.");
}
