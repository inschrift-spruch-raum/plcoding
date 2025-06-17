import plcoding
from plcoding.cpp_core.functions import prob_polarize
import numpy as np
from matplotlib import pyplot as plt

# 接收到的先验信息
p = 0.11
code_len = (1 << 10)
level = int(np.ceil(np.log2(code_len)))
src_prob = np.array([p, 1 - p])
priors = np.tile(src_prob, (code_len, 1))
# 获取构造信息
h_rank = np.argsort(np.argsort(plcoding.bec_channels(level, _log=True)))
# 仿真多次，获取每一次的无错误码率
code_rates = np.empty((1000,))
for i in range(len(code_rates)):
    symbols = np.random.choice(a=2, p=src_prob, size=(code_len,))
    pmf, sym = prob_polarize(priors, symbols)
    is_error = (np.argmax(pmf, axis=1) != sym)
    code_rates[i] = (np.min(h_rank[is_error]) + 1) / code_len
# 绘制BLER-Rate曲线
test_rates = np.linspace(0, 1, 21)
test_blers = np.empty_like(test_rates)
for i in range(len(test_rates)):
    test_blers[i] = (code_rates < test_rates[i]).mean()
plt.figure(figsize=(8, 4))
plt.semilogy(test_rates, test_blers, 'r-x')
ymin = 10 ** -np.log10(len(code_rates))
plt.vlines(plcoding.h2_of(p), ymin, 1)
plt.title(f"Code Length of {code_len}")
plt.xlabel("Code Rate")
plt.ylabel("Block Error Rate")
plt.axis([0, 1, ymin, 1.1])
plt.grid()
plt.show()
