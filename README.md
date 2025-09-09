<table>
  <tr>
    <td><img src="logo.png" alt="plcoding logo" width="60"/></td>
    <td><h1>Plcoding: A Python Library for the Research of Polar Codes</h1></td>
  </tr>
</table>

这是一个关于极化码（polar codes）的Python库。最初供我自己做科研使用，现在开源到github，希望能够帮助到做相关研究、或者单纯是对信息论与编码技术感兴趣的同学们。

在库中，提供了各种基于Numpy的信息论与编码理论基础工具箱、以及基于极化现象的Python模块，还基于这些模块复现了一些极化码方向的经典工作，如**信源极化压缩**、**SW-问题的单调链极化码**、**极化码列表译码算法**、**有损信源编码**等等。其主要目的是辅助科学研究和理论分析，并不打算实现工业级别的解决方案。如果你对极化码的实用化感兴趣，不妨以这个库为基础进行开发。相比Matlab一类的偏仿真模拟的工业语言，Python/C++这类通用编程语言具有更良好的设计范式，更适合方便快速地集成到其他系统中。

Plcoding库配套一册基于Jupyter Notebook的在线免费文档，其中给出了大量的**可执行代码**，并配有详实的文字说明。撰写该手册的最初目的是用于助教工作。其现版本可以视作一本极化码入门手册。其中第一章《极化码简介》在没有对理论原理进行深入介绍的情况下，直接给出了明确的构造、编码和译码算法，适合那些不希望深入了解极化原理、只希望实现一份简单但完备的极化码解决方案的同学们。

如果对你有帮助，请务必在Github [项目主页](https://github.com/RenZichang/plcoding) 的右上角，点击star支持本项目。**这次一定！**

## 入门教程

👉 [Plcoding用户手册](https://renzichang.github.io/plcoding/)

> 为实时获取手册的最新内容，可在阅读时使用组合按键 `Shift+F5` 强制刷新浏览器缓存。

**除了入门手册外，在本项目的./plcoding/tests/目录下，也给出了测试用例，可用于快速了解plcoding库的基本用法。**

## 细节说明

这个库包括如下几个子模块：

- **`plcoding`**  
  包括一些常用的基本函数，例如：
  - 熵的计算与逆运算
  - 极化码的递归编码
  - 比特反转置换（bit-reversal permutation）
  - 二进制擦除信道（binary erasure channel）构造法
  - 矩阵的Kronecker幂运算
  - ...

- **`plcoding.channel`**  
  用于支持初级的信道纠错极化码研究，包括：
  - 常用信道，如AWGN信道、比特翻转信道等
  - 常用调制算法，如BPSK、QAM，以及对应的软/硬判决算法
  - 基于SC译码的信道极化码
  - ...

- **`plcoding.source`**  
  用于支持初级的信源压缩极化码研究，包括：
  - 几种经典的无失真极化压缩技术
  - 适配实用压缩系统的极化熵编码器
  - ...

- **`plcoding.research`**  
  用于支持进阶的极化码研究，包括：
  - 密度演化（density evolution）构造法
  - 单调链（Monotone chain）极化码
  - 限失真信源编码
  - ...

- **`plcoding.cpp_core`**  
  - 高性能计算的核心部分，部分其他模块是对该核心的Python封装
  - 这部分源码是可见的，如果确定理解它们的工作方式，可以绕过封装直接使用

## 安装方法

这个库使用pybind11来绑定C++模块到Python，并且使用了FFTW3来加速卷积运算，因此环境需求为：

- C++（源码编译）
  - FFTW3

- Python3（运行环境）
  - pybind11
  - numpy

上述是plcoding库的基本运行环境，如果想要顺利执行手册中的代码，则可能还需要安装matplotlib、tqdm、jupyter、torch等常用Python库，这里不再赘述。

**从源码**：下载源码到本地、配好上述运行环境之后，进入CMakeLists.txt同级目录，执行pip的本地安装指令：

```bash
pip install .
```

**从PyPI**：（推荐）

```bash
pip install plcoding
```

## 开发进度

- 已实现：
  - C++核心功能模块，包括快速极化模块、通用的SC译码器模块（2025.05.25）
  - plcoding模块下的部分基本函数（2025.05.27）
  - 配套手册的基本内容已完成（2025.06.03）
  - plcoding.channel模块下的channel类和modulator类（2025.06.04）
  - plcoding.channel模块下的coder类（2025.07.01）
  - plcoding.source模块、以及从整数到比特流的映射函数（2025.07.15）
  - plcoding.research.monotone模块，可用于研究单调链极化码（2025.08.24）
  - 发不到pypi，包名为plcoding（2025.09.07）

- 待实现
  - 各类构造法（启发式、密度演化等）
  - 各类快速译码算法（对数似然比、最小和近似）
  - 各类极化核矩阵的编译码
  - ...

### License

MIT License

## 开发者

欢迎在github提交issue，或者私下交流：

- 📧 Email: [rzc1937986979@163.com](mailto:rzc1937986979@163.com)
- 💬 QQ: 1937986979
