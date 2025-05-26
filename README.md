<table>
  <tr>
    <td><img src="logo.png" alt="plcoding logo" width="60"/></td>
    <td><h1>Plcoding: A Python Library for Arikan's Polar Codes</h1></td>
  </tr>
</table>

This is an under development Python library (with C++ acceleration) for advanced research on Arikan's polar codes. It provides high-performance implementations of various methods for channel coding, source coding, and theoretical investigations, with a modular design aimed at researchers and developers.

## Online Tutorial

👉 [Plcoding用户手册](https://renzichang.github.io/plcoding/)

This project includes a Python library for researching polar codes, and is accompanied by complete teaching notes and implementation instructions.

## Features

The library is organized into three submodules:

- **`channel`**  
  Implements methods for polar codes in channel coding scenarios, including:
  - BEC construction
  - BPSK-based decoding using LLR
  - Min-sum approximation algorithms

- **`source`**  
  Provides tools for source compression using polar codes, including:
  - Arikan's linear compression scheme
  - A construction-free compression scheme proposed by the author
  - Lossy compression methods
  - Monotone chain polar codes for Slepian–Wolf coding

- **`research`**  
  Contains advanced experimental modules for theoretical exploration, such as:
  - Non-binary polar codes
  - Bayesian network-based polar decoding iterators

## Installation

This library uses `pybind11` to bind C++ acceleration modules.

### Requirements

- Python ≥ 3.8
- C++17 compiler (e.g., `g++`, `clang++`)
- `pybind11`
- `FFTW3`
- `NumPy`

### Install with pip (from source)

```bash
pip install .
```

### Example Usage

```python
import numpy as np
from plcoding.source import encode_cdf

bitstream = encode_cdf(cdf_tensor, symbol_tensor)
```
More detailed examples and Jupyter notebooks will be available in the tests/ directory.

## Development Progress

- Implemented:
  - C++ backend for Bayesian network-based Polar Iterator
  - Construction-free compression scheme for source coding

- In progress:
  - Generalized polar decoding frameworks
  - Tools for empirical performance evaluation
  - Visual analytics for decoding processes
  - Interfaces for ML-based code design and inference

### License

MIT License

## Author

This library is developed and maintained by **Zichang Ren**.

Feel free to reach out:

- 📧 Email: [rzc1937986979@163.com](mailto:rzc1937986979@163.com)
- 💬 QQ: 1937986979

Contributions, feedback, and collaborations are warmly welcome!
