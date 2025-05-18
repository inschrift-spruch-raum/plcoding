from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "plcoding.source_core",
        ["plcoding/source_core.cpp"],
        libraries=["fftw3", "fftw3_threads"],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-ffast-math"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="plcoding",
    version="0.1.0",
    author="Zichang Ren",
    author_email="vutoc_rcz@163.com",
    description="A Python library for polar codes using Python/C++/PyBind11",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithub/plcoding",  # 可选
    packages=["plcoding"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
