# Installation

## Requirements

| Component | Requirements |
| --- | --- |
| Build | CMake ≥ 3.20, C99 compiler, zlib |
| CPU backend | OpenMP, LP64 BLAS/LAPACK |
| CUDA backend | CUDA Toolkit ≥ 12.4 with NVCC; compatible NVIDIA GPU to run |
| Python bindings | C++17 compiler, Python ≥ 3.8, NumPy ≥ 1.21, SciPy ≥ 1.8 |
| Multi-GPU (optional) | MPI, NCCL, C++17 compiler |

Backend build choices and runtime selection are documented together in
[Devices](devices.md).

## Python Package

```bash
pip install pdhcg
```

### CVXPY

For the optional CVXPY integration:

```bash
pip install "pdhcg[cvxpy]"
```

### From Source

```bash
git clone https://github.com/pdhcg/PDHCG.git
cd PDHCG
pip install .
```

Verify the installation:

```python
import pdhcg
print(pdhcg.__version__)
```

Continue with the [Python quick start](python/quickstart.md).

## Command-Line Executable

The same source build provides the native library and executable:

```bash
git clone https://github.com/pdhcg/PDHCG.git
cd PDHCG
cmake -S . -B build
cmake --build build
./build/pdhcg --help
```

See [Command line](cli.md) for file solving and [C API](c/overview.md) for linking
an application. For backend options and multi-GPU builds, see
[Devices](devices.md#build-backends).

## CUDA Compiler

To select a CUDA toolchain explicitly, set `CUDACXX` before configuring:

```bash
CUDACXX=/path/to/nvcc cmake -S . -B build
```

The same environment variable applies to `pip install .` source builds.

## Development Installation

From a checkout:

```bash
pip install -e ".[dev,test]"
pre-commit install
```

The hooks run Ruff, clang-format, and file consistency checks.
