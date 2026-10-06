# Installation

## Requirements

| Component | Requirements |
| --- | --- |
| Build | CMake ≥ 3.20, C99 compiler, zlib |
| CPU backend | OpenMP, LP64 BLAS/LAPACK |
| CUDA backend | CUDA Toolkit ≥ 12.4 with NVCC; compatible NVIDIA GPU to run |
| Python bindings | C++17 compiler, Python ≥ 3.8, NumPy ≥ 1.21, SciPy ≥ 1.8 |
| Multi-GPU (optional) | MPI, NCCL, C++17 compiler |

## Command-Line Executable

### Build from Source

Clone the repository and compile the project using CMake:

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
cmake -S . -B build
cmake --build build --clean-first
```

This will create the solver binary at `./build/pdhcg`.

`PDHCG_DEVICES=AUTO` (default) builds the backends whose dependencies are available.
CMake prints the selected devices at the end of configuration.
To select explicitly, use `-DPDHCG_DEVICES=CPU`, `CUDA`, or `"CPU;CUDA"`;
missing dependencies then cause an error.

The default backend is CUDA when available under AUTO, or the first backend in an
explicit list; override it with `PDHCG_DEFAULT_DEVICE`. Its executable is `pdhcg`;
additional backends use suffixes such as `pdhcg_cpu`.
Run `./build/pdhcg --list-devices` to display the compiled devices.

### Build the CPU Backend

```bash
cmake -S . -B build-cpu -DPDHCG_DEVICES=CPU
cmake --build build-cpu
./build-cpu/pdhcg problem.qps results/ --threads 8
```

`--threads` sets the CPU thread limit; `0` (default) inherits OpenMP settings.
PDHCG automatically limits supported OpenBLAS/MKL libraries to one BLAS thread.
OpenBLAS uses a shared setting: applications must synchronize other calls and
thread-setting changes on the same library instance with PDHCG solves.
For other threaded BLAS libraries, configure one BLAS thread manually;
CMake reports the detected support.

When enabling PreFOS with CPU, also set `-DPDHCG_PREFOS_ENABLE_CUDA=OFF`.

### Specifying CUDA Compiler

If your system has multiple CUDA versions or the default nvcc is outdated, explicitly specify the path to your CUDA compiler:

```bash
# Replace '/your/path/to/nvcc' with the actual path, e.g., /usr/local/cuda-12.6/bin/nvcc
CUDACXX=/your/path/to/nvcc cmake -S . -B build
cmake --build build --clean-first
```

### Build with Multi-GPU Support

Multi-GPU solving requires the CUDA backend, MPI and NCCL:

```bash
cmake -S . -B build -DPDHCG_COMPILE_DISTRIBUTED=ON
cmake --build build --clean-first
```

When enabled, the solver binary automatically detects whether it is launched with multiple MPI ranks and switches to the distributed solver.

## Python Package

!!! note "Multi-GPU support"
    The Python interface currently supports single-GPU solving only. For multi-GPU distributed solving, build the native executable with `-DPDHCG_COMPILE_DISTRIBUTED=ON` and launch it via `mpirun`.

### From PyPI (Recommended)

```bash
pip install pdhcg
```

### From Source

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
pip install .
```

Source builds use the same AUTO selection. To select backends explicitly:

```bash
pip install . -Ccmake.define.PDHCG_DEVICES=CPU
# Or include both:
pip install . '-Ccmake.define.PDHCG_DEVICES=CPU;CUDA'
```

Use `model.optimize(device="cpu")` or `model.optimize(device="cuda")` to select
among compiled backends for each solve; omitting `device` uses the build default.
`pdhcg.built_devices()` returns the compiled devices, and `pdhcg.print_devices()`
displays them. With the CVXPY adapter, use `problem.solve(solver="PDHCG", device="cpu")`.

### Development Installation

For development with editable install:

```bash
git clone https://github.com/Lhongpei/PDHCG.git
cd PDHCG
pip install -e ".[test]"
```

## Verification

### Native Executable

```bash
./build/pdhcg --help
```

### Python Package

```python
import pdhcg
print(pdhcg.__version__)
```

## Pre-commit Hooks (For Contributors)

To ensure code quality before committing, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This automatically formats Python (Ruff), C/C++ (clang-format), and checks spelling when you commit.
