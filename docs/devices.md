# Devices

PDHCG uses the same device selection in Python, C, and the command line.
CPU and CUDA implementations share the solver algorithm.

## Select a device

Choose `"cpu"` or `"cuda"` for each solve:

| Interface | Selection |
| --- | --- |
| Python | `model.optimize(device="cpu")` |
| CVXPY | `problem.solve(solver="PDHCG", device="cpu")` |
| C | Set `params.device = "cpu"` before `solve_qp_problem(prob, &params)` |
| Command line | `./build/pdhcg problem.qps output --device cpu` |

Omitting the option, using Python `None` or C `NULL`, or selecting `"auto"`
uses the build default: **CUDA before CPU**, unless overridden with
`PDHCG_DEFAULT_DEVICE`. An explicit choice does not change the build default.
Only compiled devices can be selected. If CUDA cannot initialize, the solver
reports an error; it does not fall back to CPU.

## List compiled devices

| Interface | Query |
| --- | --- |
| Python | `pdhcg.built_devices()` returns the names; `pdhcg.print_devices()` prints the names and default |
| C | `pdhcg_get_built_devices(&count)` returns the names; `pdhcg_get_default_device()` returns the default |
| Command line | `./build/pdhcg --list-devices` |

These queries report build information without initializing a device. For the
C query, declare `size_t count`; the returned array and names have static lifetime.

## Build backends

Install the [requirements](installation.md#requirements) for the desired backends.
`PDHCG_DEVICES` controls what is compiled:

| Value | Backends |
| --- | --- |
| `AUTO` (default) | All backends whose build dependencies are available |
| `CPU` | CPU only |
| `CUDA` | CUDA only |
| `CPU;CUDA` | Both |

An explicit selection fails if its dependencies are missing. CMake prints the
compiled backends and default at the end of configuration.

For example, build both backends:

```bash
cmake -S . -B build -DPDHCG_DEVICES="CPU;CUDA"
cmake --build build
```

For a Python source build, pass the same option through pip:

```bash
pip install . '-Ccmake.define.PDHCG_DEVICES=CPU;CUDA'
```

Use `-DPDHCG_DEFAULT_DEVICE=CPU` (or
`-Ccmake.define.PDHCG_DEFAULT_DEVICE=CPU` with pip) to override the default.
Builds containing CUDA require its runtime libraries even for CPU solves.
When enabling PreFOS in a build containing CPU, also set
`-DPDHCG_PREFOS_ENABLE_CUDA=OFF`.

## CPU threads

| Interface | Limit to eight threads |
| --- | --- |
| Python | `model.setParam("Threads", 8)` |
| CVXPY | Pass `Threads=8` to `problem.solve(...)` |
| C | Set `params.num_threads = 8` |
| Command line | Add `--threads 8` |

The default `0` inherits the OpenMP configuration, including `OMP_NUM_THREADS`.
A positive value limits the OpenMP team for each CPU solve; CUDA ignores it.
PDHCG limits supported OpenBLAS/MKL libraries to one BLAS thread. For other
threaded BLAS libraries, configure one thread separately. Applications sharing
OpenBLAS must coordinate unrelated calls and changes to its shared thread setting.

## Distributed solving

Multi-GPU execution uses the native C API or CLI with MPI and NCCL; Python
supports CPU and single-GPU solves. Enable distributed support and launch one
MPI rank per GPU:

```bash
cmake -S . -B build -DPDHCG_DEVICES=CUDA -DPDHCG_COMPILE_DISTRIBUTED=ON
cmake --build build
mkdir -p output
mpirun -n 4 ./build/pdhcg problem.mps output --device cuda
```

The executable selects the distributed solver when launched with multiple MPI
ranks. C applications use
[`solve_qp_problem_distributed`](c/functions.md#solve_qp_problem_distributed).

| CLI option | Meaning | Default |
| --- | --- | --- |
| `--grid_size r,c` | Process grid; `r*c` must equal the number of ranks | Automatic |
| `--partition_method` | `uniform` or `nnz` | `nnz` |
| `--permute_method` | `none`, `random`, or `block` | `block` |
| `--permute_block_size` | Block permutation size | `256` |

For example, add `--grid_size 2,2` to the four-rank command. Each PSD block stays
on one GPU; partitioning does not split its `svec` coordinates.
