# SNK integral codegen

Integral kernel code generation for evaluating 3c1e nuclear attraction-type integrals, originally used for seminumerical exchange (SNK). Both VRR and HRR kernels can be generated with/without same-center optimizations for arbitrary multishells (sets of shells sharing the same center and primitive exponents but potentially having different L quantum numbers and contraction coefficients). 

The output kernels are written as templates for mixed precision evaluation (FP32 and FP64). These kernels are optimized for CPUs with SIMD vector instructions. No external C++ headers are required and the generated code is compatible with C++11 projects.

## Requirements

- Python (3.13.5 used originally)
- SymPy  (1.13.3 used originally), only for `verify.py`; generating code needs no packages

## Structure

The code is structured as an executable `main.py` along with several helper files. `dag.py` evaluates the recurrence relations of the 3c1e integrals as a graph of intermediates: it searches for the recurrence path (which center and cartesian direction to reduce for every intermediate) with the fewest floating point operations and schedules the intermediates for output. `codegen.py` writes this graph as C++ code. `codestrings.py` is a helper file for writing C++ code. `multishells.py` contains the multishell definitions, mapping strings to lists of L quantum numbers to be used as inputs to `codegen.generate_vrr()` and `codegen.generate_hrr()` (see below). `recurrences.py` unrolls the same recurrence relations symbolically with SymPy, without any optimization; `verify.py` uses it as the reference to check generated kernels.

## How the kernels are optimized

- **Recurrence path.** Almost every intermediate of the Obara-Saika VRR can be formed in several ways (which center and which cartesian direction to reduce), and so can the intermediates of the HRR. `dag.PathSearch` minimizes the number of operations of the whole kernel over these choices (structured starting points, local search, random restarts). The differences `[a|b]^(N) - [a|b]^(N+1)` that the VRR needs are formed either by a subtraction or by the recurrence itself, whichever is cheaper, and a factor `PG_i` that a step shares with one of its differences is taken out (same center kernels). Only the choice of path differs from the plain recurrences: VRR kernels never use HRR steps, so each kernel keeps its numerical character.
- **Cache blocking.** The arrays of the interface are `gb_size` elements apart. For a `gb_size` such as 128 this distance is a divisor of the 4096 bytes spanned by one way of the L1 cache, so the same gridpoint of all arrays maps to the same few cache sets, and a kernel that touches many arrays per gridpoint keeps evicting its own data. Kernels that touch enough bytes per gridpoint therefore process the gridpoints in blocks, contract the integrals of a block in a contiguous buffer on the stack and copy the buffer to the output arrays one array at a time. Whether this pays off depends on the precision, so some kernels contain both layouts and select one by `sizeof(T)`. The blocked HRR kernels keep their base integrals in such a buffer as well and do not use the `base_integrals` argument, which remains in the interface.
- **Order of evaluation.** Intermediates are emitted next to their first use and the outputs are ordered to keep few values alive, which reduces spills of vector registers to the stack. Kernels that work in place store all results at the end of the loop body, because the compiler has to reload the inputs after every store to an array of the interface.

The search effort and the layout are controlled by a few module variables at the top of `codegen.py`.

## Verification

```bash
python verify.py                      # a default selection of kernels
python verify.py vrr:F:D:0 hrr:G:G:1  # kind:MSA:MSB:same_center
```

`verify.py` generates kernels, executes the statements of the generated C++ in exact rational arithmetic for random inputs, and compares every integral with the SymPy recurrences of `recurrences.py`. Any difference is an error of the generated code rather than roundoff.

## Usage

Run with 

```bash
python main.py
```

By default, `main.py` runs the example configuration, generating a same center VRR (D|P) kernel and a different center HRR (G|F) kernel:

```python
# For a VRR kernel:
codegen.generate_vrr(MSA="D", MSB="P", same_center=True, dname="SNK_CODEGEN", verbose=True)

# For an HRR kernel:
codegen.generate_hrr(MSA="G", MSB="F", same_center=False, dname="SNK_CODEGEN", verbose=False)
```

where `MSA`, `MSB` are strings corresponding to a multishell in `MULTISHELLS` (`multishells.py`), `same_center` is a boolean controlling whether or not a same-center kernel is generated, `dname` is the output directory name (which will be created if it does not exist), and `verbose` is a boolean controlling the level of printing during execution.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this program in your research, please cite the following:

```bibtex
@misc{dicenso2026data,
      title={Data-Driven Optimization of Seminumerical Exchange for SIMD CPUs}, 
      author={Jesse DiCenso and Henryk Laqua and Martin Head-Gordon},
      year={2026},
      eprint={2610.21075},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2610.21075} 
}
```
