# SNK integral codegen

Integral kernel code generation for evaluating 3c1e nuclear attraction-type integrals, originally used for seminumerical exchange (SNK). Both VRR and HRR kernels can be generated with/without same-center optimizations for arbitrary multishells (sets of shells sharing the same center and primitive exponents but potentially having different L quantum numbers and contraction coefficients). 

The output kernels are written as templates for mixed precision evaluation (FP32 and FP64). These kernels are optimized for CPUs with SIMD vector instructions. No external C++ headers are required and the generated code is compatible with C++11 projects.

## Requirements

- Python (3.13.5 used originally)
- SymPy  (1.13.3 used originally)

## Structure

The code is structured as an executable `main.py` along with several helper files. `recurrences.py` contains the actual recurrence relations used to produce algebraic expressions for the 3c1e integrals. `codegen.py` uses these expressions, along with Common Subexpression Elimination (CSE) to produce optimal C++ code for the actual computation of 3c1e integrals. `codestrings.py` is a helper file for writing C++ code. `multishells.py` contains the multishell definitions, mapping strings to lists of L quantum numbers to be used as inputs to `codegen.generate_vrr()` and `codegen.generate_hrr()` (see below).

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
