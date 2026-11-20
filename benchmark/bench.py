#!/usr/bin/env python3
#
#   Benchmark of generated kernels: speed and roundoff
#
#   Generates the requested kernels, compiles them together with bench.cpp and runs the
#   result.  The kernels can be compared with a baseline: kernels in a directory, kernels
#   of another revision of this generator, or this generator with its default settings.
#
#   Usage: python benchmark/bench.py [options] kind:MSA:MSB:same_center ...
#
#   Examples:
#       python benchmark/bench.py vrr:D:P:0 hrr:D:D:0 vrr:D:D:1
#       python benchmark/bench.py --baseline 134c94b vrr:F:D:0 hrr:F:F:0
#       python benchmark/bench.py --baseline default --set BLOCK_POINTS=16 --gb 128,120 hrr:F:F:0
#
#   Needs g++ with libquadmath (the reference values are computed in __float128), and
#   for the baseline of another revision what that revision needs (SymPy).
#

import argparse
import ast
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import codegen
from multishells import MULTISHELLS, num_cartesian_integrals, get_hrr_blist

class KernelSpec:
    def __init__(self, text):
        try:
            kind, self.MSA, self.MSB, sc = text.split(":")
            self.hrr = {"vrr": False, "hrr": True}[kind.lower()]
            self.same_center = bool(int(sc))
            vLA, vLB = MULTISHELLS[self.MSA], MULTISHELLS[self.MSB]
        except (ValueError, KeyError):
            sys.exit(f"bench.py: bad kernel '{text}', expected kind:MSA:MSB:same_center such as vrr:D:P:0 "
                     f"with multishells from multishells.py")
        if self.hrr and (max(vLA) == 0 or max(vLB) == 0):
            sys.exit(f"bench.py: there is no HRR kernel for '{text}'")
        self.text = text
        self.kind = kind.lower()
        self.name = f"{kind.upper()}_{self.MSA}_{self.MSB}" + ("_SAMECENTER" if self.same_center else "")
        self.LA, self.LB = max(vLA), max(vLB)
        self.nratio = len(vLA)*len(vLB) - 1
        self.nint = num_cartesian_integrals(vLA, vLB)
        self.nbase = len(get_hrr_blist(vLA, vLB, self.same_center)) if self.hrr else 0

    def arguments(self):
        """The argument list of the kernel in terms of the members of Args (bench.h)."""
        args = ["a.integrals"] + (["a.base_integrals"] if self.hrr else [])
        args += ["a.v_scaled_boys", "a.v_p", "a.v_PGx", "a.v_PGy", "a.v_PGz"]
        if not self.same_center:
            args += ["a.v_PAx", "a.v_PAy", "a.v_PAz", "a.ABx", "a.ABy", "a.ABz"]
        if self.nratio > 0:
            args += ["a.v_ratios"]
        return ", ".join(args + ["a.nprim", "a.gb_size"])

    def driver_argument(self):
        return f"{self.name}:{self.LA}:{self.LB}:{int(self.hrr)}:{int(self.same_center)}:{self.nratio}:{self.nint}:{self.nbase}"

def generate(specs, dname, settings=None):
    """Generate kernels with this generator; settings are module variables of codegen."""
    settings = settings or {}
    saved = {k: getattr(codegen, k) for k in settings}
    for k, v in settings.items():
        setattr(codegen, k, v)
    try:
        for s in specs:
            getattr(codegen, f"generate_{s.kind}")(MSA=s.MSA, MSB=s.MSB, same_center=s.same_center, dname=dname, verbose=False)
    finally:
        for k, v in saved.items():
            setattr(codegen, k, v)

def generate_revision(specs, revision, build):
    """Generate kernels with the generator of another git revision; returns their directory."""
    source = os.path.join(build, "baseline_source")
    dname = os.path.join(build, "baseline")
    shutil.rmtree(source, ignore_errors=True)
    os.makedirs(source)
    archive = subprocess.run(["git", "-C", ROOT, "archive", revision], stdout=subprocess.PIPE)
    if archive.returncode != 0:
        sys.exit(f"bench.py: '{revision}' is neither a directory nor a git revision")
    subprocess.run(["tar", "-x", "-C", source], input=archive.stdout, check=True)
    script = "import codegen\n"
    for s in specs:
        script += f"codegen.generate_{s.kind}(MSA={s.MSA!r}, MSB={s.MSB!r}, same_center={s.same_center}, dname={dname!r}, verbose=False)\n"
    print(f"Generating the baseline kernels with revision {revision}...", flush=True)
    subprocess.run([sys.executable, "-c", script], cwd=source, check=True)
    return dname

def write_translation_unit(path, variant, dname, specs, precisions):
    """Wrap the kernels of a directory in a namespace and register them under their names."""
    with open(path, "w") as f:
        f.write('#include "bench.h"\n\n')
        f.write(f"namespace {variant} {{\n")
        for s in specs:
            f.write(f'#include "{os.path.join(os.path.abspath(dname), s.name)}.C"\n')
        f.write("}\n\n")
        for s in specs:
            f.write(f"template <class T> static void {s.name}(const Args<T>& a) {{ {variant}::{s.name}<T>({s.arguments()}); }}\n")
        f.write(f"\nvoid register_{variant}(Registry& r)\n{{\n")
        for s in specs:
            for member, T in precisions:
                f.write(f'    r["{s.name}"].{member} = &{s.name}<{T}>;\n')
        f.write("}\n")

def main():
    parser = argparse.ArgumentParser(description="Speed and roundoff of generated kernels.",
                                     usage="%(prog)s [options] kind:MSA:MSB:same_center ...")
    parser.add_argument("kernels", nargs="+", help="e.g. vrr:D:P:0 hrr:F:F:1")
    parser.add_argument("--baseline", metavar="WHAT",
                        help="compare with the kernels in a directory, with a git revision of this generator, "
                             "or with 'default' (this generator without --set)")
    parser.add_argument("--set", metavar="NAME=VALUE", action="append", default=[],
                        help="set a module variable of codegen for the kernels under test, e.g. BLOCK_MIN_BYTES=400")
    parser.add_argument("--gb", default="128", help="gridpoints per batch, comma separated (default 128)")
    parser.add_argument("--nprim", default="1,3", help="numbers of primitive pairs, comma separated (default 1,3)")
    parser.add_argument("--trials", type=int, default=6, help="random shell pairs for the errors, 0 for none (default 6)")
    parser.add_argument("--seconds", type=float, default=0.5, help="time spent on each timing (default 0.5)")
    parser.add_argument("--no-timing", action="store_true", help="errors only")
    parser.add_argument("--extreme", action="store_true", help="exponents from 1e-3 to 1e5 instead of 0.08 to 30")
    parser.add_argument("--cxx", default="g++", help="compiler of the kernels (default g++)")
    parser.add_argument("--flags", default="-O3 -march=native -ffast-math -fopenmp-simd",
                        help="compiler flags of the kernels (default: %(default)s)")
    parser.add_argument("--cpu", help="pin the benchmark to this CPU with taskset")
    parser.add_argument("--jobs", type=int, default=2, help="parallel compilations (default 2)")
    parser.add_argument("--build", default="SNK_BENCH", help="build directory (default SNK_BENCH)")
    args = parser.parse_args()

    specs = [KernelSpec(k) for k in args.kernels]
    settings = {}
    for item in args.set:
        name, _, value = item.partition("=")
        if not hasattr(codegen, name) or not value:
            sys.exit(f"bench.py: --set {item}: expected NAME=VALUE with a module variable of codegen")
        settings[name] = ast.literal_eval(value)

    build = os.path.abspath(args.build)
    os.makedirs(build, exist_ok=True)

    #
    #   Kernels
    #

    test_dir = os.path.join(build, "test")
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"Generating the kernels{' with ' + ', '.join(args.set) if args.set else ''}...", flush=True)
    generate(specs, test_dir, settings)

    base_dir = None
    if args.baseline == "default":
        base_dir = os.path.join(build, "baseline")
        shutil.rmtree(base_dir, ignore_errors=True)
        generate(specs, base_dir)
    elif args.baseline and os.path.isdir(args.baseline):
        base_dir = args.baseline
    elif args.baseline:
        shutil.rmtree(os.path.join(build, "baseline"), ignore_errors=True)
        base_dir = generate_revision(specs, args.baseline, build)
    if base_dir:
        missing = [s.name for s in specs if not os.path.exists(os.path.join(base_dir, s.name + ".C"))]
        if missing:
            sys.exit(f"bench.py: the baseline {base_dir} has no kernel {', '.join(missing)}")

    #
    #   Compile
    #

    both = [("f", "float"), ("d", "double")]
    write_translation_unit(os.path.join(build, "tu_test.cpp"), "test", test_dir, specs, both)
    # the reference values come from the kernels under test in quadruple precision
    write_translation_unit(os.path.join(build, "tu_ref.cpp"), "ref", test_dir, specs, [("q", "__float128")])
    if base_dir:
        write_translation_unit(os.path.join(build, "tu_base.cpp"), "base", base_dir, specs, both)
    else:
        with open(os.path.join(build, "tu_base.cpp"), "w") as f:
            f.write('#include "bench.h"\n\nvoid register_base(Registry&) {}\n')

    include = f"-I{HERE}"
    commands = [f"{args.cxx} {args.flags} {include} -c tu_test.cpp -o tu_test.o",
                f"{args.cxx} {args.flags} {include} -c tu_base.cpp -o tu_base.o",
                f"g++ -O1 {include} -c tu_ref.cpp -o tu_ref.o",
                f"g++ -O2 {include} -c {os.path.join(HERE, 'bench.cpp')} -o bench.o"]
    print(f"Compiling ({args.cxx} {args.flags})...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda c: subprocess.run(c, shell=True, cwd=build).returncode, commands))
    if any(results):
        sys.exit("bench.py: compilation failed")
    subprocess.run("g++ bench.o tu_test.o tu_base.o tu_ref.o -lquadmath -o bench", shell=True, cwd=build, check=True)

    #
    #   Run
    #

    command = (["taskset", "-c", args.cpu] if args.cpu else []) + [os.path.join(build, "bench")]
    command += ["--nprim", args.nprim, "--trials", str(args.trials), "--seconds", str(args.seconds)]
    command += ["--extreme"] if args.extreme else []
    command += ["--no-timing"] if args.no_timing else []
    # -ffast-math sets the flush-to-zero mode of the program at link time
    command += ["--flush-denormals"] if ("-ffast-math" in args.flags or "-Ofast" in args.flags) else []
    for gb in args.gb.split(","):
        status = subprocess.run(command + ["--gb", gb] + [s.driver_argument() for s in specs]).returncode
        if status != 0:
            sys.exit(status)

if __name__ == "__main__":
    main()
