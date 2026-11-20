//
//   Accuracy and timing of generated kernels (driver of bench.py)
//
//   Usage: bench [--gb N] [--nprim 1,3] [--trials N] [--seconds S] [--extreme]
//                [--no-timing] [--flush-denormals] name:LA:LB:hrr:sc:nratio:nint:nbase ...
//
//   Every kernel is run on random shell pairs and compact batches of gridpoints.
//
//   Accuracy: the kernel under test is also instantiated in quadruple precision and run
//   on the same (already rounded) inputs, so that the difference is the roundoff of the
//   kernel alone.  Errors are relative to the largest magnitude that the same cartesian
//   integral reaches in the batch.
//
//   Timing: user space CPU cycles per gridpoint and primitive (wall time in ns where no
//   cycle counter is available), the minimum over many short bursts.  The kernels that
//   are compared run in turns on the same buffers: buffer addresses change the timings.
//

#include "bench.h"

#include <quadmath.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

typedef __float128 Q;
static const Q PI = 4*atanq(Q(1));

void register_test(Registry&);
void register_base(Registry&);
void register_ref(Registry&);

//
//   Clock
//

static int perf_fd = -1;

static void clock_init()
{
#ifdef __linux__
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    perf_fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
#endif
}

static inline double ticks()
{
#ifdef __linux__
    long long count;
    if (perf_fd >= 0 && read(perf_fd, &count, sizeof(count)) == sizeof(count)) return (double)count;
#endif
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()*1e9;
}

//
//   Test data
//

// Boys functions F_0 ... F_N
static void boys(int N, Q x, Q* F)
{
    const Q ex = expq(-x);
    if (x < 35) {
        // series for the highest order, downward recursion
        Q term = Q(1) / (2*N+1), sum = term;
        for (int k = 1; k < 2000; ++k) {
            term *= 2*x / (2*N+2*k+1);
            sum += term;
            if (term < sum*Q(1e-36)) break;
        }
        F[N] = ex*sum;
        for (int n = N-1; n >= 0; --n) F[n] = (2*x*F[n+1] + ex) / (2*n+1);
    } else {
        F[0] = sqrtq(PI/x)*erfq(sqrtq(x))/2;
        for (int n = 0; n < N; ++n) F[n+1] = ((2*n+1)*F[n] - ex) / (2*x);
    }
}

struct Data {
    int nprim, gb, Nmax, nratio;
    std::vector<Q> boys, p, PGx, PGy, PGz, PAx, PAy, PAz, ratios;
    Q AB[3];
};

// A shell pair with nprim primitive pairs and a batch of gb gridpoints within a cube of
// edge grid_size (bohr) close to the shell pair.
static Data make_data(int nprim, int gb, int Nmax, int nratio, bool same_center, unsigned seed,
                      double grid_size, bool extreme)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    Data D;
    D.nprim = nprim; D.gb = gb; D.Nmax = Nmax; D.nratio = nratio;

    double A[3], B[3], C[3];
    const double R = same_center ? 0.0 : 0.5 + 3.5*U(rng);
    const double theta = acos(2*U(rng)-1), phi = 2*M_PI*U(rng);
    for (int i = 0; i < 3; ++i) A[i] = U(rng) - 0.5;
    B[0] = A[0] + R*sin(theta)*cos(phi);
    B[1] = A[1] + R*sin(theta)*sin(phi);
    B[2] = A[2] + R*cos(theta);
    for (int i = 0; i < 3; ++i) D.AB[i] = (Q)A[i] - (Q)B[i];

    std::vector<double> G(3*gb);
    for (int i = 0; i < 3; ++i) C[i] = A[i] + (B[i]-A[i])*U(rng) + (U(rng)-0.5)*grid_size;
    for (int g = 0; g < gb; ++g)
        for (int i = 0; i < 3; ++i) G[3*g+i] = C[i] + (U(rng)-0.5)*grid_size;

    D.boys.resize((size_t)nprim*(Nmax+1)*gb);
    D.p.resize(nprim);
    D.PGx.resize((size_t)nprim*gb); D.PGy.resize((size_t)nprim*gb); D.PGz.resize((size_t)nprim*gb);
    D.PAx.resize(nprim); D.PAy.resize(nprim); D.PAz.resize(nprim);
    D.ratios.resize((size_t)nprim*std::max(nratio, 1));

    // exponents: valence-like, or from very diffuse to core-like
    const double emin = extreme ? 1e-3 : 0.08, emax = extreme ? 1e5 : 30.0;
    std::vector<Q> F(Nmax+1);
    for (int k = 0; k < nprim; ++k) {
        const double a = exp(log(emin) + U(rng)*log(emax/emin));
        const double b = exp(log(emin) + U(rng)*log(emax/emin));
        const Q p = (Q)a + (Q)b;
        Q P[3];
        for (int i = 0; i < 3; ++i) P[i] = ((Q)a*A[i] + (Q)b*B[i])/p;
        D.p[k] = p;
        D.PAx[k] = P[0]-A[0]; D.PAy[k] = P[1]-A[1]; D.PAz[k] = P[2]-A[2];
        // Gaussian product prefactor times a contraction coefficient
        const Q prefactor = 2*PI/p * expq(-(Q)a*b/p*R*R) * (0.2+U(rng));
        for (int r = 0; r < nratio; ++r) D.ratios[(size_t)k*nratio+r] = 0.2 + 1.5*U(rng);
        for (int g = 0; g < gb; ++g) {
            const Q x = P[0]-G[3*g], y = P[1]-G[3*g+1], z = P[2]-G[3*g+2];
            D.PGx[(size_t)k*gb+g] = x; D.PGy[(size_t)k*gb+g] = y; D.PGz[(size_t)k*gb+g] = z;
            boys(Nmax, p*(x*x+y*y+z*z), F.data());
            for (int N = 0; N <= Nmax; ++N) D.boys[((size_t)k*(Nmax+1) + N)*gb + g] = prefactor*F[N];
        }
    }
    return D;
}

// Array that starts on a page boundary, like the arrays of an aligned workspace
template <class T> struct Array {
    T* p = nullptr;
    size_t n = 0;
    void resize(size_t m) {
        free(p);
        n = m;
        p = (T*)aligned_alloc(4096, ((m*sizeof(T))/4096 + 2)*4096);
        for (size_t i = 0; i < m; ++i) p[i] = T(0);
    }
    Array() {}
    Array(const Array&) = delete;
    ~Array() { free(p); }
};

// The arguments of a kernel in precision T
template <class T> struct Buffers {
    Array<T> boys, p, PGx, PGy, PGz, PAx, PAy, PAz, ratios, integrals, base;
    Args<T> args;
    template <class V> static void convert(const V& src, size_t n, Array<T>& dst) {
        dst.resize(n);
        for (size_t i = 0; i < n; ++i) dst.p[i] = (T)src[i];
    }
    template <class S> Buffers(const S& D, const Q* AB, int nprim, int gb, int nint, int nbase) {
        convert(D.boys, D.boys.size(), boys); convert(D.p, D.p.size(), p);
        convert(D.PGx, D.PGx.size(), PGx); convert(D.PGy, D.PGy.size(), PGy); convert(D.PGz, D.PGz.size(), PGz);
        convert(D.PAx, D.PAx.size(), PAx); convert(D.PAy, D.PAy.size(), PAy); convert(D.PAz, D.PAz.size(), PAz);
        convert(D.ratios, D.ratios.size(), ratios);
        integrals.resize((size_t)nint*gb);
        base.resize((size_t)std::max(nbase, 1)*gb);
        args = Args<T>{integrals.p, base.p, boys.p, p.p, PGx.p, PGy.p, PGz.p, PAx.p, PAy.p, PAz.p,
                       (T)AB[0], (T)AB[1], (T)AB[2], ratios.p, nprim, gb};
    }
    std::vector<T> result() const { return std::vector<T>(integrals.p, integrals.p + integrals.n); }
};

// The inputs as they were rounded to precision T
template <class T> struct Rounded {
    std::vector<T> boys, p, PGx, PGy, PGz, PAx, PAy, PAz, ratios;
    explicit Rounded(const Buffers<T>& b) {
        auto copy = [](const Array<T>& a) { return std::vector<T>(a.p, a.p + a.n); };
        boys = copy(b.boys); p = copy(b.p); PGx = copy(b.PGx); PGy = copy(b.PGy); PGz = copy(b.PGz);
        PAx = copy(b.PAx); PAy = copy(b.PAy); PAz = copy(b.PAz); ratios = copy(b.ratios);
    }
};

//
//   Accuracy
//

struct Error {
    double max = 0, sum2 = 0;
    long n = 0;
    double rms() const { return n ? sqrt(sum2/n) : 0; }
};

template <class T>
static void accumulate(Error& e, const std::vector<Q>& ref, const std::vector<T>& val, int nint, int gb)
{
    for (int i = 0; i < nint; ++i) {
        Q scale = 0;
        for (int g = 0; g < gb; ++g) scale = std::max(scale, fabsq(ref[(size_t)i*gb+g]));
        if (scale == 0) scale = 1;
        for (int g = 0; g < gb; ++g) {
            double r = (double)(fabsq((Q)val[(size_t)i*gb+g] - ref[(size_t)i*gb+g])/scale);
            if (!(r == r)) r = 1e300;   // NaN
            e.max = std::max(e.max, r);
            e.sum2 += r*r;
            e.n++;
        }
    }
}

template <class T>
static void accuracy(const Data& D, int nint, int nbase, const Kernel& test, const Kernel& base, const Kernel& ref,
                     Error& etest, Error& ebase)
{
    Buffers<T> b(D, D.AB, D.nprim, D.gb, nint, nbase);
    // quadruple precision kernel on the rounded inputs
    Rounded<T> rounded(b);
    const Q AB[3] = {(Q)b.args.ABx, (Q)b.args.ABy, (Q)b.args.ABz};
    Buffers<Q> q(rounded, AB, D.nprim, D.gb, nint, nbase);
    ref.q(q.args);
    const std::vector<Q> exact = q.result();

    test.get<T>()(b.args);
    accumulate(etest, exact, b.result(), nint, D.gb);
    if (base.get<T>()) {
        base.get<T>()(b.args);
        accumulate(ebase, exact, b.result(), nint, D.gb);
    }
}

//
//   Timing
//

template <class T> static double time_kernel(void (*kernel)(const Args<T>&), const Args<T>& args, double seconds)
{
    typedef std::chrono::steady_clock clock;
    const auto start = clock::now();
    kernel(args);
    const double once = std::chrono::duration<double>(clock::now() - start).count();
    const int reps = std::max(1, (int)(5e-5/std::max(once, 1e-9)));   // bursts of about 50 microseconds
    double best = 1e300;
    while (std::chrono::duration<double>(clock::now() - start).count() < seconds) {
        const double t0 = ticks();
        for (int r = 0; r < reps; ++r) kernel(args);
        best = std::min(best, (ticks() - t0)/reps);
    }
    return best;
}

template <class T>
static void timing(const Data& D, int nint, int nbase, const Kernel& test, const Kernel& base, double seconds,
                   double& ttest, double& tbase)
{
    Buffers<T> b(D, D.AB, D.nprim, D.gb, nint, nbase);
    const int rounds = 10;   // the kernels take turns, to see the same state of the machine
    ttest = tbase = 1e300;
    for (int round = 0; round < rounds; ++round) {
        ttest = std::min(ttest, time_kernel(test.get<T>(), b.args, seconds/rounds));
        if (base.get<T>()) tbase = std::min(tbase, time_kernel(base.get<T>(), b.args, seconds/rounds));
    }
    const double per_point = 1.0/((double)D.gb*D.nprim);
    ttest *= per_point;
    tbase *= per_point;
}

//
//   Driver
//

template <class T>
static void report(const char* name, const char* type, int nprim, int gb, int Nmax, int nratio, bool same_center,
                   int nint, int nbase, const Kernel& test, const Kernel& base, const Kernel& ref,
                   int trials, double seconds, bool extreme, bool do_timing, unsigned seed)
{
    const bool have_base = base.get<T>() != nullptr;
    printf("%-22s %5d %7d  %-6s", name, nprim, gb, type);

    if (do_timing) {
        const Data D = make_data(nprim, gb, Nmax, nratio, same_center, 4242 + seed, 3.0, extreme);
        double ttest, tbase;
        timing<T>(D, nint, nbase, test, base, seconds, ttest, tbase);
        printf(" %9.2f", ttest);
        if (have_base) printf(" %9.2f %6.2fx", tbase, tbase/ttest);
    }

    Error etest, ebase;
    for (int t = 0; t < trials; ++t) {
        // batches of gridpoints of different extent
        const double grid_size = (t % 3 == 0) ? 1.0 : (t % 3 == 1) ? 3.0 : 8.0;
        const Data D = make_data(nprim, gb, Nmax, nratio, same_center, 1000*seed + 17*t + nprim, grid_size, extreme);
        accuracy<T>(D, nint, nbase, test, base, ref, etest, ebase);
    }
    if (trials > 0) {
        printf("  %9.2e %9.2e", etest.rms(), etest.max);
        if (have_base) printf("  %9.2e %9.2e", ebase.rms(), ebase.max);
    }
    printf("\n");
    fflush(stdout);
}

int main(int argc, char** argv)
{
    int gb = 128, trials = 6;
    double seconds = 0.5;
    bool extreme = false, do_timing = true;
    std::vector<int> nprims = {1, 3};
    std::vector<std::string> specs;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gb" && i+1 < argc) gb = atoi(argv[++i]);
        else if (arg == "--trials" && i+1 < argc) trials = atoi(argv[++i]);
        else if (arg == "--seconds" && i+1 < argc) seconds = atof(argv[++i]);
        else if (arg == "--nprim" && i+1 < argc) {
            nprims.clear();
            for (char* tok = strtok(argv[++i], ","); tok; tok = strtok(nullptr, ",")) nprims.push_back(atoi(tok));
        }
        else if (arg == "--extreme") extreme = true;
        else if (arg == "--no-timing") do_timing = false;
        else if (arg == "--flush-denormals") {
#if defined(__SSE__)
            // what a program linked with -ffast-math does at startup
            _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
        }
        else specs.push_back(arg);
    }

    Registry test, base, ref;
    register_test(test);
    register_base(base);
    register_ref(ref);
    clock_init();

    const bool have_base = !base.empty();
    const char* unit = (perf_fd >= 0) ? "cycles" : "ns";
    printf("# time: %s per gridpoint and primitive; errors: relative to the largest integral of the batch\n", unit);
    printf("%-22s %5s %7s  %-6s", "kernel", "nprim", "gb_size", "type");
    if (do_timing) {
        printf(" %9s", unit);
        if (have_base) printf(" %9s %7s", "baseline", "speedup");
    }
    if (trials > 0) {
        printf("  %9s %9s", "rms error", "max error");
        if (have_base) printf("  %9s %9s", "base rms", "base max");
    }
    printf("\n");

    for (size_t s = 0; s < specs.size(); ++s) {
        char name[128];
        int LA, LB, hrr, sc, nratio, nint, nbase;
        if (sscanf(specs[s].c_str(), "%127[^:]:%d:%d:%d:%d:%d:%d:%d", name, &LA, &LB, &hrr, &sc, &nratio, &nint, &nbase) != 8) {
            fprintf(stderr, "bench: cannot parse kernel specification %s\n", specs[s].c_str());
            return 1;
        }
        if (!test.count(name) || !ref.count(name)) {
            fprintf(stderr, "bench: kernel %s is not linked in\n", name);
            return 1;
        }
        const Kernel kbase = base.count(name) ? base[name] : Kernel();
        for (int nprim : nprims) {
            report<float>(name, "float", nprim, gb, LA+LB, nratio, sc, nint, nbase, test[name], kbase, ref[name],
                          trials, seconds, extreme, do_timing, (unsigned)s + 1);
            report<double>(name, "double", nprim, gb, LA+LB, nratio, sc, nint, nbase, test[name], kbase, ref[name],
                           trials, seconds, extreme, do_timing, (unsigned)s + 1);
        }
    }
    return 0;
}
