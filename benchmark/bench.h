//
//   Uniform interface to the generated kernels (shared by bench.cpp and the
//   translation units that bench.py writes around the kernels)
//

#ifndef SNK_BENCH_H
#define SNK_BENCH_H

#include <map>
#include <string>

// All arguments that a kernel may take; a wrapper passes on the ones of its kernel
template <class T> struct Args {
    T* integrals;
    T* base_integrals;
    const T* v_scaled_boys;
    const T* v_p;
    const T* v_PGx;
    const T* v_PGy;
    const T* v_PGz;
    const T* v_PAx;
    const T* v_PAy;
    const T* v_PAz;
    T ABx, ABy, ABz;
    const T* v_ratios;
    int nprim;
    int gb_size;
};

struct Kernel {
    void (*f)(const Args<float>&) = nullptr;
    void (*d)(const Args<double>&) = nullptr;
    void (*q)(const Args<__float128>&) = nullptr;
    template <class T> void (*get() const)(const Args<T>&);
};
template <> inline void (*Kernel::get<float>() const)(const Args<float>&) { return f; }
template <> inline void (*Kernel::get<double>() const)(const Args<double>&) { return d; }

typedef std::map<std::string, Kernel> Registry;

#endif // SNK_BENCH_H
