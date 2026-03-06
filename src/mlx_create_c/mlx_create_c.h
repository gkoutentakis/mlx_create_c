#ifndef MLX_CREATE_C
#define MLX_CREATE_C
#include <numpy/arrayobject.h>

npy_long nchoosek(npy_long, npy_long);
void create_bos_ns_core(npy_long*, npy_long, npy_long, npy_long);
void create_mapmat_bos_core(npy_long*, npy_long*, npy_long, npy_long, npy_long);

#endif
