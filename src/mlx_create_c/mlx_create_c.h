#ifndef MLX_CREATE_C
#define MLX_CREATE_C
#include <numpy/arrayobject.h>

typedef struct {
  npy_long Nstates;
  npy_long Np;
  npy_long m;
} ns_basis_parameters;

npy_long nchoosek(npy_long, npy_long);
void create_bos_ns_core(npy_long*, ns_basis_parameters);
void create_mapmat_bos_core(npy_long*, npy_long*, ns_basis_parameters);
void configuration_from_state_index_core(npy_long*, npy_long, ns_basis_parameters);
npy_long get_index_from_configuration(npy_long*, npy_long, ns_basis_parameters);

#endif
