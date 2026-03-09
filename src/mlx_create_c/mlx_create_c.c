#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mlx_create_c.h"

npy_long
nchoosek(npy_long n, npy_long k)
{
  /* maximum long double value that can be rounded to npy_long */
  register long double max_long_value = (long double)NPY_MAX_LONG -0.5L;

  long double nl = (long double)n;
  long double kl = (long double)k;

  long double result_float =
      expl(lgammal(nl + 1.0L) - lgammal(kl + 1.0L) - lgammal(nl - kl + 1.0L));

  if (!isfinite(result_float) || result_float > max_long_value || result_float < 0)
    return -1;

  return lroundl(result_float);
}

void
create_bos_ns_core(npy_long *Ns, ns_basis_parameters basis_par)
{
  npy_long i, j, stack, set;

  Ns[0] = basis_par.Np;
  for (i=1L; i<basis_par.m; i++)
    Ns[i] = 0L;

  for (i=1L; i<basis_par.Nstates; i++){
    stack = Ns[i*basis_par.m - 1];

    for (j=0L; j<basis_par.m-1L; j++){
      Ns[i*basis_par.m+j] = Ns[(i-1)*basis_par.m + j];
      if (Ns[(i-1)*basis_par.m + j] > 0L) set = j;
    }
    Ns[(i+1)*basis_par.m -1] = 0;

    Ns[i*basis_par.m+set]--;
    Ns[i*basis_par.m+set+1] += stack + 1L;
  }
  
}

void
create_mapmat_bos_core(npy_long* map, npy_long* Ns1, ns_basis_parameters basis_par)
{
  npy_long state_index, orbital_new, orbital_index, particle_count, n, k;
  npy_long *current_map_element, *current_number_state;

  current_map_element = map;
  current_number_state = Ns1;

  for (state_index=0L; state_index < basis_par.Nstates;
       state_index++, current_number_state += basis_par.m){
    for (orbital_new=0L; orbital_new<basis_par.m;
	 orbital_new++, current_map_element++){

      *current_map_element = 0L;
      particle_count = 0L;

      for (orbital_index=0L; orbital_index<basis_par.m-1; orbital_index++){
	particle_count += current_number_state[orbital_index];
	if (orbital_index == orbital_new) particle_count++;

	n = basis_par.Np + basis_par.m - orbital_index -1 - particle_count;
	k =  basis_par.m - orbital_index - 1;
	*current_map_element += nchoosek(n, k);
      }

    }
  }
}

/* void */
/* configuration_from_state_index(npy_long* configuration, npy_long index, */
/* 			       ns_basis_parameters basis_par)     */
/* { */
  
/* } */
