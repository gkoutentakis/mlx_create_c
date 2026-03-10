#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mlx_create_c.h"

/* Calculation of the binomial factor */
/* VALIDITY: This routine uses floating point approximations for the factorial.
 It is guaranteed to provide an approximate value in the range that the result
 does not overflow but there might be errors at the +- a few integer level */
npy_long
nchoosek(npy_long n, npy_long k)
{
  /* maximum long double value that can be rounded to npy_long */
  register long double max_long_value = (long double)NPY_MAX_LONG -0.5L;

  if (n < 0 || k < 0 || k > n)
    return 0;   /* or 0, depending on your convention */

  long double nl = (long double)n;
  long double kl = (long double)k;

  long double result_float =
      expl(lgammal(nl + 1.0L) - lgammal(kl + 1.0L) - lgammal(nl - kl + 1.0L));

  if (!isfinite(result_float) || result_float > max_long_value || result_float < 0)
    return -1;

  return lroundl(result_float);
}

/* This routine list all bosonic number state configurations in reverse
   lexicographic order:
   the k-th number state (zero based indexing) corresponds to the range
   Ns[k*basis_par.m] up to Ns[(k+1)*basis_par.m-1] */
/* VALIDITY: This routine is exact but memory intensive, in practice
 basis_par.Ns < 1e9 for runs in 2026 PC hardware */
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

/* Index calculation for bosonic number states in reverse lexicographic
   order according to:
   A. I. Streltsov, et. al, Phys. Rev. A 81, 022124 (2010)
   DOI: https://doi.org/10.1103/PhysRevA.81.022124
   see Eq. (20). The code is adapted for zero based indexing. */
/* VALIDITY: This routine uses a floating point approximation via nchoosek.
 so it has the same regime of validity (see related comment in nchoosek). */
npy_long
get_index_from_configuration(npy_long* configuration,
			     npy_long orbital_new,
			     ns_basis_parameters basis_par)
{
  npy_long orbital_index, n, k;
  npy_long particle_count=0L, output=0;

  for (orbital_index=0L; orbital_index<basis_par.m-1; orbital_index++){
    particle_count += configuration[orbital_index];
    if (orbital_index == orbital_new) particle_count++;

    n = basis_par.Np + basis_par.m - orbital_index -1 - particle_count;
    k =  basis_par.m - orbital_index - 1;
    output += nchoosek(n, k);
  }

  return output;
}

/* The mapping matrix encoding the action of creation operator the N particle
   number state with index k under the action of the creation operator for the
   orbital j maps to the state with index map[k*basis_par.m + j]. */
/* VALIDITY: This routine uses a floating point approximation via nchoosek
 (see get_index_from_configuration) so it has the same regime of validity
 (see related comment in nchoosek). */
void
create_mapmat_bos_core(npy_long* map, npy_long* Ns1, ns_basis_parameters basis_par)
{
  npy_long state_index, orbital_new;
  npy_long *current_map_element = map;
  npy_long *current_number_state = Ns1;

  for (state_index=0L; state_index < basis_par.Nstates;
       state_index++, current_number_state += basis_par.m)
    for (orbital_new=0L; orbital_new<basis_par.m;
	 orbital_new++, current_map_element++)
      *current_map_element =
	get_index_from_configuration(current_number_state, orbital_new, basis_par);
}


/* Bosonic number state configuration calculation from the index in reverse
   lexicographic order according to:
   A. I. Streltsov, et. al, Phys. Rev. A 81, 022124 (2010)
   DOI: https://doi.org/10.1103/PhysRevA.81.022124
   see Eq. (20). The code is adapted for zero based indexing. */
/* VALIDITY: this routine is safe only when
 basis_par.Nstates * basis_par.Np < MAX_NPY_LONG
 as the block_size update step can overflow. */
void
configuration_from_state_index_core(npy_long *configuration,
                                    npy_long index,
                                    ns_basis_parameters basis_par)
{
  npy_long i, j, n, k, block_size;

  npy_long particles_remaining = basis_par.Np;
  npy_long *occupation_current_orbital = configuration;

  for (i = 0, k = basis_par.m - 2; i < basis_par.m - 1;
       i++, occupation_current_orbital++, k--) {
    /* Iterate over the blocks of states with equal occupation
     for the current orbital:
     Start from the first block where all remaining particles are there,
     then move to the next block by decreasing its occupation.
     At the same time we keep track on how:
     (i) the size of the block (block_size) and (ii) the n of the
     binomial giving block_size change as we go from one block to
     the next. */
    for (*occupation_current_orbital = particles_remaining, // first block
	    n = k,
	    block_size = 1;
         *occupation_current_orbital > 0;                   // last block
	 (*occupation_current_orbital)--,                   // next block
	    n++,
	    block_size = (block_size * n) / (n - k)) {

      if (index > block_size) {
	// We move past the block
        index -= block_size;
	continue;
      }

      if (index < block_size) {
	// We are some where in this block so we go to the next orbital
	// and do the same blocking idea again for the remaining particles
        particles_remaining -= *occupation_current_orbital;
        break;
      } 

      /* If equal (nor larger or smaller) we found the state:
       It is the first state of the next block, having the following
       properties: */

      /* 1. the occupation of the current orbital is smaller by one, */
      (*occupation_current_orbital)--;

      /* 2. the remaining particles (with the ones of the current orbital
         subtracted of course) are on the next orbital to the current, */
      occupation_current_orbital[1] =
	particles_remaining - occupation_current_orbital[0];

      /* 3. the rest of the orbitals have zero particles. */
      for (npy_long j = 2; j < basis_par.m - i; j++)
        occupation_current_orbital[j] = 0;

      /* we are done here */
      return;
    }
  }

  /* if some particles remain put them on the last orbital */
  *occupation_current_orbital = particles_remaining;
}
