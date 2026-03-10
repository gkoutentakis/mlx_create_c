#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include "mlx_create_c.h"
#include <stdio.h>

static
PyObject* create_bos_ns(PyObject* self, PyObject* args)
{
  npy_long Np, m;

  // Inputs
  if (!PyArg_ParseTuple(args, "ll", &Np, &m))
      return NULL;
  if (Np < 0 || m < 0) {
      PyErr_SetString(PyExc_ValueError, "Require Np>=0, m>=0");
      return NULL;
  }

  // Outputs
  const ns_basis_parameters bas_par = {
    .Nstates = nchoosek(Np + m -1, m-1),
    .Np = Np,
    .m = m
  };

  npy_intp dims[2] = { (npy_intp) bas_par.Nstates, (npy_intp) bas_par.m };
  PyObject* out = PyArray_SimpleNew(2, dims, NPY_LONG);
  if (!out) return NULL;

  // Type cast output to C array
  npy_long* v = (npy_long*) PyArray_DATA((PyArrayObject*)out);

  // Call the C function
  Py_BEGIN_ALLOW_THREADS;
  create_bos_ns_core(v, bas_par);
  Py_END_ALLOW_THREADS;

  return out;
}

static
PyObject* create_mapmat_bos(PyObject* self, PyObject* args)
{
  PyObject *input_object = NULL;
  PyArrayObject *input_array = NULL;
  npy_intp number_of_rows, number_of_columns;

  // parse object
  if (!PyArg_ParseTuple(args, "O", &input_object))
      return NULL;

  // parse array from object
  input_array = (PyArrayObject *) PyArray_FROMANY(
      input_object, NPY_LONG, 2, 2, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
  if (input_array == NULL)
    return NULL;

  // find metadata of the array
  number_of_rows = PyArray_DIM(input_array, 0);
  number_of_columns = PyArray_DIM(input_array, 1);

  // calculate parameters from input array sizes
  npy_long* Ns1 = (npy_long*) PyArray_DATA(input_array);

  const ns_basis_parameters bas_par = {
    .Nstates = number_of_rows,
    .Np = Ns1[0],
    .m = number_of_columns
  };
  npy_intp dims[2] = { (npy_intp) bas_par.Nstates, (npy_intp) bas_par.m };
  PyObject* out = PyArray_SimpleNew(2, dims, NPY_LONG);
  if (!out) return NULL;

  //type cast to C array
  npy_long* map = (npy_long*) PyArray_DATA((PyArrayObject*)out);

  // Call the C function
  Py_BEGIN_ALLOW_THREADS;
  create_mapmat_bos_core(map, Ns1, bas_par);
  Py_END_ALLOW_THREADS;

  Py_DECREF(input_array);

  return out;
}

/* for testing purposes */
static PyObject*
configuration_from_state_index_scalar(npy_long index, ns_basis_parameters bas_par)
{
    npy_intp dims[1] = { (npy_intp) bas_par.m };
    PyObject* out = PyArray_SimpleNew(1, dims, NPY_LONG);
    if (!out) return NULL;

    // Type cast output to C array
    npy_long* v = (npy_long*) PyArray_DATA((PyArrayObject*)out);

    // Call the C function
    Py_BEGIN_ALLOW_THREADS;
    configuration_from_state_index_core(v, index, bas_par);
    Py_END_ALLOW_THREADS;
    return out;
}

static PyObject*
configuration_from_state_index_array(npy_long* indices,
				     int ndims_input,
				     npy_intp* dims_input,
				     npy_intp number_of_elements_input,
				     ns_basis_parameters bas_par)
{
    npy_intp *out_dims = PyMem_Malloc((size_t)(ndims_input + 1) * sizeof(npy_intp));
    if (out_dims == NULL)
      return NULL;

    memcpy(out_dims, dims_input, (size_t)ndims_input * sizeof(npy_intp));
    out_dims[ndims_input] = (npy_intp) bas_par.m;

    PyObject *out = PyArray_SimpleNew(ndims_input + 1, out_dims, NPY_LONG);
    PyMem_Free(out_dims);
    if (!out)
      return NULL;

    // Type cast output to C array
    npy_long *out_data = (npy_long *) PyArray_DATA((PyArrayObject *)out);

    Py_BEGIN_ALLOW_THREADS;
    for (npy_intp i = 0; i < number_of_elements_input; i++)
      configuration_from_state_index_core(out_data + i * bas_par.m, indices[i],
                                          bas_par);
    Py_END_ALLOW_THREADS;
    return out;
}

static PyObject*
configuration_from_state_index(PyObject* self, PyObject* args)
{
  PyObject *py_object;
  npy_long Np, m;

  // Inputs
  if (!PyArg_ParseTuple(args, "Oll", &py_object, &Np, &m))
      return NULL;
  // Throw the Np, m errors
  if (Np < 0 || m < 0) {
      PyErr_SetString(PyExc_ValueError, "Require Np>=0, m>=0");
      return NULL;
  }

  // we have everything to define the basis
  const ns_basis_parameters bas_par = {
    .Nstates = nchoosek(Np + m -1, m-1), // we don't need to calculate this
    .Np = Np,
    .m = m
  };

  if (PyArray_IsScalar(py_object, Integer) || PyLong_Check(py_object)) {
    npy_long index = (npy_long) PyLong_AsLong(py_object);
    if (index < 0) {
      PyErr_SetString(PyExc_ValueError, "Indices should be non-negative");
      return NULL;
    }

    return configuration_from_state_index_scalar(index, bas_par);
  }

  PyArrayObject *indices_np = (PyArrayObject *)PyArray_FromAny(
      py_object, PyArray_DescrFromType(NPY_LONG), 0, 0,
      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSUREARRAY, NULL);
  if (!indices_np)
    return NULL;

  int ndims_input = PyArray_NDIM(indices_np);
  npy_intp number_of_elements_input = PyArray_SIZE(indices_np);

  if (number_of_elements_input == 0) {
      npy_intp out_dims[2] = {(npy_intp) 0, (npy_intp) m};
      PyObject *out = (PyObject *)PyArray_SimpleNew(2, out_dims, NPY_LONG);
      Py_DECREF(indices_np);
      return out;
  }

  npy_long *indices = (npy_long *) PyArray_DATA(indices_np);

  if (ndims_input == 0) {
    if (number_of_elements_input != 1) {
	PyErr_SetString(PyExc_RuntimeError, "0-D array with size != 1");
	Py_DECREF(indices_np);
	return NULL;
    }

    npy_long index = indices[0];
    Py_DECREF(indices_np);

    if (index < 0) {
      PyErr_SetString(PyExc_ValueError, "Invalid index in 0-D array");
      return NULL;
    }

    PyObject *out = configuration_from_state_index_scalar(index, bas_par);
    return out;
  }

  npy_intp *dims_input = PyArray_DIMS(indices_np);
  PyObject *out = configuration_from_state_index_array(indices,
						       ndims_input,
						       dims_input,
						       number_of_elements_input,
						       bas_par);
  Py_DECREF(indices_np);
  return out;
}

/* several inefficiencies here maybe we need to write a dedicated core for this */
void
creation_operator_ufunc(char **args,
                        const npy_intp *dimensions,
                        const npy_intp *steps,
                        void *data)
{
  npy_long ns_ind_in, orbital_ind_in, ns_ind_out, Np_in, m_in;
  double weight_out;
  ns_basis_parameters bas_par;

  npy_long configuration_data[128];

  npy_intp n = dimensions[0];

  char *pointer_ns_ind_in   = args[0];
  char *pointer_orbital_ind = args[1];
  char *pointer_Np_in       = args[2];
  char *pointer_m_in        = args[3];
  char *pointer_ns_ind_out  = args[4];
  char *pointer_weight_out  = args[5];

  npy_intp step_ns_ind_in   = steps[0];
  npy_intp step_orbital_ind = steps[1];
  npy_intp step_Np_in       = steps[2];
  npy_intp step_m_in        = steps[3];
  npy_intp step_ns_ind_out  = steps[4];
  npy_intp step_weight_out  = steps[5];

  for (npy_intp i = 0; i < n; ++i) {
    ns_ind_in      = *(npy_long *) pointer_ns_ind_in;
    orbital_ind_in = *(npy_long *) pointer_orbital_ind;
    Np_in          = *(npy_long *) pointer_Np_in;
    m_in           = *(npy_long *) pointer_m_in;

    bas_par.Nstates = nchoosek(Np_in + m_in-1, m_in-1);
    bas_par.Np = Np_in;
    bas_par.m = m_in;

    if (Np_in < 0){
        fprintf(stderr, "number of particles should be positive");
	return;
    }
    if (m_in < 0){
        fprintf(stderr, "number of orbitals should be positive");
	return;
    }
    if (m_in >= 128){
        fprintf(stderr, "this function supports up to 128 orbitals");
	return;
    }
    if (ns_ind_in<0 || ns_ind_in>=bas_par.Nstates){
        fprintf(stderr, "number state index out of bounds");
	return;
    }
    if (orbital_ind_in<0 || orbital_ind_in >= bas_par.m){
        fprintf(stderr, "orbital index out of bounds");
	return;
    }
    // core
    /* configuration_data = (npy_long*) malloc((unsigned long) m_in * sizeof(npy_long)); */
    /* if (!configuration_data){ */
    /*   fprintf(stderr, "Out of memory\n"); */
    /*   return; */
    /* } */
    configuration_from_state_index_core(configuration_data, ns_ind_in, bas_par);
    ns_ind_out = get_index_from_configuration(configuration_data, orbital_ind_in, bas_par);
    weight_out = sqrt((double) (configuration_data[orbital_ind_in] + 1));
    /* free(configuration_data); */

    *(npy_long *)pointer_ns_ind_out = ns_ind_out;
    *(double *)pointer_weight_out   = weight_out;

    pointer_ns_ind_in   += step_ns_ind_in;
    pointer_orbital_ind += step_orbital_ind;
    pointer_Np_in       += step_Np_in;
    pointer_m_in        += step_m_in;
    pointer_ns_ind_out  += step_ns_ind_out;
    pointer_weight_out  += step_weight_out;
  }
}

static PyUFuncGenericFunction funcs[1] = { &creation_operator_ufunc };
static char types[6] = { NPY_LONG, NPY_LONG, NPY_LONG, NPY_LONG, NPY_LONG, NPY_DOUBLE };


static PyMethodDef Methods[] = {
    {"create_bos_ns", create_bos_ns, METH_VARARGS,
     "Make the table of bosonic number states"
     "Args: Np(long>=0), m(long>=0). Returns np.ndarray long."},
    {"create_mapmat_bos", create_mapmat_bos, METH_VARARGS,
     "Create the mapping matrix from numbers states with N-1 to N bosons in m orbitals"
     "Args: Ns1(boson number state table). Returns np.ndarray long."},
    {"configuration_from_state_index", configuration_from_state_index, METH_VARARGS,
     "Get the configuration of a number state from its index in lexicographic order"
     "Args: index(long), Np(long), m(long). Returns np.ndarray long."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_mlx_create_c", "MLX create library C implementation", -1, Methods
};

PyMODINIT_FUNC PyInit__mlx_create_c(void) {
  PyObject *ufunc, *m, *d;
  import_array();
  import_umath();

  m = PyModule_Create(&moduledef);
  if (m == NULL)
    return NULL;

  ufunc = PyUFunc_FromFuncAndData(
      funcs,
      NULL,
      types,
      1,   /* ntypes */
      4,   /* nin */
      2,   /* nout */
      PyUFunc_None,
      "creation_operator",
      "returns the index and weight of a creation operator",
      0
  );

  if (ufunc == NULL) {
      Py_DECREF(m);
      return NULL;
  }

 d = PyModule_GetDict(m);
 if (PyDict_SetItemString(d, "creation_operator", ufunc) < 0) {
     Py_DECREF(ufunc);
     Py_DECREF(m);
     return NULL;
 }
 Py_DECREF(ufunc);

 return m;
}
