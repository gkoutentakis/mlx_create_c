#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include "mlx_create_c.h"

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
  double* map = (double*) PyArray_DATA((PyArrayObject*)out);

  // Call the C function
  Py_BEGIN_ALLOW_THREADS;
  create_mapmat_bos_core(map, Ns1, bas_par);
  Py_END_ALLOW_THREADS;

  Py_DECREF(input_array);

  return out;
}

static PyMethodDef Methods[] = {
    {"create_bos_ns", create_bos_ns, METH_VARARGS,
     "Make the table of bosonic number states"
     "Args: Np(int>=0), m(float>=0). Returns np.ndarray long."},
    {"create_mapmat_bos", create_mapmat_bos, METH_VARARGS,
     "Create the mapping matrix from numbers states with N-1 to N bosons in m orbitals"
     "Args: Ns1(boson number state table). Returns np.ndarray long."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_mlx_create_c", "MLX create library C implementation", -1, Methods
};

PyMODINIT_FUNC PyInit__mlx_create_c(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
