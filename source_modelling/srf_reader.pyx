cimport numpy as np
from libc.stdio cimport *
from libc.stdlib cimport *

from typing import TextIO

import numpy as pnp
import numpy.typing as npt
import scipy as sp
from scipy.sparse import csr_array

np.import_array()

# These types are used in place of double, float, etc because they exactly
# correspond to what numpy uses on whatever system this code runs on.
ctypedef np.double_t DTYPE_t
ctypedef np.int64_t ITYPE_t
DTYPE = pnp.double

# NOTE: sparse matrix structs do not support Python docstrings.
cdef struct sparse_matrix:
    # An intermediate CSR matrix representation used for reading slip values.
    # 
    # The CSR matrix representation is a highly compressed sparse matrix format
    # optimised for row operations and matrix multiplication. See the wikipedia
    # description of the CSR matrix format[0] for a description of how the col_ptr
    # and row_ptr arrays record the indices of the data.
    # 
    # Attributes
    # -----------
    # rows : int
    # Counts the number of rows in the sparse matrix.
    # row_counter : int
    # Records the current row.
    # row_ptr : int*
    # Row index value array.
    # entries : int
    # Counts the total number of entries in the array.
    # col_ptr : int*
    # Column index value array.
    # data : double*
    # Entry value array.
    # 
    # References
    # ----------
    # [0]: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    
    int rows
    int row_counter
    ITYPE_t* row_ptr
    int entries
    ITYPE_t* col_ptr
    DTYPE_t* data
     

cdef void malloc_sparse_matrix(sparse_matrix* slip_matrix, int num_rows):
    """ Initialise an empty sparse matrix. 

    Parameters
    ----------
    slip_matrix : sparse_matrix*
        The slip matrix to initialise.
    num_rows : int
        The total number of rows for the slip matrix.
    """
    slip_matrix.rows = num_rows
    slip_matrix.row_counter = 0
    slip_matrix.row_ptr = <ITYPE_t *>malloc(num_rows * sizeof(ITYPE_t))
    slip_matrix.entries = 0
    slip_matrix.col_ptr = NULL
    slip_matrix.data = NULL

cdef void free_sparse_matrix(sparse_matrix slip_matrix):
    """ Free a sparse matrix. 
    
    Parameters
    ----------
    slip_matrix : sparse_matrix
        The sparse matrix to free.
    """
    free(slip_matrix.row_ptr);
    free(slip_matrix.col_ptr);
    free(slip_matrix.data);

cdef void read_slipt_values(FILE* srf_file, int start_column, int slip_count, sparse_matrix* slip_matrix):
    """ Read slip values from srf_file into the sparse matrix as a new row. 

    Parameters
    ----------
    srf_file : FILE*
        File handle for the srf file.
    start_column : int
        Store the slip values starting at this column.
    slip_count : int
        The number of slip values to read.
    slip_matrix : sparse_matrix*
        The matrix to read slip values into.
    """
    cdef int slip_index
    # The value of row_ptr[i] records the index of the first value of i-th
    # row in the data array. Because we are starting a new row, we thus
    # record it's starting position at the end of the current data array.
    slip_matrix.row_ptr[slip_matrix.row_counter] = slip_matrix.entries
    slip_matrix.row_counter += 1

    if slip_count > 0:
        slip_matrix.entries += slip_count
        # Either allocate or grow the slip matrix row, column and data arrays
        # to fit the new slip values.
        if slip_matrix.col_ptr == NULL:
            slip_matrix.col_ptr = <ITYPE_t*> malloc(slip_matrix.entries * sizeof(ITYPE_t));
        else:
            slip_matrix.col_ptr = <ITYPE_t*>realloc(slip_matrix.col_ptr, slip_matrix.entries  * sizeof(ITYPE_t));

        if slip_matrix.data == NULL:
            slip_matrix.data = <DTYPE_t*>malloc(slip_matrix.entries * sizeof(DTYPE_t));
        else:
            slip_matrix.data = <DTYPE_t*>realloc(slip_matrix.data , slip_matrix.entries * sizeof(DTYPE_t));

        for slip_index in range(slip_matrix.entries - slip_count, slip_matrix.entries):
            # The column ptr records the column for each entry in the data array.
            #           column index        = start column + current offset from start of this row's slip values
            #                                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            slip_matrix.col_ptr[slip_index] = start_column + slip_index - slip_matrix.entries + slip_count
            fscanf(srf_file, "%lf", &slip_matrix.data[slip_index])
    
cdef void read_srf_points_loop(FILE* srf_file, int point_count, DTYPE_t[:, :] metadata, sparse_matrix* slipt1s, sparse_matrix* slipt2s, sparse_matrix* slipt3s):
    """ Loop over the number of points and read in the SRF data from srf_file. 

    Parameters
    ----------
    srf_file : FILE*
        SRF file to read from.
    point_count : int
        Number of points to read.
    metadata : DTYPE_t[:, :]
        Metadata about each point (lat, lon, dt, etc).
    slipt{i}s : sparse_matrix*
        The slip for each point in the i-th component. The matrix is established such that

        slipt{i}s[j, k] = the slip value for point j in component i at time k*dt

        Thus, we assume that dt is constant for each point.
    """
    cdef int nt1
    cdef int nt2
    cdef int nt3
    cdef int i
    cdef int start_column_index
    for i in range(point_count):
        fscanf(srf_file, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %d %lf %d",
               &metadata[i, 0],  # lat
               &metadata[i, 1],  # lon
               &metadata[i, 2],  # dep
               &metadata[i, 3],  # stk
               &metadata[i, 4],  # dip
               &metadata[i, 5],  # area
               &metadata[i, 6],  # tinit
               &metadata[i, 7],  # dt
               &metadata[i, 8],  # rake
               &metadata[i, 9],  # slip1
               &nt1,             # number of slipt1 values for this point
               &metadata[i, 10], # slip2
               &nt2,             # number of slipt2 values for this point
               &metadata[i, 11], # slip3
               &nt3              # number of slipt3 values for this point
               )
        start_column_index = <int> (metadata[i, 6] / metadata[i, 7])
        read_slipt_values(srf_file, start_column_index, nt1, slipt1s)
        read_slipt_values(srf_file, start_column_index, nt2, slipt2s)
        read_slipt_values(srf_file, start_column_index, nt3, slipt3s)

# NOTE: not sure what the return type should be here because it a C function
# returning a Python value. So I have just left it blank.
cdef sparse_matrix_to_csr(sparse_matrix matrix):
    """ Convert the internal sparse matrix representation into a scipy sparse csr array. 

    Parameters
    ----------
    matrix : sparse_matrix
        The matrix to convert.

    Returns
    -------
    csr_array
        The scipy csr_array representation of matrix, or None if the matrix is empty.
    """
    if matrix.entries == 0:
        return None

    
    # Basically, we just have to copy the data from the arrays into numpy arrays.
    # We do this in a loop to ensure that the values are owned by numpy
    # (and hence free'd by python instead of leaking).
    cdef np.ndarray[DTYPE_t, ndim=1] data = pnp.zeros(matrix.entries, dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] col_indices = pnp.zeros(matrix.entries, dtype=pnp.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] row_indices = pnp.zeros(matrix.rows + 1, dtype=pnp.int64)
    cdef int i = 0
    cdef int max_col_ind = 0  

    for i in range(matrix.entries):
        # These "=" assignments are done in Python, not C!
        data[i] = matrix.data[i]
        col_indices[i] = matrix.col_ptr[i]
        if max_col_ind <  matrix.col_ptr[i]:
            # This is a regular C assignment
            max_col_ind = matrix.col_ptr[i]

    i = 0
    for i in range(matrix.rows):
        # Also Python assignment.
        row_indices[i] = matrix.row_ptr[i]
    row_indices[matrix.rows] = matrix.entries
    return sp.sparse.csr_array((data, col_indices, row_indices))
        
        
def read_srf_points(
        srf_file_handle_py: TextIO, 
        point_count: int
    ) -> tuple[npt.NDArray[DTYPE], csr_array, csr_array, csr_array]:
    """ Read point_count SRF points from an SRF file.

    Parameters
    ----------
    srf_file_handle_py : TextIO
        The SRF file to read from.
    point_count : int
        The number of points to read.

    Returns
    -------
    metadata : array
        The metadata for each point.
    slip1ts : sparse matrix
        The sparse matrix of slip values in the first component for each
        point.
    slip2ts : sparse matrix
        The sparse matrix of slip values in the second component for each
        point.
    slip3ts : sparse matrix
        The sparse matrix of slip values in the third component for each
        point.
    """
    cdef np.ndarray[DTYPE_t, ndim=2] metadata = pnp.zeros([point_count, 12], dtype=DTYPE)
    cdef DTYPE_t[:, :] metadata_view = metadata
    
    cdef sparse_matrix slip1ts
    cdef sparse_matrix slip2ts
    cdef sparse_matrix slip3ts
    malloc_sparse_matrix(&slip1ts, point_count)
    malloc_sparse_matrix(&slip2ts, point_count)
    malloc_sparse_matrix(&slip3ts, point_count)

    # This code obtains a C file handle pointing to the same location as
    # the Python file handle.
    cdef FILE* cfile = fdopen(srf_file_handle_py.fileno(), 'rb')
    fseek(cfile, srf_file_handle_py.tell(), SEEK_SET)

    read_srf_points_loop(cfile, point_count, metadata_view, &slip1ts, &slip2ts, &slip3ts)
    slip1ts_mat = sparse_matrix_to_csr(slip1ts)
    slip2ts_mat = sparse_matrix_to_csr(slip2ts)
    slip3ts_mat = sparse_matrix_to_csr(slip3ts)
    
    free_sparse_matrix(slip1ts)
    free_sparse_matrix(slip2ts)
    free_sparse_matrix(slip3ts)
    
    return metadata, slip1ts_mat, slip2ts_mat, slip3ts_mat
