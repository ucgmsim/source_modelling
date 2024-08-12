cimport numpy as np
import numpy as pnp
import scipy as sp
from libc.stdio cimport *
from libc.stdlib cimport *
np.import_array()

cdef struct sparse_matrix:
     int rows
     int row_counter
     int* row_ptr
     int entries
     int* col_ptr
     double* data


     
ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t
DTYPE = pnp.double


cdef void malloc_sparse_matrix(sparse_matrix* slip_matrix, int num_rows):
    slip_matrix.rows = num_rows
    slip_matrix.row_counter = 0
    slip_matrix.row_ptr = <int *>malloc(num_rows * sizeof(int))
    slip_matrix.entries = 0
    slip_matrix.col_ptr = NULL
    slip_matrix.data = NULL

cdef void free_sparse_matrix(sparse_matrix slip_matrix):
    free(slip_matrix.row_ptr);
    free(slip_matrix.col_ptr);
    free(slip_matrix.data);

cdef void read_slipt_values(FILE* srf_file, int start_column, int slip_count, sparse_matrix* slip_matrix):
    cdef int slip_index
    cdef void* intermediate
    slip_matrix.row_ptr[slip_matrix.row_counter] = slip_matrix.entries
    slip_matrix.row_counter += 1
    if slip_count > 0:
        # printf("%d\n", slip_count);
        slip_matrix.entries += slip_count
        # printf("Entries: %d\n", slip_matrix.entries)
        if slip_matrix.col_ptr == NULL:
            slip_matrix.col_ptr = <int*> malloc(slip_matrix.entries * sizeof(int));
        else:
            # printf("Column PTR: %p\n", slip_matrix.col_ptr);
            # printf("Can Deref? %d\n", slip_matrix.col_ptr[0]);
            # printf("New Size: %d\n", slip_matrix.entries * sizeof(int))
            slip_matrix.col_ptr = <int*>realloc(slip_matrix.col_ptr, slip_matrix.entries  * sizeof(int));
        if slip_matrix.data == NULL:
            slip_matrix.data = <double*>malloc(slip_matrix.entries * sizeof(double));
        else:
            # printf("Column PTR: %p\n", slip_matrix.data);
            # printf("New Size: %d\n", slip_matrix.entries * sizeof(int))
            slip_matrix.data = <double*>realloc(slip_matrix.data , slip_matrix.entries * sizeof(double));
        for slip_index in range(slip_matrix.entries - slip_count, slip_matrix.entries):
            # printf("Storing %d at %d\n", start_column + slip_index - slip_matrix.entries + slip_count, slip_index)
            slip_matrix.col_ptr[slip_index] = start_column + slip_index - slip_matrix.entries + slip_count
            fscanf(srf_file, "%lf", &slip_matrix.data[slip_index])
        


cdef void read_srf_points_loop(FILE* srf_file, int point_count, DTYPE_t[:, :] metadata, sparse_matrix* slipt1s, sparse_matrix* slipt2s, sparse_matrix* slipt3s):
    cdef int nt1
    cdef int nt2
    cdef int nt3
    cdef int i
    cdef int start_column_index
    for i in range(point_count):
        fscanf(srf_file, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %d %lf %d",
               &metadata[i, 0], # lat
               &metadata[i, 1], # lon
               &metadata[i, 2], # dep
               &metadata[i, 3], # stk
               &metadata[i, 4], # dip
               &metadata[i, 5], # area
               &metadata[i, 6], # tinit
               &metadata[i, 7], # dt
               &metadata[i, 8], # rake
               &metadata[i, 9], # slip1
               &nt1,
               &metadata[i, 10], # slip2
               &nt2,
               &metadata[i, 11], # slip3
               &nt3
               )
        start_column_index = <int> (metadata[i, 6] / metadata[i, 7])
        read_slipt_values(srf_file, start_column_index, nt1, slipt1s)
        read_slipt_values(srf_file, start_column_index, nt2, slipt2s)
        read_slipt_values(srf_file, start_column_index, nt3, slipt3s)

cdef sparse_matrix_to_csr(sparse_matrix matrix):
    if matrix.entries == 0:
        return None
    cdef np.ndarray[DTYPE_t, ndim=1] data = pnp.zeros(matrix.entries, dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] col_indices = pnp.zeros(matrix.entries, dtype=pnp.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] row_indices = pnp.zeros(matrix.rows, dtype=pnp.int64)
    cdef int i = 0
    
    for i in range(matrix.entries):
        data[i] = matrix.data[i]
        col_indices[i] = matrix.col_ptr[i]
    i = 0
    for i in range(matrix.rows):
        row_indices[i] = matrix.row_ptr[i]
    
    return sp.sparse.csr_matrix((data, col_indices, row_indices))
        
    
        
        
def read_srf_points(srf_file_handle_py, point_count):
    cdef np.ndarray[DTYPE_t, ndim=2] metadata = pnp.zeros([point_count, 12], dtype=DTYPE)
    cdef DTYPE_t[:, :] metadata_view = metadata
    
    cdef sparse_matrix slip1ts
    cdef sparse_matrix slip2ts
    cdef sparse_matrix slip3ts
    malloc_sparse_matrix(&slip1ts, point_count)
    malloc_sparse_matrix(&slip2ts, point_count)
    malloc_sparse_matrix(&slip3ts, point_count)
    
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
