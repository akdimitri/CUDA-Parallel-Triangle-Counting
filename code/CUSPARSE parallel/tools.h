#ifndef TOOLS_H_   /* Include guard */
#define TOOLS_H_
void readSparseMatrixCOO( int *rows, int *cols, float *values, int nnz, const char *filename);
void printSparseMatrixCOO( int *rows, int *cols, float *values, int nnz);
void printSparseMatrixCSR( int *offsets, int *cols, float *values, int nnz, int N);
void readVector( float *Vector, int N, const char *filename);
void printVector( float *Vector, int N);
int test( int *A_rows, int *A_cols, float *A_values, int nnzA, int *B_rows, int *B_cols, float *B_values, int nnzB);
#endif // TOOLS_H_
