#include <stdio.h>

void readSparseMatrixCOO( int *rows, int *cols, float *values, int nnz, const char *filename){
  int i;
  FILE *fid;
  fid = fopen( filename, "r");

  for( i = 0; i < nnz; i++){
    fscanf( fid, "%d %d %f\n", &rows[i], &cols[i], &values[i]);
  }

  fclose(fid);
}

void printSparseMatrixCOO( int *rows, int *cols, float *values, int nnz){
  int i;
  printf("Rows\tCols\tValues\n");
  for( i = 0; i < nnz; i++)
    printf("%d\t%d\t%.1f\n", rows[i], cols[i], values[i]);
}

void printSparseMatrixCSR( int *offsets, int *cols, float *values, int nnz, int N){
  int i;
  printf("Offsets\tCols\tValues\n");
  for( i = 0; i < nnz; i++){
    if( i < N)
      printf("%d\t%d\t%.1f\n", offsets[i], cols[i], values[i]);
    else
      printf("  \t%d\t%.1f\n", cols[i], values[i]);
  }
}


void readVector( float *Vector, int N, const char *filename){
  int i;
  FILE *fid;
  fid = fopen( filename, "r");

  for( i = 0; i < N; i++){
    fscanf( fid, "%f\n", &Vector[i]);
  }

  fclose(fid);
}

void printVector( float *Vector, int N){
  int i;
  printf("Vector\n");
  for( i = 0; i < N; i++)
    printf("%.1f\n", Vector[i]);
  printf("\n");
}

int test( int *A_rows, int *A_cols, float *A_values, int nnzA, int *B_rows, int *B_cols, float *B_values, int nnzB){
  int i;
  if( nnzA != nnzB)
    return 0;

  for( i = 0; i < nnzA; i++)
    if( A_rows[i] != B_rows[i] || A_cols[i] != B_cols[i] || A_values[i] != B_values[i])
      return 0;

  return 1;
}
