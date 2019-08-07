/*  $ nvcc main.cu tools.cu -lcusparse -lcudart -o test      *
*                                                           *
*  $ ./test [N] [nnzA]                                      */

#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include "tools.h"
#include <sys/time.h>
#include <assert.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed with error (%d) at line %d\n",                 \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(int argc, char const *argv[]) {

  /* Initialize Host Memory */
  int N = atoi(argv[1]);      // Matrix size N*N
  int nnzA = atoi(argv[2]);  // A number of non zero elements
      // A matrix
  int *hA_rows = (int*)malloc(nnzA*sizeof(int));
  int *hA_cols = (int*)malloc(nnzA*sizeof(int));
  float *hA_values = (float*)malloc(nnzA*sizeof(float));

  readSparseMatrixCOO( hA_rows, hA_cols, hA_values, nnzA, "./MatrixACOO-R-C-V.txt");
  //printf("Matrix A: COO Format\n");
  //printSparseMatrixCOO( hA_rows, hA_cols, hA_values, nnzA);
  //printf("\n");


  /* Initialize Device Memory */
    // A matrix
  int *dA_rows, *dA_cols;
  float *dA_values;
  CHECK_CUDA(cudaMalloc( &dA_rows, nnzA*sizeof(int)));
  CHECK_CUDA(cudaMalloc( &dA_cols, nnzA*sizeof(int)));
  CHECK_CUDA(cudaMalloc( &dA_values, nnzA*sizeof(float)));
  CHECK_CUDA(cudaMemcpy( dA_rows, hA_rows, nnzA*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy( dA_cols, hA_cols, nnzA*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy( dA_values, hA_values, nnzA*sizeof(float), cudaMemcpyHostToDevice));

  /* Initialize CUSPARSE library */
  cusparseHandle_t handle = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  /* A matrix: Convert COO Format to CSR */
  int *dA_csr_offsets;
  int *hA_csr_offsets;
  hA_csr_offsets = (int*)malloc((N+1)*sizeof(int));
  CHECK_CUDA(cudaMalloc( &dA_csr_offsets, (N+1)*sizeof(int)));
  CHECK_CUSPARSE(cusparseXcoo2csr( handle, dA_rows, nnzA, N, dA_csr_offsets, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUDA(cudaMemcpy( hA_csr_offsets, dA_csr_offsets, (N+1)*sizeof(int), cudaMemcpyDeviceToHost));
  /*printf("\nMatrix A: CSR Format\n");
  printSparseMatrixCSR( hA_csr_offsets, hA_cols, hA_values, nnzA, N);
  */

  /* Create Sparse Matrix A Struct */
  cusparseMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matA));
  cusparseSetMatType( matA, CUSPARSE_MATRIX_TYPE_GENERAL);  //unnecessary
  cusparseSetMatIndexBase( matA, CUSPARSE_INDEX_BASE_ZERO); //unnecessary

  /* Create Sparse Matrix C Struct */
  cusparseMatDescr_t matC;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matC));
  cusparseSetMatType( matC, CUSPARSE_MATRIX_TYPE_GENERAL);  //unnecessary
  cusparseSetMatIndexBase( matC, CUSPARSE_INDEX_BASE_ZERO); //unnecessary

  /* Create Sparse Matrix D Struct */
  cusparseMatDescr_t matD;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matD));
  cusparseSetMatType( matD, CUSPARSE_MATRIX_TYPE_GENERAL);  //unnecessary
  cusparseSetMatIndexBase( matD, CUSPARSE_INDEX_BASE_ZERO); //unnecessary

  /* Perform SpMat*SpMat operation */
  struct timeval startwtime, endwtime;  // variables to hold execution time
  int baseC, nnzC;
  csrgemm2Info_t info = NULL;
  size_t bufferSize;
  void *buffer = NULL;
  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzC;
  float alpha = 1.0;
  //float beta  = 0;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  // step 1: create an opaque structure
  cusparseCreateCsrgemm2Info(&info);

  gettimeofday (&startwtime, NULL);

  // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
  cusparseScsrgemm2_bufferSizeExt( handle, N, N, N,
                                &alpha,
                                matA, nnzA, dA_csr_offsets, dA_cols,
                                matA, nnzA, dA_csr_offsets, dA_cols,
                                NULL,
                                matD, 0, NULL, NULL,
                                info,
                                &bufferSize);

  CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

  // step 3: Initialize Matrix C
  int *dC_csr_offsets, *dC_cols, *hC_csr_offsets, *hC_cols;
  float *dC_values, *hC_values;

  // step 4: compute dC_csr_offsets
  CHECK_CUDA(cudaMalloc( &dC_csr_offsets, (N+1)*sizeof(int)));
  cusparseXcsrgemm2Nnz(handle, N, N, N,
        matA, nnzA, dA_csr_offsets, dA_cols,
        matA, nnzA, dA_csr_offsets, dA_cols,
        matD, 0, NULL, NULL,
        matC, dC_csr_offsets, nnzTotalDevHostPtr,
        info, buffer );
  if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
  }else{
    CHECK_CUDA(cudaMemcpy(&nnzC, dC_csr_offsets+N, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&baseC, dC_csr_offsets, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }

  // step 5: finish sparsity pattern and value of C
  cudaMalloc((void**)&dC_cols, sizeof(int)*nnzC);
  cudaMalloc((void**)&dC_values, sizeof(float)*nnzC);

  cusparseScsrgemm2( handle, N, N, N, &alpha,
                  matA, nnzA, dA_values, dA_csr_offsets, dA_cols,
                  matA, nnzA, dA_values, dA_csr_offsets, dA_cols,
                  NULL,
                  matD, 0, NULL, NULL, NULL,
                  matC, dC_values, dC_csr_offsets, dC_cols,
                  info,
                  buffer);



  // step 5: destroy the opaque structure
  cusparseDestroyCsrgemm2Info(info);

  /* Empty Space from GPU and CPU*/
  cudaFree(buffer);

  /* C matrix: Convert CSR Format to COO */
  int *hC_rows;
  int *dC_rows;
  hC_rows = (int*)malloc(nnzC*sizeof(int));
  CHECK_CUDA(cudaMalloc( &dC_rows, nnzC*sizeof(int)));
  cusparseXcsr2coo(handle, dC_csr_offsets, nnzC, N, dC_rows, CUSPARSE_INDEX_BASE_ZERO);

  /* Copy C to Host */
  hC_rows = (int*)malloc(nnzC*sizeof(int));
  hC_cols = (int*)malloc(nnzC*sizeof(int));
  hC_csr_offsets = (int*)malloc((N+1)*sizeof(int));
  hC_values = (float*)malloc(nnzC*sizeof(float));
  cudaMemcpy( hC_rows, dC_rows, nnzC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( hC_cols, dC_cols, nnzC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( hC_csr_offsets, dC_csr_offsets, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( hC_values, dC_values, nnzC*sizeof(float), cudaMemcpyDeviceToHost);

  /* Calculate triangles */
  int triangles = 0;
  int i = 0, k = 0;

  while( (i < nnzA) && (k < nnzC)){
    if( (hA_rows[i] == hC_rows[k]) && (hA_cols[i] == hC_cols[k])){
      triangles += hC_values[k];
      i++;
      k++;
    }
    else{
      if     ( hA_rows[i] < hC_rows[k]){
        i++;
      }
      else if( hA_rows[i] > hC_rows[k]){
        k++;
      }
      else if( hA_cols[i] < hC_cols[k]){
        i++;
      }
      else if( hA_cols[i] > hC_cols[k]){
        k++;
      }
    }
  }

  gettimeofday (&endwtime, NULL);

  /* get time in seconds */
	double time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);


  /*printf("\nMatrix C:\n");
  printSparseMatrixCSR( hC_csr_offsets, hC_cols, hC_values, nnzC, N);
  printSparseMatrixCOO( hC_rows, hC_cols, hC_values, nnzC);
  */

  /* Test Results */
  /*int nnzM = atoi(argv[3]);
  int *Matlab_rows = (int*)malloc(nnzM*sizeof(int));
  int *Matlab_cols = (int*)malloc(nnzM*sizeof(int));
  float *Matlab_values = (float*)malloc(nnzM*sizeof(float));
  readSparseMatrixCOO( Matlab_rows,  Matlab_cols, Matlab_values, nnzM, "../test/MatrixCCOO-R-C-V.txt");

  int pass = test( hC_rows, hC_cols, hC_values, nnzC, Matlab_rows, Matlab_cols, Matlab_values, nnzM);
  printf("Multiplication TEST %s\n",(pass) ? "PASSed" : "FAILed");
  assert(pass);*/

  /* print execution time */
  printf("Sequential wall clock time: %f sec\n", time);
  printf("Triangles: %d\n", triangles);

  /* Clean Up */
  cudaFree(dA_rows);    // dA_rows no longer needed
  free(hA_rows);        // hA_rows no longer needed
  free(hC_cols);
  free(hC_csr_offsets);
  free(hC_rows);
  free(hC_values);
  free(hA_csr_offsets);
  free(hA_cols);
  free(hA_values);
  cudaFree(dA_csr_offsets);
  cudaFree(dA_cols);
  cudaFree(dA_values);
  cudaFree(dC_cols);
  cudaFree(dC_rows);
  cudaFree(dC_csr_offsets);
  cudaFree(dC_values);


  return 0;
}
