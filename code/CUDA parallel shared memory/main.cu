#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "../data.h"

#define MAX_NNZ_PER_LINE 64

inline cudaError_t checkCuda(cudaError_t result){
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void kernel( int *csr_rows_d, int *csr_cols_d, int *total_sum_d);

__device__
int compare_blocks( int *cols, int *shared , int n_shared, int start_block_k, int stop_block_k);

void read_from_file_CSR_ROWS( int *rows);
void read_from_file_CSR_COLS( int *cols);

int main(int argc, char const *argv[]) {
  printf("Parallel and Distributed Project 4\n");
  printf("Subject: Graph's Triangle Counting\n");
  printf("Parallel Implementation\n");

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  /*printf("DEVICE NAME %s\n", prop.name);
  printf("SHARED MEMORY PER BLOCK %lu bytes\n", prop.sharedMemPerBlock);
  printf("MAX THREADS PER BLOCK %d\n", prop.maxThreadsPerBlock);
  printf("MAX THREADS DIMENSIONS IN A BLOCK [%d %d %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("MAX BLOCKS DIMENSIONS IN A GRID [%d %d %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("NUMBER OF STREAMING MULTIPROCESSORS %d\n", prop.multiProcessorCount);
  printf("WARP SIZE %d\n", prop.warpSize);*/

  // Initializations host
  int *csr_rows_h = (int*)malloc( (N + 1) * sizeof(int));
  int *csr_cols_h = (int*)malloc( NNZ * sizeof(int));
  int total_sum_h = 0;


  read_from_file_CSR_ROWS(csr_rows_h);
  read_from_file_CSR_COLS(csr_cols_h);

  // Initializations Device
  int *csr_rows_d, *csr_cols_d, *total_sum_d;
  checkCuda(cudaMalloc( &csr_rows_d, (N + 1) * sizeof(int)));
  checkCuda(cudaMalloc( &csr_cols_d, NNZ * sizeof(int)));
  checkCuda(cudaMalloc( &total_sum_d, sizeof(int)));

  checkCuda(cudaMemcpy( csr_rows_d, csr_rows_h, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy( csr_cols_d, csr_cols_h, NNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy( total_sum_d, &total_sum_h, sizeof(int), cudaMemcpyHostToDevice));


  struct timeval start, stop;
  gettimeofday( &start, NULL);

  kernel<<< N, prop.warpSize >>>( csr_rows_d, csr_cols_d, total_sum_d);
  cudaDeviceSynchronize();

  gettimeofday( &stop, NULL);

  double time = ((double)(stop.tv_sec-start.tv_sec)) + ((double)(stop.tv_usec-start.tv_usec))/((double)1000000);
  printf("Time: %lf secs\n", time);

  checkCuda(cudaMemcpy( &total_sum_h, total_sum_d, sizeof(int), cudaMemcpyDeviceToHost));
  printf("TRIANGLES: %d\n", total_sum_h/3);
  cudaFree(csr_rows_d);
  cudaFree(csr_cols_d);
  cudaFree(total_sum_d);
  free(csr_rows_h);
  free(csr_cols_h);
  return 0;
}


__global__
void kernel( int *csr_rows_d, int *csr_cols_d, int *total_sum_d){

  int i, k, j, start_block_i, stop_block_i, start_block_k, stop_block_k, temp, nnz_at_row;

  __shared__ int shared[MAX_NNZ_PER_LINE];

  i = blockIdx.x;                                   // represents row index
  start_block_i = csr_rows_d[i];                    // represents first element index of i_th row at col array.
  stop_block_i  = csr_rows_d[i + 1] - 1;            // represents last element index of i_th row at col array.
  nnz_at_row = stop_block_i - start_block_i + 1;    // Number of Non Zero Elements of i_th row


  // load elements of row i at shared memory.
  __syncthreads();

  j = threadIdx.x;
  while (j < nnz_at_row) {
    shared[j] = csr_cols_d[start_block_i + j];
    j = j + blockDim.x;
  }

  __syncthreads();


  j = start_block_i + threadIdx.x;

  while (j <= stop_block_i){
    temp = 0;
    k = csr_cols_d[j];        // k column, k is the value of element col[j]
    if( k > blockIdx.x){
      start_block_k = csr_rows_d[k];
      stop_block_k  = csr_rows_d[k + 1] - 1;
      temp = compare_blocks( csr_cols_d, &shared[0], nnz_at_row, start_block_k, stop_block_k);
      //printf("BLOCK %d ELEMENT %d SUM %d\n", blockIdx.x, j, sum);
      atomicAdd( total_sum_d, temp);
    }
    j = j + blockDim.x;
    __syncthreads();
  }
}


__device__
int compare_blocks( int *cols, int *shared , int n_shared, int start_block_k, int stop_block_k){
  int i, k;
  int found = 0;

  i = 0;
  k = start_block_k;

  while ( i < n_shared && k <= stop_block_k) {
    if( shared[i] == cols[k]){
      found++;
      i++;
      k++;
    }
    else if( shared[i] < cols[k]){
      i++;
    }
    else{
      k++;
    }
  }

  return found;

}

void read_from_file_CSR_ROWS( int *rows){
  FILE *fp;
  int i = 0;

  fp = fopen( path_csr_rows, "r");
  if (fp == NULL){
    printf("ERROR: FAILed to open file CSR_ROWS.txt\n");
    exit(EXIT_FAILURE);
  }

  while ( fscanf( fp, "%d\n", &rows[i]) != EOF){
    i++;
  }

  if( fclose(fp) != 0){
    printf("ERROR: FAILed to close file CSR_ROWS.txt\n");
    exit(EXIT_FAILURE);
  }

  printf("Elements read from %s: %d\n", path_csr_rows, i);

  return;
}

void read_from_file_CSR_COLS( int *cols){
  FILE *fp;
  int i = 0;

  fp = fopen( path_csr_cols, "r");
  if (fp == NULL){
    printf("ERROR: FAILed to open file CSR_COLS.txt\n");
    exit(EXIT_FAILURE);
  }

  while ( fscanf( fp, "%d\n", &cols[i]) != EOF){
      i++;
  }

  if( fclose(fp) != 0){
    printf("ERROR: FAILed to close file CSR_COLS.txt\n");
    exit(EXIT_FAILURE);
  }

  printf("Elements read from %s: %d\n", path_csr_cols, i);

  return;
}
