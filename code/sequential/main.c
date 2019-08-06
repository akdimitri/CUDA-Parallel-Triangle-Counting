#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../data.h"




void read_from_file_CSR_ROWS( int *rows);
void read_from_file_CSR_COLS( int *cols);
int compare_blocks( int *cols, int start_block_i, int stop_block_i, int start_block_k, int stop_block_k);

int main(int argc, char const *argv[]) {
  printf("Parallel and Distributed Project 4\n");
  printf("Subject: Graph's Triangle Counting\n");
  printf("Sequential Implementation\n");

  // Initializations
  int *csr_rows = (int*)malloc( (N + 1) * sizeof(int));
  int *csr_cols = (int*)malloc( NNZ * sizeof(int));

  read_from_file_CSR_ROWS(csr_rows);
  read_from_file_CSR_COLS(csr_cols);


  // Sequential Algorithm
  int i, j, k, start_block_i, stop_block_i, start_block_k, stop_block_k, sum;
  struct timeval start, stop;

  sum = 0;

  gettimeofday( &start, NULL);
  for( i = 0; i < N; i++){
    start_block_i = csr_rows[i];            // index position of first element of row i in col array.
    stop_block_i  = csr_rows[i + 1] - 1;    // index position of last  element of row i in col array.

    for( j = start_block_i; j <= stop_block_i; j++){
      k = csr_cols[j];        // k column, k is the value of element col[j]
      if( k > i){
        start_block_k = csr_rows[k];
        stop_block_k  = csr_rows[k + 1] - 1;
        sum = sum + compare_blocks( csr_cols, start_block_i, stop_block_i, start_block_k, stop_block_k);
      }
    }
  }
  gettimeofday( &stop, NULL);

  double time = ((double)(stop.tv_sec-start.tv_sec)) + ((double)(stop.tv_usec-start.tv_usec))/((double)1000000);
  printf("Time: %lf secs\n", time);

  printf("NUMBER OF TRIANGLES: %d\n", sum/3);
  free(csr_rows);
  free(csr_cols);
  return 0;
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

  printf("Elements read from CSR_ROWS.txt %d\n", i);

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

  printf("Elements read from CSR_COLS.txt %d\n", i);

  return;
}

int compare_blocks( int *cols, int start_block_i, int stop_block_i, int start_block_k, int stop_block_k){
  int i, k;
  int found = 0;

  i = start_block_i;
  k = start_block_k;

  while ( i <= stop_block_i && k <= stop_block_k) {
    if( cols[i] == cols[k]){
      found++;
      i++;
      k++;
    }
    else if( cols[i] < cols[k]){
      i++;
    }
    else{
      k++;
    }
  }

  return found;

}
