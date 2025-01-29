#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

__global__ void kCore(int *row_ptr, int *col_ind, int *degrees, int k, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    if (degrees[node] < k) {
        degrees[node] = 0;
        int row_start = row_ptr[node];
        int row_end = row_ptr[node + 1];
        for (int i = row_start; i < row_end; i++) {
            atomicSub(&degrees[col_ind[i]], 1);
        }
    }
}

void computeKCore(int *h_row_ptr, int *h_col_ind, int *h_adj_list, int *h_offset, int *h_degrees, int num_nodes, int k) {
    int *d_row_ptr, *d_col_ind, *d_adj_list, *d_offset, *d_degrees;
    cudaMalloc(&d_degrees, num_nodes * sizeof(int));
    cudaMemcpy(d_degrees, h_degrees, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    
        cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_col_ind, h_row_ptr[num_nodes] * sizeof(int));
        cudaMemcpy(d_row_ptr, h_row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind, h_row_ptr[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
        
        kCore<<<num_nodes, 256>>>(d_row_ptr, d_col_ind, d_degrees, k, num_nodes);
        
        cudaFree(d_row_ptr);
        cudaFree(d_col_ind);
    
    
    cudaMemcpy(h_degrees, d_degrees, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_degrees);
}

int main() {
    int num_nodes = 5;
    int k = 2;
    int h_row_ptr[] = {0, 2, 5, 7, 9, 10};
    int h_col_ind[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_adj_list[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_offset[] = {0, 2, 5, 7, 9, 10};
    int h_degrees[] = {2, 3, 2, 2, 3};
    
    computeKCore(h_row_ptr, h_col_ind, h_adj_list, h_offset, h_degrees, num_nodes, k);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d -> Degree %d\n", i, h_degrees[i]);
    }
    

    
    return 0;
}
