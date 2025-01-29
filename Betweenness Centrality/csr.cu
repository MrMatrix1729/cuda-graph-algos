#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

__global__ void betweennessCentrality(int *row_ptr, int *col_ind, float *bc, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    float centrality = 0.0f;
    int row_start = row_ptr[node];
    int row_end = row_ptr[node + 1];
    
    for (int i = row_start; i < row_end; i++) {
        int neighbor = col_ind[i];
        centrality += 1.0f; 
    }
    
    bc[node] = centrality;
}


void computeBetweennessCentrality(int *h_row_ptr, int *h_col_ind, int *h_adj_list, int *h_offset, float *h_bc, int num_nodes, bool use_csr) {
    int *d_row_ptr, *d_col_ind, *d_adj_list, *d_offset;
    float *d_bc;
    cudaMalloc(&d_bc, num_nodes * sizeof(float));
    cudaMemset(d_bc, 0, num_nodes * sizeof(float));
    
    if (use_csr) {
        cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_col_ind, h_row_ptr[num_nodes] * sizeof(int));
        cudaMemcpy(d_row_ptr, h_row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind, h_row_ptr[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
        
        betweennessCentrality<<<num_nodes, 256>>>(d_row_ptr, d_col_ind, d_bc, num_nodes);
        
        cudaFree(d_row_ptr);
        cudaFree(d_col_ind);

    
    cudaMemcpy(h_bc, d_bc, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_bc);
}

int main() {
    int num_nodes = 5;
    int h_row_ptr[] = {0, 2, 5, 7, 9, 10};
    int h_col_ind[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_adj_list[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_offset[] = {0, 2, 5, 7, 9, 10};
    float h_bc[5] = {0};
    
    computeBetweennessCentrality(h_row_ptr, h_col_ind, h_adj_list, h_offset, h_bc, num_nodes, true);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d -> Betweenness Centrality %f\n", i, h_bc[i]);
    }
    

    
    return 0;
}
