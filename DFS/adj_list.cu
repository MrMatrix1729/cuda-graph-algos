#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>


__global__ void dfst(int *adj_list, int *offset, int *visited, int *stack, int *stack_size, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || visited[node]) return;
    
    visited[node] = 1;
    int start = offset[node];
    int end = offset[node + 1];
    
    for (int i = start; i < end; i++) {
        int neighbor = adj_list[i];
        if (!visited[neighbor]) {
            int index = atomicAdd(stack_size, 1);
            stack[index] = neighbor;
        }
    }
}

void computeDFS(int *h_row_ptr, int *h_col_ind, int *h_adj_list, int *h_offset, int num_nodes) {
    int *d_row_ptr, *d_col_ind, *d_adj_list, *d_offset, *d_visited, *d_stack, *d_stack_size;
    int h_stack_size = 0;
    int h_visited[num_nodes] = {0};
    int h_stack[num_nodes];
    
    cudaMalloc(&d_visited, num_nodes * sizeof(int));
    cudaMalloc(&d_stack, num_nodes * sizeof(int));
    cudaMalloc(&d_stack_size, sizeof(int));
    cudaMemcpy(d_visited, h_visited, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_adj_list, h_offset[num_nodes] * sizeof(int));
        cudaMalloc(&d_offset, (num_nodes + 1) * sizeof(int));
        cudaMemcpy(d_adj_list, h_adj_list, h_offset[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        dfs<<<(num_nodes + 255) / 256, 256>>>(d_adj_list, d_offset, d_visited, d_stack, d_stack_size, num_nodes);
        
        cudaFree(d_adj_list);
        cudaFree(d_offset);
    
    cudaMemcpy(h_visited, d_visited, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_visited);
    cudaFree(d_stack);
    cudaFree(d_stack_size);
    
    printf("DFS Traversal Result:\n");
    for (int i = 0; i < num_nodes; i++) {
        if (h_visited[i]) printf("Node %d is visited\n", i);
    }
}

int main() {
    int num_nodes = 5;
    int h_row_ptr[] = {0, 2, 5, 7, 9, 10};
    int h_col_ind[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_adj_list[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_offset[] = {0, 2, 5, 7, 9, 10};
    
    computeDFS(h_row_ptr, h_col_ind, h_adj_list, h_offset, num_nodes);
    
    return 0;
}
