#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>


__global__ void prims(int *adj_list, int *offset, int *weights, int *mst_edges, int *visited, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || visited[node]) return;
    
    visited[node] = 1;
    int start = offset[node];
    int end = offset[node + 1];
    
    int min_weight = INT_MAX;
    int min_neighbor = -1;
    
    for (int i = start; i < end; i++) {
        int neighbor = adj_list[i];
        int weight = weights[i];
        if (!visited[neighbor] && weight < min_weight) {
            min_weight = weight;
            min_neighbor = neighbor;
        }
    }
    
    if (min_neighbor != -1) {
        mst_edges[node] = min_neighbor;
    }
}

void computeMST(int *h_row_ptr, int *h_col_ind, int *h_weights, int *h_adj_list, int *h_offset, int num_nodes, bool use_csr) {
    int *d_row_ptr, *d_col_ind, *d_weights, *d_adj_list, *d_offset, *d_visited, *d_mst_edges;
    int h_visited[num_nodes] = {0};
    int h_mst_edges[num_nodes];
    
    cudaMalloc(&d_visited, num_nodes * sizeof(int));
    cudaMalloc(&d_mst_edges, num_nodes * sizeof(int));
    cudaMemcpy(d_visited, h_visited, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_adj_list, h_offset[num_nodes] * sizeof(int));
        cudaMalloc(&d_offset, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_weights, h_offset[num_nodes] * sizeof(int));
        cudaMemcpy(d_adj_list, h_adj_list, h_offset[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights, h_offset[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
        
        prims<<<(num_nodes + 255) / 256, 256>>>(d_adj_list, d_offset, d_weights, d_mst_edges, d_visited, num_nodes);
        
        cudaFree(d_adj_list);
        cudaFree(d_offset);
        cudaFree(d_weights);

    
    cudaMemcpy(h_mst_edges, d_mst_edges, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_visited);
    cudaFree(d_mst_edges);
    
    printf("MST Result:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d is connected to %d in MST\n", i, h_mst_edges[i]);
    }
}

int main() {
    int num_nodes = 5;
    int h_row_ptr[] = {0, 2, 5, 7, 9, 10};
    int h_col_ind[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_weights[] = {2, 3, 2, 4, 6, 4, 5, 1, 2, 7};
    int h_adj_list[] = {1, 4, 0, 2, 4, 1, 3, 2, 4, 3};
    int h_offset[] = {0, 2, 5, 7, 9, 10};
    

    
    computeMST(h_row_ptr, h_col_ind, h_weights, h_adj_list, h_offset, num_nodes, false);
    
    return 0;
}
