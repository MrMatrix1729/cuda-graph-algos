%%cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKS 1023
#define THREADS 1023

__global__ void bfsKernel(
    int *v_adj_list, int *v_adj_begin, int *v_adj_length,
    int *result, bool *still_running, int num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_vertices) {
        printf("Processing vertex %d with distance %d\n", tid, result[tid]);

        for (int n = 0; n < v_adj_length[tid]; n++) {
            int neighbor = v_adj_list[v_adj_begin[tid] + n];
            printf("Checking neighbor %d\n", neighbor);

            if (result[neighbor] > result[tid] + 1) {
                printf("Updating distance for vertex %d\n", neighbor);
                result[neighbor] = result[tid] + 1;
                *still_running = true;
            }
        }
    }
}

int main() {
    // Example graph definition
    const int num_vertices = 5;   // Number of vertices
    const int num_edges = 8;      // Number of edges
    const int source = 0;         // Starting vertex

    // Host graph representation
    int h_v_adj_list[] = {1, 2, 0, 3, 0, 4, 1, 2};  // Adjacency list
    int h_v_adj_begin[] = {0, 2, 4, 6, 7};          // Start indices
    int h_v_adj_length[] = {2, 2, 2, 1, 1};         // Number of neighbors per vertex

    // Result array
    int *result = (int*)malloc(num_vertices * sizeof(int));
    for (int i = 0; i < num_vertices; i++) {
        result[i] = (i == source) ? 0 : INT_MAX; // Initialize distances
    }

    // Device memory allocation
    int *d_v_adj_list, *d_v_adj_begin, *d_v_adj_length, *d_result;
    bool *d_still_running, h_still_running = true, h_false_value = false;

    cudaMalloc(&d_v_adj_list, num_edges * sizeof(int));
    cudaMalloc(&d_v_adj_begin, num_vertices * sizeof(int));
    cudaMalloc(&d_v_adj_length, num_vertices * sizeof(int));
    cudaMalloc(&d_result, num_vertices * sizeof(int));
    cudaMalloc(&d_still_running, sizeof(bool));

    // Copy data to the device
    cudaMemcpy(d_v_adj_list, h_v_adj_list, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_begin, h_v_adj_begin, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_adj_length, h_v_adj_length, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    // BFS Loop
    while (h_still_running) {
        cudaMemcpy(d_still_running, &h_false_value, sizeof(bool), cudaMemcpyHostToDevice);

        bfsKernel<<<BLOCKS, THREADS>>>(
            d_v_adj_list, d_v_adj_begin, d_v_adj_length, d_result, d_still_running, num_vertices
        );
        cudaMemcpy(&h_still_running, d_still_running, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Copy the result back to the host
    cudaMemcpy(result, d_result, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Shortest distances from source %d:\n", source);
    for (int i = 0; i < num_vertices; i++) {
        printf("Vertex %d: %d\n", i, (result[i] == INT_MAX) ? -1 : result[i]);
    }

    // Cleanup
    free(result);
    cudaFree(d_v_adj_list);
    cudaFree(d_v_adj_begin);
    cudaFree(d_v_adj_length);
    cudaFree(d_result);
    cudaFree(d_still_running);
}
