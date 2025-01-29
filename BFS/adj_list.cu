#include <cuda_runtime.h>
#include <stdio.h>

// Kernel to perform BFS for a single level
__global__ void bfsKernel(int* graph, int* offsets, int* frontier, int* nextFrontier, int* visited, int* nextFrontierSize, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices || !frontier[tid]) return; // Skip if not in the frontier

    frontier[tid] = 0; // Clear the current frontier for this vertex
    int start = offsets[tid];
    int end = offsets[tid + 1];
    for (int i = start; i < end; i++) {
        int neighbor = graph[i];
        if (atomicExch(&visited[neighbor], 1) == 0) { // Atomically mark as visited
            nextFrontier[neighbor] = 1; // Add neighbor to the next frontier
            atomicAdd(nextFrontierSize, 1); // Increment next frontier size
        }
    }
}

int main() {
    // Example graph definition
    const int numVertices = 5;   // Number of vertices
    const int numEdges = 8;      // Number of edges
    const int source = 0;        // Starting vertex for BFS

    // Graph structure: adjacency list and offsets
    int h_graph[] = {1, 2, 0, 3, 0, 4, 1, 2};  // Flattened adjacency list
    int h_offsets[] = {0, 2, 4, 6, 7, 8};      // Offsets for each vertex

    // Allocate device memory
    int *d_graph, *d_offsets, *d_frontier, *d_nextFrontier, *d_visited, *d_nextFrontierSize;
    cudaMalloc(&d_graph, sizeof(int) * numEdges);
    cudaMalloc(&d_offsets, sizeof(int) * (numVertices + 1));
    cudaMalloc(&d_frontier, sizeof(int) * numVertices);
    cudaMalloc(&d_nextFrontier, sizeof(int) * numVertices);
    cudaMalloc(&d_visited, sizeof(int) * numVertices);
    cudaMalloc(&d_nextFrontierSize, sizeof(int));

    // Copy graph data to device memory
    cudaMemcpy(d_graph, h_graph, sizeof(int) * numEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, sizeof(int) * (numVertices + 1), cudaMemcpyHostToDevice);
    cudaMemset(d_frontier, 0, sizeof(int) * numVertices);
    cudaMemset(d_nextFrontier, 0, sizeof(int) * numVertices);
    cudaMemset(d_visited, 0, sizeof(int) * numVertices);

    // Initialize BFS
    int *h_frontier = (int*)calloc(numVertices, sizeof(int)); // Frontier array on the host
    h_frontier[source] = 1; // Mark the source node in the frontier
    cudaMemcpy(d_frontier, h_frontier, sizeof(int) * numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_visited[source], &h_frontier[source], sizeof(int), cudaMemcpyHostToDevice);

    int nextFrontierSize;
    dim3 blockSize(256);
    dim3 gridSize((numVertices + blockSize.x - 1) / blockSize.x);

    // Perform BFS until no more nodes are in the frontier
    while (true) {
        nextFrontierSize = 0;
        cudaMemcpy(d_nextFrontierSize, &nextFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

        // Launch BFS kernel
        bfsKernel<<<gridSize, blockSize>>>(d_graph, d_offsets, d_frontier, d_nextFrontier, d_visited, d_nextFrontierSize, numVertices);
        cudaDeviceSynchronize();

        // Check if the next frontier is empty
        cudaMemcpy(&nextFrontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        if (nextFrontierSize == 0) break;

        // Update frontier for the next iteration
        cudaMemcpy(d_frontier, d_nextFrontier, sizeof(int) * numVertices, cudaMemcpyDeviceToDevice);
        cudaMemset(d_nextFrontier, 0, sizeof(int) * numVertices);
    }

    // Copy visited array back to host
    int *h_visited = (int*)calloc(numVertices, sizeof(int));
    cudaMemcpy(h_visited, d_visited, sizeof(int) * numVertices, cudaMemcpyDeviceToHost);

    // Print the traversed vertices
    printf("Traversed vertices: ");
    for (int i = 0; i < numVertices; i++) {
        if (h_visited[i]) {
            printf("%d ", i);
        }
    }
    printf("\n");

    // Cleanup
    cudaFree(d_graph);
    cudaFree(d_offsets);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_visited);
    cudaFree(d_nextFrontierSize);
    free(h_frontier);
    free(h_visited);
}


