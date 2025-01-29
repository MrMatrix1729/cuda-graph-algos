%%cuda
//TC adj list

#include <vector>
#include <iostream>

__global__ void countTriangles(int *adjList, int *adjIndex, int numNodes, int *triangleCounts) {
    int u = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one node

    if (u < numNodes) {
        int triangleCount = 0;

        // Iterate over neighbors of u
        for (int i = adjIndex[u]; i < adjIndex[u + 1]; i++) {
            int v = adjList[i]; // Neighbor of u

            // Iterate over neighbors of v
            for (int j = adjIndex[v]; j < adjIndex[v + 1]; j++) {
                int w = adjList[j]; // Neighbor of v

                // Check if w is also a neighbor of u
                for (int k = adjIndex[u]; k < adjIndex[u + 1]; k++) {
                    if (adjList[k] == w) {
                        triangleCount++;
                        break;
                    }
                }
            }
        }

        // Each triangle is counted 3 times, divide by 3
        triangleCounts[u] = triangleCount / 3;
    }
}


void triangleCount(const std::vector<int> &adjList, const std::vector<int> &adjIndex, int numNodes) {
    int *d_adjList, *d_adjIndex, *d_triangleCounts;
    int *h_triangleCounts = new int[numNodes];

    // Allocate device memory
    cudaMalloc(&d_adjList, adjList.size() * sizeof(int));
    cudaMalloc(&d_adjIndex, adjIndex.size() * sizeof(int));
    cudaMalloc(&d_triangleCounts, numNodes * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_adjList, adjList.data(), adjList.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjIndex, adjIndex.data(), adjIndex.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;
    countTriangles<<<gridSize, blockSize>>>(d_adjList, d_adjIndex, numNodes, d_triangleCounts);

    // Copy results back to host
    cudaMemcpy(h_triangleCounts, d_triangleCounts, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    int totalTriangles = 0;
    for (int i = 0; i < numNodes; i++) {
        totalTriangles += h_triangleCounts[i];
    }
    std::cout << "Total Triangles: " << totalTriangles << std::endl;

    // Free memory
    delete[] h_triangleCounts;
    cudaFree(d_adjList);
    cudaFree(d_adjIndex);
    cudaFree(d_triangleCounts);
}




int main() {
    // Define adjacency list representation
    int numNodes = 4;
    std::vector<int> adjList = {1, 3, 0, 3, 2, 3};  // Flattened adjacency list
    std::vector<int> adjIndex = {0, 2, 4, 5, 6};    // Start index of neighbors for each node

    // Call triangle counting function
    triangleCount(adjList, adjIndex, numNodes);

    return 0;
}
