
%%cuda
// pagerank AdjList

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-6

// Kernel to calculate new PageRank values using adjacency list
__global__ void calculatePageRank(float *d_pageRank, float *d_newPageRank, int *d_adjList, int *d_adjIndex, float *d_outDegree, int numNodes, float dampingFactor) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < numNodes) {
        float rank = 0.0f;

        // Sum contributions from incoming neighbors
        for (int edge = d_adjIndex[node]; edge < d_adjIndex[node + 1]; edge++) {
            int neighbor = d_adjList[edge];
            rank += d_pageRank[neighbor] / d_outDegree[neighbor];
        }

        // Apply damping factor and teleportation
        d_newPageRank[node] = dampingFactor * rank + (1.0f - dampingFactor) / numNodes;
    }
}

// Kernel to check convergence
__global__ void checkConvergence(float *d_pageRank, float *d_newPageRank, float *d_diff, int numNodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < numNodes) {
        float diff = fabsf(d_newPageRank[node] - d_pageRank[node]);
        atomicAdd(d_diff, diff);
    }
}

// Host function for PageRank
void pagerank(const std::vector<int> &adjList, const std::vector<int> &adjIndex, const std::vector<float> &outDegree, int numNodes, float dampingFactor, int maxIterations) {
    // Allocate host memory for PageRank
    std::vector<float> h_pageRank(numNodes, 1.0f / numNodes);
    std::vector<float> h_newPageRank(numNodes, 0.0f);

    // Allocate device memory
    float *d_pageRank, *d_newPageRank, *d_diff;
    int *d_adjList, *d_adjIndex;
    float *d_outDegree;

    cudaMalloc(&d_pageRank, numNodes * sizeof(float));
    cudaMalloc(&d_newPageRank, numNodes * sizeof(float));
    cudaMalloc(&d_adjList, adjList.size() * sizeof(int));
    cudaMalloc(&d_adjIndex, adjIndex.size() * sizeof(int));
    cudaMalloc(&d_outDegree, numNodes * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_pageRank, h_pageRank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjList, adjList.data(), adjList.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjIndex, adjIndex.data(), adjIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outDegree, outDegree.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < maxIterations; iter++) {
        // Reset convergence diff
        cudaMemset(d_diff, 0, sizeof(float));

        // Calculate new PageRank values
        calculatePageRank<<<numBlocks, BLOCK_SIZE>>>(d_pageRank, d_newPageRank, d_adjList, d_adjIndex, d_outDegree, numNodes, dampingFactor);

        // Check for convergence
        checkConvergence<<<numBlocks, BLOCK_SIZE>>>(d_pageRank, d_newPageRank, d_diff, numNodes);

        // Copy convergence diff back to host
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        // Swap pointers for the next iteration
        std::swap(d_pageRank, d_newPageRank);

        // Check if converged
        if (h_diff < EPSILON) {
            break;
        }
    }

    // Copy final PageRank values back to host
    cudaMemcpy(h_pageRank.data(), d_pageRank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    for (int i = 0; i < numNodes; i++) {
        std::cout << "Node " << i << ": " << h_pageRank[i] << std::endl;
    }

    // Clean up
    cudaFree(d_pageRank);
    cudaFree(d_newPageRank);
    cudaFree(d_adjList);
    cudaFree(d_adjIndex);
    cudaFree(d_outDegree);
    cudaFree(d_diff);
}

int main() {
    // Example graph
    int numNodes = 4;
    std::vector<int> adjList = {1, 2, 0, 3, 1, 3};
    std::vector<int> adjIndex = {0, 2, 4, 5, 6};
    std::vector<float> outDegree = {2.0, 2.0, 1.0, 1.0};

    float dampingFactor = 0.85f;
    int maxIterations = 100;

    pagerank(adjList, adjIndex, outDegree, numNodes, dampingFactor, maxIterations);

    return 0;
}

