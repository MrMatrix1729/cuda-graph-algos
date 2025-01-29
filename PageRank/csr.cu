%%cuda

//pagerank csr
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-6

// Kernel to calculate the new PageRank values using CSR
__global__ void calculatePageRankCSR(float *d_pageRank, float *d_newPageRank, int *d_rowOffsets, int *d_colIndices, float *d_outDegree, int numNodes, float dampingFactor) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < numNodes) {
        float rank = 0.0f;

        // Accumulate contributions from incoming neighbors
        for (int edge = d_rowOffsets[node]; edge < d_rowOffsets[node + 1]; edge++) {
            int neighbor = d_colIndices[edge];
            rank += d_pageRank[neighbor] / d_outDegree[neighbor];
        }

        // Apply the damping factor and teleportation
        d_newPageRank[node] = dampingFactor * rank + (1.0f - dampingFactor) / numNodes;
    }
}

// Kernel to check for convergence
__global__ void checkConvergence(float *d_pageRank, float *d_newPageRank, float *d_diff, int numNodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < numNodes) {
        float diff = fabsf(d_newPageRank[node] - d_pageRank[node]);
        atomicAdd(d_diff, diff);
    }
}

// Main function
void pagerankCSR(const std::vector<int> &rowOffsets, const std::vector<int> &colIndices, const std::vector<float> &outDegree, int numNodes, float dampingFactor, int maxIterations) {
    // Initialize host and device memory
    float *h_pageRank = new float[numNodes];
    float *h_newPageRank = new float[numNodes];
    float *d_pageRank, *d_newPageRank, *d_diff;
    int *d_rowOffsets, *d_colIndices;
    float *d_outDegree;

    // Initialize PageRank values
    for (int i = 0; i < numNodes; i++) {
        h_pageRank[i] = 1.0f / numNodes;
    }

    // Allocate device memory
    cudaMalloc(&d_pageRank, numNodes * sizeof(float));
    cudaMalloc(&d_newPageRank, numNodes * sizeof(float));
    cudaMalloc(&d_rowOffsets, (numNodes + 1) * sizeof(int));
    cudaMalloc(&d_colIndices, colIndices.size() * sizeof(int));
    cudaMalloc(&d_outDegree, numNodes * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_pageRank, h_pageRank, numNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowOffsets, rowOffsets.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndices, colIndices.data(), colIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outDegree, outDegree.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < maxIterations; iter++) {
        // Reset convergence diff
        cudaMemset(d_diff, 0, sizeof(float));

        // Calculate new PageRank values
        calculatePageRankCSR<<<numBlocks, BLOCK_SIZE>>>(d_pageRank, d_newPageRank, d_rowOffsets, d_colIndices, d_outDegree, numNodes, dampingFactor);

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
    cudaMemcpy(h_pageRank, d_pageRank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    for (int i = 0; i < numNodes; i++) {
        std::cout << "Node " << i << ": " << h_pageRank[i] << std::endl;
    }

    // Clean up
    cudaFree(d_pageRank);
    cudaFree(d_newPageRank);
    cudaFree(d_rowOffsets);
    cudaFree(d_colIndices);
    cudaFree(d_outDegree);
    cudaFree(d_diff);
    delete[] h_pageRank;
    delete[] h_newPageRank;
}

int main() {
    // Example graph represented in CSR format
    int numNodes = 4;
    std::vector<int> rowOffsets = {0, 2, 4, 5, 6};
    std::vector<int> colIndices = {1, 2, 0, 3, 1, 3};
    std::vector<float> outDegree = {2.0, 2.0, 1.0, 1.0}; // Number of outgoing edges per node

    float dampingFactor = 0.85f;
    int maxIterations = 100;

    pagerankCSR(rowOffsets, colIndices, outDegree, numNodes, dampingFactor, maxIterations);

    return 0;
}
