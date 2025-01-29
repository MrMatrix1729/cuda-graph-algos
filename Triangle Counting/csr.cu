%%cuda
//TC CSR

__device__ bool isNeighbor(int *colIndices, int start, int end, int node) {
    // Binary search for node in colIndices[start:end]
    while (start < end) {
        int mid = (start + end) / 2;
        if (colIndices[mid] == node) return true;
        if (colIndices[mid] < node) start = mid + 1;
        else end = mid;
    }
    return false;
}

__global__ void countTrianglesCSR(int *rowOffsets, int *colIndices, int numNodes, int *triangleCounts) {
    int u = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one node

    if (u < numNodes) {
        int triangleCount = 0;

        // Iterate over neighbors of u
        for (int i = rowOffsets[u]; i < rowOffsets[u + 1]; i++) {
            int v = colIndices[i]; // Neighbor of u

            // Iterate over neighbors of v
            for (int j = rowOffsets[v]; j < rowOffsets[v + 1]; j++) {
                int w = colIndices[j]; // Neighbor of v

                // Check if w is also a neighbor of u
                if (isNeighbor(colIndices, rowOffsets[u], rowOffsets[u + 1], w)) {
                    triangleCount++;
                }
            }
        }

        // Each triangle is counted 3 times, divide by 3
        triangleCounts[u] = triangleCount / 3;
    }
}
