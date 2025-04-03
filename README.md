# CUDA Graph Algorithms
## Repository Structure

The code is organized by algorithm, with each algorithm having both CSR and adjacency list implementations (where applicable):

```
Betweenness Centrality/
    adj_list.cu
    csr.cu
BFS/
    adj_list_first.cu
    adj_list.cu
DFS/
    adj_list.cu
    csr.cu
Graph Colouring/
    adj_list.cu
    csr.cu
K-Core/
    adj_list.cu
    csr.cu
MST/
    prim_adj_list.cu
    prim_csr.cu
PageRank/
    adj_list.cu
    csr.cu
SSSP/
    adj_list.cu
Triangle Counting/
    adj_list.cu
    csr.cu
```

## Algorithms Implemented

1. **Betweenness Centrality** - Measures the centrality of vertices in a graph based on shortest paths
2. **BFS (Breadth-First Search)** - Graph traversal algorithm that explores neighbor vertices before moving to next level
3. **DFS (Depth-First Search)** - Graph traversal algorithm that explores as far as possible along branches
4. **Graph Coloring** - Assigns colors to vertices such that no adjacent vertices have the same color
5. **K-Core** - Finds maximal subgraphs where all vertices have at least k connections
6. **MST (Minimum Spanning Tree)** - Finds a subset of edges that forms a tree including all vertices with minimum weight
7. **PageRank** - Algorithm to rank vertices in a graph based on the structure of incoming links
8. **SSSP (Single-Source Shortest Path)** - Finds shortest paths from a source vertex to all other vertices
9. **Triangle Counting** - Counts the number of triangles in a graph

## Graph Representations

### Compressed Sparse Row (CSR)
CSR is a memory-efficient representation for sparse graphs that uses three arrays:
- `row_ptr`: Stores indices in the `col_ind` array that indicate where each vertex's adjacency list begins
- `col_ind`: Stores the destination vertices of each edge

### Adjacency List
The adjacency list representation in these implementations uses:
- `adj_list`: Flattened list of neighbor vertices
- `offset`: Array that stores the starting index of each vertex's neighbors in the `adj_list`

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (compatible with the code, recommended 10.0+)
- C++ compiler compatible with your CUDA installation

## Compilation

To compile any of the algorithm implementations, use nvcc:

```bash
nvcc -o algorithm_name algorithm_path.cu
```

For example:
```bash
nvcc -o bfs BFS/adj_list.cu
```

## Usage

After compilation, run the executable:

```bash
./algorithm_name
```

Most implementations include a small test graph in the main function to demonstrate functionality.

## Example

The following example shows how to compile and run the BFS implementation:

```bash
nvcc -o bfs BFS/adj_list.cu
./bfs
```

## Notes

- The implementations use simple test graphs defined in the main functions
- Modify the graph definitions in the main functions to test with your own graphs

## Performance Considerations

- CSR implementations typically offer better memory efficiency
- Adjacency list implementations may provide more intuitive code structure
- Performance characteristics depend on the specific graph structure and algorithm