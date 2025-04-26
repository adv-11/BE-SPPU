#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_VERTICES 10000   // Increase the number of vertices to 10,000

// Graph structure
struct Graph {
    int V;
    int** adj;  // Dynamic adjacency matrix
};

// Function to initialize graph
void initGraph(struct Graph* g, int vertices) {
    g->V = vertices;
    
    // Dynamically allocate memory for adjacency matrix
    g->adj = (int**)malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        g->adj[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; j++) {
            g->adj[i][j] = 0;
        }
    }
}

// Function to free dynamically allocated memory for the graph
void freeGraph(struct Graph* g) {
    for (int i = 0; i < g->V; i++) {
        free(g->adj[i]);
    }
    free(g->adj);
}

// Function to add edge to graph
void addEdge(struct Graph* g, int u, int v) {
    g->adj[u][v] = 1;
    g->adj[v][u] = 1;  // Undirected graph
}

// Function to generate random edges for large graphs
void generateRandomEdges(struct Graph* g, int max_edges) {
    int edges = 0;
    while (edges < max_edges) {
        int u = rand() % g->V;
        int v = rand() % g->V;
        if (u != v && g->adj[u][v] == 0) {  // Avoid self-loops and duplicate edges
            addEdge(g, u, v);
            edges++;
        }
    }
}

// Sequential DFS (non-parallel)
void sequentialDFS(struct Graph* g, int start, int* visited) {
    visited[start] = 1;    // Mark the current node as visited

    // Visit all the neighbors of the current node
    for (int i = 0; i < g->V; i++) {
        if (g->adj[start][i] && !visited[i]) {  // If there's an edge and the neighbor is not visited
            sequentialDFS(g, i, visited);  // Recursively visit the neighbor
        }
    }
}

// Parallel DFS using OpenMP (with task parallelism)
void parallelDFS(struct Graph* g, int start, int* visited) {
    visited[start] = 1;    // Mark the current node as visited

    // Parallel section: visit all the neighbors of the current node
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < g->V; i++) {
                if (g->adj[start][i] && !visited[i]) {  // If there's an edge and the neighbor is not visited
                    #pragma omp task
                    parallelDFS(g, i, visited);  // Recursively visit the neighbor in parallel
                }
            }
        }
    }
}

int main() {
    int i, j;
    double inputs[5][6];  // 5 sets: vertices, edges, start vertex, sequential time, parallel time, parallel speedup
    int u, v, start;

    srand(time(NULL));  // Set random seed once to ensure randomness

    // Take input for 5 different cases
    for (i = 0; i < 5; i++) {
        printf("Enter the number of vertices for graph %d (1 to 10000): ", i + 1);
        scanf("%lf", &inputs[i][0]);
        if (inputs[i][0] < 1 || inputs[i][0] > 10000) {
            printf("Number of vertices must be between 1 and 10000.\n");
            return 1;
        }

        int max_edges = (int)(inputs[i][0] * (inputs[i][0] - 1) / 2);
        printf("Automatically generating up to %d edges for graph %d.\n", max_edges, i + 1);

        struct Graph g;
        initGraph(&g, (int)inputs[i][0]);

        generateRandomEdges(&g, max_edges);

        // Take input for the start vertex for DFS
        printf("Enter the starting vertex for DFS for graph %d: ", i + 1);
        scanf("%d", &start);
        if (start < 0 || start >= (int)inputs[i][0]) {
            printf("Invalid starting vertex. Please enter a vertex between 0 and %d.\n", (int)inputs[i][0] - 1);
            return 1;
        }

        clock_t start_time, end_time;
        double execution_time;

        // Measure execution time for sequential DFS
        int* visited = (int*)malloc((int)inputs[i][0] * sizeof(int));
        for (int j = 0; j < (int)inputs[i][0]; j++) visited[j] = 0;

        start_time = clock();
        printf("DFS traversal starting from node %d (Sequential): ", start);
        sequentialDFS(&g, start, visited);
        printf("\n");
        end_time = clock();
        execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        inputs[i][3] = execution_time;  // Sequential time

        // Measure execution time for parallel DFS
        for (int j = 0; j < (int)inputs[i][0]; j++) visited[j] = 0;  // Reset visited array

        double parallel_start_time = omp_get_wtime();
        printf("DFS traversal starting from node %d (Parallel): ", start);
        parallelDFS(&g, start, visited);
        printf("\n");
        double parallel_end_time = omp_get_wtime();
        execution_time = parallel_end_time - parallel_start_time;
        inputs[i][4] = execution_time;  // Parallel time

        // Calculate speedup
        if (inputs[i][3] > 0) {
            inputs[i][5] = inputs[i][3] / inputs[i][4];  // Speedup = sequential time / parallel time
        }

        free(visited);  // Free the memory allocated for visited array
        freeGraph(&g);  // Free the memory allocated for the graph
    }

    // Display the results
    printf("Dhruv Sawant - 41055 BE-A");
    printf("\n------------------------------------------------------------\n");
    printf("| Graph No | Vertices | Sequential Time | Parallel Time | Speedup |\n");
    printf("------------------------------------------------------------\n");

    for (i = 0; i < 5; i++) {
        printf("| %d        | %.0lf       | %.6f          | %.6f        | %.2f   |\n", 
                i + 1, inputs[i][0], inputs[i][3], inputs[i][4], inputs[i][5]);
    }
    printf("------------------------------------------------------------\n");

    return 0;
}

