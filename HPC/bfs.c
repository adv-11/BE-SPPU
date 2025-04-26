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

// Sequential BFS (non-parallel)
void sequentialBFS(struct Graph* g, int start) {
    int V = g->V;
    int visited[V];
    for (int i = 0; i < V; i++) visited[i] = 0;  // Mark all vertices as unvisited

    visited[start] = 1;  // Mark the start node as visited
    int* q = (int*)malloc(MAX_VERTICES * sizeof(int));  // Dynamic queue
    int front = 0, rear = 0;
    q[rear++] = start;  // Enqueue the start node

    printf("BFS traversal starting from node %d (Sequential): ", start);

    while (front != rear) {
        int node = q[front++];  // Dequeue a node

        // Print the node (this is the BFS visit)
        printf("%d ", node);

        // Visit all the neighbors of the current node
        for (int i = 0; i < V; i++) {
            if (g->adj[node][i] && !visited[i]) {  // If there's an edge and the neighbor is not visited
                visited[i] = 1;  // Mark the neighbor as visited
                q[rear++] = i;   // Enqueue the neighbor
            }
        }
    }

    printf("\n");
    free(q);  // Free the dynamically allocated memory for the queue
}

// Optimized Parallel BFS using OpenMP with thread-local queues and task parallelism
void parallelBFS(struct Graph* g, int start) {
    int V = g->V;
    int* visited = (int*)malloc(V * sizeof(int));  // Dynamic array for visited
    for (int i = 0; i < V; i++) visited[i] = 0;  // Mark all vertices as unvisited

    visited[start] = 1;  // Mark the start node as visited
    int* q = (int*)malloc(MAX_VERTICES * sizeof(int));  // Dynamic queue
    int front = 0, rear = 0;
    q[rear++] = start;  // Enqueue the start node

    printf("BFS traversal starting from node %d (Parallel): ", start);

    while (front != rear) {
        int localFront = front;  // Local front for each thread
        int localRear = rear;    // Local rear for each thread

        // Parallel section: each thread processes a part of the queue
        #pragma omp parallel
        {
            int* thread_queue = (int*)malloc(MAX_VERTICES * sizeof(int)); // Thread-local queue
            int thread_rear = 0;

            // Parallel section: visit nodes in the current frontier
            #pragma omp for
            for (int i = localFront; i < localRear; i++) {
                int node = q[i];  // Get the node to process

                // Visit all neighbors of the current node
                for (int j = 0; j < V; j++) {
                    if (g->adj[node][j] && !visited[j]) {  // If there's an edge and the neighbor is not visited
                        visited[j] = 1;  // Mark the neighbor as visited

                        // Add the neighbor to the thread's local queue
                        thread_queue[thread_rear++] = j;
                    }
                }
            }

            // Merge the thread's queue into the global queue (done atomically at the end of each iteration)
            #pragma omp critical
            {
                for (int i = 0; i < thread_rear; i++) {
                    q[rear++] = thread_queue[i]; // Enqueue all new neighbors
                }
            }

            free(thread_queue); // Free thread-local queue memory
        }

        // Update front and rear for the next round
        front = localRear;
    }

    printf("\n");
    free(q);  // Free the dynamically allocated memory for the queue
    free(visited);  // Free the dynamically allocated memory for visited array
}

int main() {
    int i, j;
    double inputs[5][6];  // 5 sets: vertices, edges, start vertex, sequential time, parallel time, parallel speedup
    int u, v, start;

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

        srand(time(NULL) + i);
        generateRandomEdges(&g, max_edges);

        // Take input for the start vertex for BFS
        printf("Enter the starting vertex for BFS for graph %d: ", i + 1);
        scanf("%d", &start);
        if (start < 0 || start >= (int)inputs[i][0]) {
            printf("Invalid starting vertex. Please enter a vertex between 0 and %d.\n", (int)inputs[i][0] - 1);
            return 1;
        }

        clock_t start_time, end_time;
        double execution_time;

        // Measure execution time for sequential BFS
        start_time = clock();
        sequentialBFS(&g, start);
        end_time = clock();
        execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        inputs[i][3] = execution_time;  // Sequential time

        // Measure execution time for parallel BFS
        double parallel_start_time = omp_get_wtime();
        parallelBFS(&g, start);
        double parallel_end_time = omp_get_wtime();
        execution_time = parallel_end_time - parallel_start_time;
        inputs[i][4] = execution_time;  // Parallel time

        // Calculate speedup
        if (inputs[i][3] > 0) {
            inputs[i][5] = inputs[i][3] / inputs[i][4];  // Speedup = sequential time / parallel time
        }

        freeGraph(&g);  // Free the memory allocated for the graph
    }

    // Display the results
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

