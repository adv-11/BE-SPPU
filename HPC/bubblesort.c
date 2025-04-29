gcc -f openmp filename.c -o filename
./ filename

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void bubbleSortOddEvenParallel(int arr[], int n) {
    int isSorted = 0;
    while (!isSorted) {
        isSorted = 1;

        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }
    }
}

void bubbleSortOddEvenSequential(int arr[], int n) {
    int isSorted = 0;
    while (!isSorted) {
        isSorted = 1;

        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }

        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }
    }
}

void generateRandomArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int main() {
    srand(time(0));
    omp_set_num_threads(omp_get_max_threads()); // Use all available threads

    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *arr1 = (int *)malloc(n * sizeof(int));
    int *arr2 = (int *)malloc(n * sizeof(int));

    printf("+-----------+------------------------------+------------------------------+------------+-------------+\n");
    printf("| Iteration | Odd-Even Bubble Sort Seq Time | Odd-Even Bubble Sort Par Time | Speedup    | Efficiency  |\n");
    printf("+-----------+------------------------------+------------------------------+------------+-------------+\n");

    for (int i = 0; i < 5; i++) {
        generateRandomArray(arr1, n);
        for (int j = 0; j < n; j++) {
            arr2[j] = arr1[j];
        }

        double start_time, end_time;

        // Odd-Even Bubble Sort Sequential
        start_time = omp_get_wtime();
        bubbleSortOddEvenSequential(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSeqTime = end_time - start_time;

        // Odd-Even Bubble Sort Parallel
        start_time = omp_get_wtime();
        bubbleSortOddEvenParallel(arr2, n);
        end_time = omp_get_wtime();
        double bubbleParTime = end_time - start_time;

        double bubbleSpeedup = bubbleSeqTime / bubbleParTime;
        double efficiency = bubbleSpeedup / omp_get_max_threads();

        printf("| %9d | %28f | %28f | %10f | %11f |\n",
               i + 1, bubbleSeqTime, bubbleParTime, bubbleSpeedup, efficiency);
    }

    printf("+-----------+------------------------------+------------------------------+------------+-------------+\n");

    free(arr1);
    free(arr2);
    return 0;
}