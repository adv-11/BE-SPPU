#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void mergeSortSequential(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, left, mid);
            
            #pragma omp section
            mergeSortParallel(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

void bubbleSortOddEvenParallel(int arr[], int n) {
    int temp;

    // Parallelize Odd Phase
    #pragma omp parallel for private(temp)
    for (int i = 0; i < n / 2; i++) {
        for (int j = 1 + i; j < n - i - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    // Parallelize Even Phase
    #pragma omp parallel for private(temp)
    for (int i = 0; i < n / 2; i++) {
        for (int j = i; j < n - i - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void generateRandomArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int main() {
    srand(time(0));

    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *arr1 = (int *)malloc(n * sizeof(int));
    int *arr2 = (int *)malloc(n * sizeof(int));
    int *arr3 = (int *)malloc(n * sizeof(int));

    // Generate random numbers for the array
    generateRandomArray(arr1, n);
    for (int i = 0; i < n; i++) {
        arr2[i] = arr1[i];
        arr3[i] = arr1[i];
    }
    printf("Prathmesh Kulkarni 41036 BE A");
    printf("\n+----------------------+------------------------+------------------------+------------------+------------------+\n");
    printf("| Iteration | Odd-Even Bubble Sort Seq Time | Odd-Even Bubble Sort Par Time | Merge Sort Seq Time | Merge Sort Par Time | Speedup | Efficiency |\n");
    printf("+----------------------+------------------------+------------------------+------------------+------------------+\n");

    // Measure performance of sequential algorithms and parallel algorithms for 5 observations
    for (int i = 0; i < 5; i++) {
        // Sequential Odd-Even Bubble Sort
        double start_time = omp_get_wtime();
        bubbleSortOddEvenParallel(arr1, n);
        double end_time = omp_get_wtime();
        double bubbleSortSeqTime = end_time - start_time;

        // Parallel Odd-Even Bubble Sort
        start_time = omp_get_wtime();
        bubbleSortOddEvenParallel(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSortParTime = end_time - start_time;

        // Sequential Merge Sort or Insertion Sort based on size
        start_time = omp_get_wtime();
        if (n < 20) {
            insertionSort(arr2, n);  // Use Insertion Sort if n is less than 20
        } else {
            mergeSortSequential(arr2, 0, n - 1);  // Otherwise, use Merge Sort
        }
        end_time = omp_get_wtime();
        double mergeSortSeqTime = end_time - start_time;

        // Parallel Merge Sort or Insertion Sort based on size
        start_time = omp_get_wtime();
        if (n < 20) {
            insertionSort(arr3, n);  // Use Insertion Sort if n is less than 20
        } else {
            mergeSortParallel(arr3, 0, n - 1);  // Otherwise, use Parallel Merge Sort
        }
        end_time = omp_get_wtime();
        double mergeSortParTime = end_time - start_time;

        // Calculate Speedup and Efficiency
        double bubbleSortSpeedup = bubbleSortSeqTime / bubbleSortParTime;
        double mergeSortSpeedup = mergeSortSeqTime / mergeSortParTime;
        double efficiency = (bubbleSortSpeedup + mergeSortSpeedup) / 6;

        // Print results in table format
        printf("| %9d | %22f | %22f | %16f | %16f | %8f | %10f |\n", 
               i + 1, bubbleSortSeqTime, bubbleSortParTime, mergeSortSeqTime, mergeSortParTime, 
               bubbleSortSpeedup + mergeSortSpeedup, efficiency);
    }

    printf("+----------------------+------------------------+------------------------+------------------+------------------+\n");

    free(arr1);
    free(arr2);
    free(arr3);

    return 0;
}

