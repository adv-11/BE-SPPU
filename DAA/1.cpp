#include <iostream>
#include <ctime> // For measuring time
using namespace std;

// Iterative Fibonacci
int fib_iterative(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Recursive Fibonacci
int fib_recursive(int n) {
    if (n <= 1) return n;
    return fib_recursive(n - 1) + fib_recursive(n - 2);
}

int main() {
    int n;
    cout << "Enter a number: ";
    cin >> n;

    // Measure time for iterative Fibonacci
    clock_t start_iter = clock();
    int result_iter = fib_iterative(n);
    clock_t end_iter = clock();
    double duration_iter = double(end_iter - start_iter) / CLOCKS_PER_SEC;

    // Calculate space required for iterative approach
    int space_iter = sizeof(n) + sizeof(int) * 3;  // n, a, b, c variables

    cout << "Iterative Fibonacci of " << n << " is: " << result_iter << endl;
    cout << "Time required (iterative): " << duration_iter << " seconds" << endl;
    cout << "Space required (iterative): " << space_iter << " bytes" << endl;

    // Measure time for recursive Fibonacci
    clock_t start_rec = clock();
    int result_rec = fib_recursive(n);
    clock_t end_rec = clock();
    double duration_rec = double(end_rec - start_rec) / CLOCKS_PER_SEC;

    // Calculate space required for recursive approach
    // Recursion uses a stack frame for each call, with `sizeof(int)` for each level
    int space_rec = sizeof(n) * (n > 1 ? n : 1);

    cout << "Recursive Fibonacci of " << n << " is: " << result_rec << endl;
    cout << "Time required (recursive): " << duration_rec << " seconds" << endl;
    cout << "Space required (recursive): " << space_rec << " bytes" << endl;

    return 0;
}

