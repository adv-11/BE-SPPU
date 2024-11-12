#include <iostream>
#include <ctime>  // For clock() to measure execution time

using namespace std;

#define MAX 10 // Define a maximum board size (change as needed)

bool isSafe(int board[MAX][MAX], int row, int col, int n) {
    // Check this row on the left side
    for (int i = 0; i < col; i++) {
        if (board[row][i]) return false;
    }

    // Check upper diagonal on the left side
    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j]) return false;
    }

    // Check lower diagonal on the left side
    for (int i = row, j = col; i < n && j >= 0; i++, j--) {
        if (board[i][j]) return false;
    }

    return true;
}

bool solveNQueensUtil(int board[MAX][MAX], int col, int n) {
    // If all queens are placed
    if (col >= n) return true;

    // Try placing this queen in all rows one by one
    for (int i = 0; i < n; i++) {
        if (isSafe(board, i, col, n)) {
            // Place queen
            board[i][col] = 1;

            // Recur to place the rest
            if (solveNQueensUtil(board, col + 1, n)) return true;

            // Backtrack if placing queen doesn't lead to a solution
            board[i][col] = 0;
        }
    }

    return false;
}

void solveNQueens(int n) {
    int board[MAX][MAX] = {0};  // Initialize the board with all 0s

    if (solveNQueensUtil(board, 0, n)) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << (board[i][j] ? "Q " : ". ");
            }
            cout << endl;
        }
    } else {
        cout << "Solution does not exist" << endl;
    }
}

int main() {
    int n;
    cout << "Enter the value of N: ";
    cin >> n;

    if (n > MAX) {
        cout << "N is too large. Maximum supported value is " << MAX << endl;
        return 1;
    }

    clock_t start = clock();  // Start measuring time

    solveNQueens(n);

    clock_t end = clock();  // End measuring time
    double time_taken = double(end - start) / CLOCKS_PER_SEC;

    // Time and space complexity
    cout << "Time Complexity: O(" << n << "!)" << endl;
    cout << "Space Complexity: O(" << n << "^2)" << endl;
    cout << "Execution time: " << time_taken << " seconds" << endl;

    return 0;
}