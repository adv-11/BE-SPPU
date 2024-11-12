#include <iostream>
#include <vector>
using namespace std;

// Function to solve 0-1 Knapsack problem using dynamic programming
int knapsackDP(int capacity, vector<int>& weights, vector<int>& values, int n) {
    // Create a 2D vector to store the maximum value for each subproblem
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));

    // Build the DP table
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    // The last cell of the table contains the maximum value for the knapsack
    return dp[n][capacity];
}

int main() {
    int n; // Number of items
    int capacity; // Capacity of the knapsack

    // User input for the number of items
    cout << "Enter the number of items: ";
    cin >> n;

    vector<int> values(n), weights(n);

    // Input the values and weights of the items
    for (int i = 0; i < n; i++) {
        cout << "Enter value and weight of item " << i + 1 << ": ";
        cin >> values[i] >> weights[i];
    }

    // Input the capacity of the knapsack
    cout << "Enter the capacity of the knapsack: ";
    cin >> capacity;

    // Solve the 0-1 Knapsack problem
    int maxValue = knapsackDP(capacity, weights, values, n);

    // Output the result
    cout << "Maximum value that can be carried in the knapsack: " << maxValue << endl;

    return 0;
}