#include <iostream>
#include <algorithm>

using namespace std;

struct Item {
    int weight;
    int value;
};

// Comparator function to sort items by value-to-weight ratio
bool compare(Item a, Item b) {
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2;
}

double fractionalKnapsack(int W, Item items[], int n) {
    // Sort items by value-to-weight ratio
    sort(items, items + n, compare);

    double maxValue = 0.0;

    for (int i = 0; i < n; i++) {
        if (W >= items[i].weight) {
            // Take the whole item
            W -= items[i].weight;
            maxValue += items[i].value;
        } else {
            // Take the fraction of the item
            maxValue += items[i].value * ((double)W / items[i].weight);
            break;
        }
    }

    return maxValue;
}

int main() {
    int W, n;

    cout << "Enter the capacity of the knapsack: ";
    cin >> W;

    cout << "Enter the number of items: ";
    cin >> n;

    Item items[n]; // Array to store items

    // Input weights and values for each item
    for (int i = 0; i < n; i++) {
        cout << "Enter weight and value for item " << i + 1 << ": ";
        cin >> items[i].weight >> items[i].value;
    }

    double maxValue = fractionalKnapsack(W, items, n);

    cout << "Maximum value in Knapsack = " << maxValue << endl;

    return 0;
}