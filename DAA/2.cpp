#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <map>
#include <cmath>
using namespace std;

class Node {
public:
    int freq;
    char symbol;
    Node* left;
    Node* right;
    char huff;

    Node(int freq, char symbol, Node* left = nullptr, Node* right = nullptr)
        : freq(freq), symbol(symbol), left(left), right(right), huff(0) {}

    bool operator<(const Node& other) const {
        return freq > other.freq;
    }
};

void calculateHuffmanCodes(const Node* node, const string& code, map<char, string>& huffmanCodes) {
    if (node) {
        if (!node->left && !node->right) {
            huffmanCodes[node->symbol] = code;
        }
        calculateHuffmanCodes(node->left, code + "0", huffmanCodes);
        calculateHuffmanCodes(node->right, code + "1", huffmanCodes);
    }
}

int main() {
    int n;
    cout << "Enter the number of characters: ";
    cin >> n;

    vector<char> chars(n);
    vector<int> freq(n);

    // Input characters and their frequencies from user
    cout << "Enter characters and their frequencies:\n";
    for (int i = 0; i < n; ++i) {
        cout << "Character " << i + 1 << ": ";
        cin >> chars[i];
        cout << "Frequency of " << chars[i] << ": ";
        cin >> freq[i];
    }

    priority_queue<Node> nodes;

    // Push nodes into the priority queue
    for (size_t i = 0; i < chars.size(); ++i) {
        nodes.push(Node(freq[i], chars[i]));
    }

    // Measure the time for Huffman tree construction
    auto start_time = chrono::high_resolution_clock::now();

    while (nodes.size() > 1) {
        Node* left = new Node(nodes.top());
        nodes.pop();
        Node* right = new Node(nodes.top());
        nodes.pop();
        left->huff = '0';
        right->huff = '1';
        Node* newNode = new Node(left->freq + right->freq, left->symbol + right->symbol, left, right);
        nodes.push(*newNode);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    cout << "Huffman Tree Construction Elapsed Time: " << duration.count() << " microseconds" << endl;

    // Measure the time for Huffman code calculation
    auto code_start_time = chrono::high_resolution_clock::now();

    map<char, string> huffmanCodes;
    calculateHuffmanCodes(&nodes.top(), "", huffmanCodes);

    auto code_end_time = chrono::high_resolution_clock::now();
    auto code_duration = chrono::duration_cast<chrono::microseconds>(code_end_time - code_start_time);
    cout << "Huffman Code Calculation Elapsed Time: " << code_duration.count() << " microseconds" << endl;

    // Calculate space used for the Huffman codes
    auto space_start_time = chrono::high_resolution_clock::now();

    double spaceUsed = 0;
    for (const auto& kv : huffmanCodes) {
        spaceUsed += kv.first * kv.second.length();
    }
    spaceUsed = ceil(spaceUsed / 8); // Convert bits to bytes

    auto space_end_time = chrono::high_resolution_clock::now();
    auto space_duration = chrono::duration_cast<chrono::microseconds>(space_end_time - space_start_time);
    cout << "Estimated Space Used for Huffman Codes: " << spaceUsed << " bytes" << endl;
    cout << "Space Calculation Elapsed Time: " << space_duration.count() << " microseconds" << endl;

    // Output Huffman Codes
    cout << "Huffman Codes:\n";
    for (const auto& kv : huffmanCodes) {
        cout << kv.first << " -> " << kv.second << endl;
    }

    return 0;
}
