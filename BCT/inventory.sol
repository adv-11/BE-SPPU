// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ProductInventory {
    
    // Structure to represent a product
    struct Product {
        uint256 productId;
        string name;
        uint256 price; // price in Wei (smallest unit of Ether)
        uint256 quantity;
    }

    // Mapping to store products with their ID as key
    mapping(uint256 => Product) public products;

    // Counter for product IDs
    uint256 public productCount = 0;

    // Event to log when a product is received
    event ProductReceived(uint256 productId, string name, uint256 quantity, uint256 price);

    // Event to log when a product is sold
    event ProductSold(uint256 productId, string name, uint256 quantity, uint256 totalPrice);

    // Function to receive new products or add stock to an existing product
    function receiveProduct(string memory _name, uint256 _price, uint256 _quantity) public {
        require(_quantity > 0, "Quantity must be greater than zero");
        require(_price > 0, "Price must be greater than zero");

        // Create new product or update existing product
        productCount++;
        products[productCount] = Product(productCount, _name, _price, _quantity);

        // Emit an event for product received
        emit ProductReceived(productCount, _name, _quantity, _price);
    }

    // Function to sell a product
    function sellProduct(uint256 _productId, uint256 _quantity) public payable {
        require(_quantity > 0, "Quantity must be greater than zero");
        Product storage product = products[_productId];
        require(product.quantity >= _quantity, "Not enough stock available");
        require(msg.value >= _quantity * product.price, "Insufficient payment");

        // Deduct the quantity sold
        product.quantity -= _quantity;

        // Emit an event for product sold
        emit ProductSold(_productId, product.name, _quantity, msg.value);
    }

    // Function to display stock of a specific product
    function displayStock(uint256 _productId) public view returns (string memory name, uint256 price, uint256 quantity) {
        Product memory product = products[_productId];
        return (product.name, product.price, product.quantity);
    }

    // Function to display all available products and their stock
    function displayAllProducts() public view returns (Product[] memory) {
        Product[] memory allProducts = new Product[](productCount);
        for (uint256 i = 1; i <= productCount; i++) {
            allProducts[i - 1] = products[i];
        }
        return allProducts;
    }
}
