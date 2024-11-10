// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    // do w def struct , array, fallback 

    struct Student {
        uint id;
        string name;
        uint age;
        string course;
    }

    Student[] public students;

    // Mapping to track student IDs to ensure uniqueness

    mapping(uint => bool) public studentExists;

    // Fallback function to handle any ETH sent to the contract

    fallback() external payable {
        // You can log a message or handle unexpected ETH transfers here
        revert ("Fallback Called.");

    }
        function addStudent (uint _id, string memory _name, uint _age, string memory _course) public {
        require(!studentExists[_id], "Student ID already exists.");
        students.push(Student(_id, _name, _age, _course));
        studentExists[_id] = true;
    }
    

    function getStudent(uint index) public view returns (uint, string memory, uint, string memory) {
        require(index < students.length, "Invalid index.");
        Student memory student = students[index];
        return (student.id , student.name, student.age, student.course);
    }

    function getStudentCount() public view returns (uint) {
        return students.length;
    }
    
    // Receive ETH
    receive() external payable {}

}