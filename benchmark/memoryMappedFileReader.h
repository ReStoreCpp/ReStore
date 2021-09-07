#include "restore/helpers.hpp"
#include <bits/stdint-uintn.h>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct MemoryMappedFile {
    int    fd;
    size_t currentPos;
    size_t size;
    char*  data;

    MemoryMappedFile(const std::string& filename) {
        currentPos = 0;
        fd         = open(filename.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "Failed opening file " << filename << std::endl;
            exit(1);
        }
        struct stat fileStat;
        if (fstat(fd, &fileStat) == -1) {
            std::cerr << "Error getting file stats" << std::endl;
            close(fd);
            exit(1);
        }
        size = static_cast<size_t>(fileStat.st_size);

        data = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) {
            std::cerr << "Failed mapping memory for file" << std::endl;
            close(fd);
            exit(1);
        }
    }

    ~MemoryMappedFile() {
        if (munmap(data, size) == -1) {
            std::cerr << "Failed unmapping memory mapped file" << std::endl;
            close(fd);
            exit(1);
        }
        close(fd);
    }

    bool isValidPosition() const {
        return currentPos < size;
    }

    char getCurrentSymbol() const {
        return data[currentPos];
    }

    void next() {
        ++currentPos;
    }
};

struct MemoryMappedFileReader {
    MemoryMappedFile file;

    MemoryMappedFileReader(const std::string& fileName) : file(fileName) {}

    void skipSpaces() {
        while (file.isValidPosition() && file.getCurrentSymbol() == ' ') {
            file.next();
        }
    }

    void skipNewline() {
        assert(file.isValidPosition() && file.getCurrentSymbol() == '\n');
        file.next();
    }

    void skipLine() {
        while (file.isValidPosition() && file.getCurrentSymbol() != '\n') {
            file.next();
        }
        if (file.isValidPosition()) {
            skipNewline();
        }
    }

    char getLetter() {
        assert(file.isValidPosition());
        char letter = file.getCurrentSymbol();
        if (!std::isalpha(letter)) {
            std::cerr << "Expected letter but found " << letter << std::endl;
        }
        file.next();
        skipSpaces();
        return letter;
    }

    bool isLetter() {
        return std::isalpha(file.getCurrentSymbol());
    }

    int getInt() {
        int number = 0;
        assert(file.isValidPosition());
        if (!std::isdigit(file.getCurrentSymbol())) {
            std::cerr << "Expected digit but found " << file.getCurrentSymbol() << std::endl;
        }
        while (file.isValidPosition() && std::isdigit(file.getCurrentSymbol())) {
            const int digit = file.getCurrentSymbol() - '0';
            number *= 10;
            number += digit;
            file.next();
        }
        skipSpaces();
        return number;
    }

    uint64_t getuint64_t() {
        uint64_t number = 0;
        assert(file.isValidPosition());
        if (!std::isdigit(file.getCurrentSymbol())) {
            std::cerr << "Expected digit but found " << file.getCurrentSymbol() << std::endl;
        }
        while (file.isValidPosition() && std::isdigit(file.getCurrentSymbol())) {
            const int digit = file.getCurrentSymbol() - '0';
            number *= 10;
            number += asserting_cast<unsigned>(digit);
            file.next();
        }
        skipSpaces();
        return number;
    }

    bool finishedFile() {
        return !file.isValidPosition();
    }
};
