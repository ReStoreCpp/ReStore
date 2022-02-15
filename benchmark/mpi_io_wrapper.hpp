// This file is part of ReStore.
//
// Copyright 2021 Tim Niklas Uhl
//
// ReStore is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. ReStore is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with ReStore.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <cstddef>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpi_io {

struct MPIException : public std::exception {
    MPIException(const std::string& msg) : msg_() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::stringstream out;
        out << "[R" << rank << "] " << msg;
        msg_ = out.str();
    }
    const char* what() const throw() override {
        return msg_.c_str();
    }

    private:
    std::string msg_;
};

inline void check_mpi_error(int errcode, const std::string& file, int line) {
    if (errcode != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> buf;
        int                                    resultlen;
        MPI_Error_string(errcode, buf.data(), &resultlen);
        std::string msg(buf.begin(), buf.begin() + resultlen);
        msg = msg + " in " + file + ":" + std::to_string(line);
        throw MPIException(msg);
    }
}


class ConcurrentFile {
    public:
    enum class AccessMode { ReadOnly, WriteOnly, ReadAndWrite, Create };

    private:
    static int mpi_amode(ConcurrentFile::AccessMode access_mode) {
        using AccessMode = ConcurrentFile::AccessMode;
        switch (access_mode) {
            case AccessMode::ReadOnly:
                return MPI_MODE_RDONLY;
            case AccessMode::WriteOnly:
                return MPI_MODE_WRONLY;
            case AccessMode::ReadAndWrite:
                return MPI_MODE_RDWR;
            case AccessMode::Create:
                return MPI_MODE_CREATE;
        }
        return 0;
    }

    static int mpi_amode(const std::vector<ConcurrentFile::AccessMode>& access_mode) {
        return std::accumulate(access_mode.begin(), access_mode.end(), 0, [](int acc, auto mode) {
            acc |= mpi_amode(mode);
            return acc;
        });
    }

    public:
    template <class AmodeType>
    ConcurrentFile(const std::string& file_name, AmodeType access_mode, MPI_Comm comm) : handle(nullptr) {
        int err = MPI_File_open(comm, file_name.c_str(), mpi_amode(access_mode), MPI_INFO_NULL, &handle);
        check_mpi_error(err, __FILE__, __LINE__);
        blocksize = std::numeric_limits<int>::max();
        MPI_Type_contiguous(static_cast<int>(blocksize), MPI_BYTE, &page_type);
        MPI_Type_commit(&page_type);
    }

    virtual ~ConcurrentFile() {
        if (handle != nullptr) {
            int err = MPI_File_close(&handle);
            check_mpi_error(err, __FILE__, __LINE__);
            handle = nullptr;
        }
        MPI_Type_free(&page_type);
    }

    /**
     * @brief
     *
     * @tparam T the type of elements to read
     * @param buffer the read buffer (will be resized)
     * @param elements_to_read number of elements to read
     * @param position the file position to start from
     *
     * @return the actual number of read elements
     */
    template <typename T>
    size_t read(std::vector<T>& buffer, size_t elements_to_read, size_t position = 0) {
        buffer.resize(elements_to_read);
        MPI_Status status;
        size_t     bytes_to_read = elements_to_read * sizeof(T);
        size_t     total_bytes_read;
        if (bytes_to_read <= blocksize) {
            int err = MPI_File_read_at(handle, position, buffer.data(), bytes_to_read, MPI_BYTE, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_read = static_cast<size_t>(bytes_read);
        } else {
            size_t pages_to_read = bytes_to_read / blocksize;
            if (pages_to_read > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw MPIException("To many blocks in read");
            }
            int err = MPI_File_read_at(handle, position, buffer.data(), pages_to_read, page_type, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int pages_read;
            MPI_Get_count(&status, page_type, &pages_read);
            size_t remaining_bytes = bytes_to_read % blocksize;
            position += static_cast<size_t>(pages_read) * blocksize;
            std::byte* buffer_pointer = reinterpret_cast<std::byte*>(buffer.data()) + pages_read * blocksize;
            err                       = MPI_File_read_at(
                handle, static_cast<MPI_Offset>(position), buffer_pointer, static_cast<int>(remaining_bytes), MPI_BYTE,
                &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_read = static_cast<size_t>(pages_read) * blocksize + static_cast<size_t>(bytes_read);
        }
        size_t elements_read = total_bytes_read / sizeof(T);
        buffer.resize(elements_read);
        return elements_read;
    }

    /**
     * @brief
     *
     * @tparam T the type of elements to read
     * @param buffer the read buffer (will be resized)
     * @param elements_to_read number of elements to read
     * @param position the file position to start from
     *
     * @return the actual number of read elements
     */
    template <typename T>
    size_t read_collective(std::vector<T>& buffer, size_t elements_to_read, size_t position = 0) {
        buffer.resize(elements_to_read);
        MPI_Status status;
        size_t     bytes_to_read = elements_to_read * sizeof(T);
        size_t     total_bytes_read;
        size_t     pages_to_read = bytes_to_read / blocksize;
        if (pages_to_read > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw MPIException("To many blocks in read");
        }
        int err = MPI_File_read_at_all(
            handle, static_cast<MPI_Offset>(position), buffer.data(), static_cast<int>(pages_to_read), page_type,
            &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int pages_read;
        MPI_Get_count(&status, page_type, &pages_read);
        size_t remaining_bytes = bytes_to_read % blocksize;
        position += static_cast<size_t>(pages_read) * blocksize;
        std::byte* buffer_pointer =
            reinterpret_cast<std::byte*>(buffer.data()) + static_cast<size_t>(pages_read) * blocksize;
        err = MPI_File_read_at_all(
            handle, static_cast<MPI_Offset>(position), buffer_pointer, static_cast<int>(remaining_bytes), MPI_BYTE,
            &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        total_bytes_read     = static_cast<size_t>(pages_read) * blocksize + static_cast<size_t>(bytes_read);
        size_t elements_read = total_bytes_read / sizeof(T);
        buffer.resize(elements_read);
        return elements_read;
    }

    template <typename T>
    size_t write(const std::vector<T>& buffer, size_t position = 0) {
        MPI_Status status;
        size_t     bytes_to_write = buffer.size() * sizeof(T);
        size_t     total_bytes_written;
        if (bytes_to_write <= blocksize) {
            int err = MPI_File_write_at(
                handle, static_cast<MPI_Offset>(position), buffer.data(), static_cast<int>(bytes_to_write), MPI_BYTE,
                &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_written;
            MPI_Get_count(&status, MPI_BYTE, &bytes_written);
            total_bytes_written = static_cast<size_t>(bytes_written);
        } else {
            size_t pages_to_write = bytes_to_write / blocksize;
            if (pages_to_write > std::numeric_limits<int>::max()) {
                throw MPIException("To many blocks in write");
            }
            int err = MPI_File_write_at(
                handle, static_cast<MPI_Offset>(position), buffer.data(), static_cast<int>(pages_to_write), page_type,
                &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int pages_written;
            MPI_Get_count(&status, page_type, &pages_written);
            size_t remaining_bytes = bytes_to_write % blocksize;
            position += static_cast<size_t>(pages_written) * blocksize;
            const std::byte* buffer_pointer =
                reinterpret_cast<const std::byte*>(buffer.data()) + static_cast<size_t>(pages_written) * blocksize;
            err = MPI_File_write_at(
                handle, static_cast<MPI_Offset>(position), buffer_pointer, static_cast<int>(remaining_bytes), MPI_BYTE,
                &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_written = static_cast<size_t>(pages_written) * blocksize + static_cast<size_t>(bytes_read);
        }
        size_t elements_written = total_bytes_written / sizeof(T);
        return elements_written;
    }

    template <typename T>
    size_t write_collective(const std::vector<T>& buffer, size_t position = 0) {
        // MPI_Status status;
        // size_t bytes_to_write = buffer.size() * sizeof(T);
        // int err = MPI_File_write_at_all(handle, position, buffer.data(), bytes_to_write, MPI_BYTE, &status);
        // check_mpi_error(err, __FILE__, __LINE__);
        // int bytes_written;
        // MPI_Get_count(&status, MPI_BYTE, &bytes_written);
        // size_t elements_written = static_cast<size_t>(bytes_written) / sizeof(T);
        // return elements_written;
        MPI_Status status;
        size_t     bytes_to_write = buffer.size() * sizeof(T);
        size_t     total_bytes_written;
        size_t     pages_to_write = bytes_to_write / blocksize;
        if (pages_to_write > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw MPIException("To many blocks in write");
        }
        int err =
            MPI_File_write_at_all(handle, (MPI_Offset)position, buffer.data(), (int)pages_to_write, page_type, &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int pages_written;
        MPI_Get_count(&status, page_type, &pages_written);
        size_t remaining_bytes = bytes_to_write % blocksize;
        position += static_cast<size_t>(pages_written) * blocksize;
        const std::byte* buffer_pointer =
            reinterpret_cast<const std::byte*>(buffer.data()) + static_cast<size_t>(pages_written) * blocksize;
        err = MPI_File_write_at_all(
            handle, static_cast<MPI_Offset>(position), buffer_pointer, static_cast<int>(remaining_bytes), MPI_BYTE,
            &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        total_bytes_written     = static_cast<size_t>(pages_written) * blocksize + static_cast<size_t>(bytes_read);
        size_t elements_written = total_bytes_written / sizeof(T);
        return elements_written;
    }

    size_t size() {
        MPI_Offset filesize;
        MPI_File_get_size(handle, &filesize);
        return size_t(filesize);
    }

    private:
    MPI_File     handle;
    MPI_Datatype page_type;
    size_t       blocksize;
};

inline std::vector<ConcurrentFile::AccessMode>
operator|(const ConcurrentFile::AccessMode& lhs, const ConcurrentFile::AccessMode& rhs) {
    return {lhs, rhs};
}
inline std::vector<ConcurrentFile::AccessMode>
operator|(std::vector<ConcurrentFile::AccessMode>& lhs, const ConcurrentFile::AccessMode& rhs) {
    lhs.emplace_back(rhs);
    return lhs;
}

} // namespace mpi_io
