#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mmq {

struct MappedFile {
    const uint8_t * data = nullptr;
    uint64_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif

    MappedFile() = default;
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    MappedFile(MappedFile && o) noexcept { *this = std::move(o); }
    MappedFile & operator=(MappedFile && o) noexcept {
        if (this == &o) {
            return *this;
        }
        close();
        data = o.data;
        size = o.size;
#ifdef _WIN32
        file = o.file;
        mapping = o.mapping;
        o.file = INVALID_HANDLE_VALUE;
        o.mapping = nullptr;
#else
        fd = o.fd;
        o.fd = -1;
#endif
        o.data = nullptr;
        o.size = 0;
        return *this;
    }
    ~MappedFile() { close(); }

    void close() {
#ifdef _WIN32
        if (data) {
            UnmapViewOfFile(data);
            data = nullptr;
        }
        if (mapping) {
            CloseHandle(mapping);
            mapping = nullptr;
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
        }
#else
        if (data && data != MAP_FAILED) {
            munmap(const_cast<uint8_t *>(data), static_cast<size_t>(size));
        }
        data = nullptr;
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
        size = 0;
    }

    void open(const std::filesystem::path & path) {
        close();
#ifdef _WIN32
        file = CreateFileW(path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("CreateFileW failed for " + path.string());
        }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(file, &sz) || sz.QuadPart < 0) {
            throw std::runtime_error("GetFileSizeEx failed for " + path.string());
        }
        size = static_cast<uint64_t>(sz.QuadPart);
        if (size == 0) {
            throw std::runtime_error("empty file " + path.string());
        }
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            throw std::runtime_error("CreateFileMappingW failed for " + path.string());
        }
        void * view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            throw std::runtime_error("MapViewOfFile failed for " + path.string());
        }
        data = static_cast<const uint8_t *>(view);
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("open failed for " + path.string());
        }
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            throw std::runtime_error("fstat failed for " + path.string());
        }
        size = static_cast<uint64_t>(st.st_size);
        void * view = mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (view == MAP_FAILED) {
            data = nullptr;
            throw std::runtime_error("mmap failed for " + path.string());
        }
        data = static_cast<const uint8_t *>(view);
#endif
    }
};

inline std::string read_bytes(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline FILE * open_file(const std::filesystem::path & path, const char * mode) {
#ifdef _WIN32
    std::wstring wmode;
    for (const char * p = mode; *p; ++p) {
        wmode.push_back(static_cast<wchar_t>(*p));
    }
    return _wfopen(path.c_str(), wmode.c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

}  // namespace mmq
