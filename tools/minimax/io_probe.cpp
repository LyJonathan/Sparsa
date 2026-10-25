// io_probe - the drive-side measurement for the MiniMax SSD tier.
//
// It issues the same shape of request the decode path issues (4 KiB-aligned, unbuffered, whole role slices) at a
// chosen queue depth and reports the achieved rate, so a slow token can be attributed to the read path rather than
// guessed at.  It also A/Bs the one flag the engine used to set without measuring it:
// FILE_FLAG_RANDOM_ACCESS.
//
//   cl /O2 /EHsc /std:c++17 io_probe.cpp /Fe:io_probe.exe
//   io_probe.exe <file> [--bytes 5700000] [--reads 64] [--qd 1,2,4,8,16,32] [--flags random|plain]
//                 [--chunks 1] [--reps 3]
//
// Only a measurement tool: the engine does not include it.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kAlign = 4096;

struct Req {
    OVERLAPPED ov{};
    uint64_t off = 0;
    uint32_t len = 0;
    uint32_t idx = 0;
};

HANDLE open_file(const std::wstring& path, bool random_access) {
    DWORD flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
    if (random_access) flags |= FILE_FLAG_RANDOM_ACCESS;
    return CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
}

double now_ms() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1e6;
}

uint64_t rng_state = 0x243F6A8885A308D3ull;
uint64_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: io_probe <file> [--bytes N] [--reads N] [--qd a,b,..] [--flags random|plain]\n");
        return 2;
    }
    std::wstring path = argv[1];
    uint64_t bytes = 5452595;
    int reads = 64;
    std::vector<int> qds{1, 2, 4, 8, 16, 32};
    int flags = -1;   // -1 = both
    int chunks = 1;
    int reps = 3;
    for (int i = 2; i < argc; ++i) {
        std::wstring a = argv[i];
        auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
        if (a == L"--bytes") bytes = _wcstoui64(next().c_str(), nullptr, 10);
        else if (a == L"--reads") reads = _wtoi(next().c_str());
        else if (a == L"--qd") {
            qds.clear();
            std::wstring v = next();
            size_t at = 0;
            while (at <= v.size()) {
                const size_t c = v.find(L',', at);
                std::wstring p = v.substr(at, c == std::wstring::npos ? std::wstring::npos : c - at);
                if (!p.empty()) qds.push_back(_wtoi(p.c_str()));
                if (c == std::wstring::npos) break;
                at = c + 1;
            }
        } else if (a == L"--flags") {
            std::wstring v = next();
            flags = v == L"random" ? 1 : 0;
        } else if (a == L"--chunks") chunks = std::max(1, _wtoi(next().c_str()));
        else if (a == L"--reps") reps = std::max(1, _wtoi(next().c_str()));
    }
    bytes = (bytes + kAlign - 1) & ~(uint64_t)(kAlign - 1);

    HANDLE probe = open_file(path, false);
    if (probe == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "cannot open the file (error %lu)\n", GetLastError());
        return 1;
    }
    LARGE_INTEGER sz{};
    GetFileSizeEx(probe, &sz);
    CloseHandle(probe);
    const uint64_t file_size = (uint64_t)sz.QuadPart;
    if (file_size < bytes + kAlign) {
        std::fprintf(stderr, "file is %llu B, smaller than one read\n", (unsigned long long)file_size);
        return 1;
    }

    // One fixed request list, so every configuration reads the same offsets.  Chunks split each role into equal
    // 4 KiB pieces, which is what --minimax-ssd-chunks does.
    const uint32_t step = (uint32_t)(((bytes + (uint64_t)chunks - 1) / (uint64_t)chunks + kAlign - 1) & ~(uint64_t)(kAlign - 1));
    std::vector<uint64_t> offsets((size_t)reads);
    for (int i = 0; i < reads; ++i) {
        const uint64_t span = file_size - bytes - kAlign;
        uint64_t off = (rng() % span) & ~(uint64_t)(kAlign - 1);
        offsets[(size_t)i] = off;
    }
    const uint64_t total_bytes = (uint64_t)reads * bytes;
    std::printf("file=%llu MiB reads=%d bytes/read=%.2f MiB chunks=%d total=%.1f MiB\n",
                (unsigned long long)(file_size >> 20), reads, (double)bytes / 1048576.0, chunks,
                (double)total_bytes / 1048576.0);

    for (int rnd = 0; rnd < 2; ++rnd) {
        const bool random_access = flags == -1 ? rnd == 0 : flags == 1;
        HANDLE h = open_file(path, random_access);
        if (h == INVALID_HANDLE_VALUE) {
            std::fprintf(stderr, "cannot open\n");
            return 1;
        }
        HANDLE port = CreateIoCompletionPort(h, nullptr, 0, 1);
        // 4 KiB-aligned heap: one buffer per in-flight piece.  Each chunk piece writes into its own region so the
        // measured bytes are the same range either way.
        const int pieces_per_read = (int)((bytes + step - 1) / step);
        const int max_pieces = std::max(qds.back(), 1) * pieces_per_read + 8;
        const size_t buf_bytes = (size_t)max_pieces * step;
        uint8_t* buf = (uint8_t*)VirtualAlloc(nullptr, buf_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        std::vector<Req> reqs((size_t)max_pieces);
        for (int qd : qds) {
            if (qd < 1) continue;
            double best = 1e9, sum = 0;
            for (int rep = 0; rep < reps; ++rep) {
                // Free request objects, not an index: completions arrive out of order, so the object must stay
                // owned by its read until the port reports it.
                std::vector<int> free_list;
                for (int i = 0; i < max_pieces; ++i) free_list.push_back(i);
                int issued = 0;   // reads (not pieces) started
                auto issue = [&](int idx, int piece) {
                    if (idx >= reads || piece >= pieces_per_read || free_list.empty()) return false;
                    const int slot = free_list.back();
                    free_list.pop_back();
                    Req& r = reqs[(size_t)slot];
                    std::memset(&r.ov, 0, sizeof r.ov);
                    const uint64_t off = offsets[(size_t)idx] + (uint64_t)piece * step;
                    const uint32_t len = (uint32_t)std::min<uint64_t>(step, bytes - (uint64_t)piece * step);
                    r.off = off;
                    r.len = len;
                    r.idx = (uint32_t)idx;
                    r.ov.Offset = (DWORD)(off & 0xFFFFFFFFull);
                    r.ov.OffsetHigh = (DWORD)(off >> 32);
                    uint8_t* dst = buf + (size_t)slot * step;
                    if (!ReadFile(h, dst, len, nullptr, &r.ov)) {
                        const DWORD e = GetLastError();
                        if (e != ERROR_IO_PENDING) {
                            std::fprintf(stderr, "ReadFile failed %lu\n", e);
                            std::exit(1);
                        }
                    }
                    return true;
                };
                std::vector<int> remain((size_t)reads, pieces_per_read);
                int outstanding = 0;
                auto start_read = [&](int idx) {
                    for (int p = 0; p < pieces_per_read; ++p) issue(idx, p);
                    ++issued;
                    ++outstanding;
                };
                const double t0 = now_ms();
                for (int q = 0; q < qd && q < reads; ++q) start_read(q);
                int done = 0;
                const int total_pieces = reads * pieces_per_read;
                OVERLAPPED_ENTRY entries[64];
                while (done < total_pieces) {
                    ULONG got = 0;
                    if (!GetQueuedCompletionStatusEx(port, entries, 64, &got, INFINITE, FALSE)) continue;
                    for (ULONG i = 0; i < got; ++i) {
                        Req* r = (Req*)entries[i].lpOverlapped;
                        free_list.push_back((int)(r - reqs.data()));
                        ++done;
                        if (--remain[r->idx] == 0) --outstanding;
                        // Refill only when a whole read is back, so `qd` counts reads in flight.
                        while (outstanding < qd && issued < reads) start_read(issued);
                    }
                }
                const double dt = (now_ms() - t0) / 1000.0;
                const double gbs = (double)total_bytes / dt / 1e9;
                best = std::min(best, dt);
                sum += gbs;
            }
            std::printf("%-6s qd=%2d chunks=%d  %6.2f GB/s  (rep mean %.2f, best-of-%d %.2f GB/s)\n",
                        random_access ? "random" : "plain", qd, chunks, sum / reps, sum / reps, reps,
                        (double)total_bytes / best / 1e9);
        }
        VirtualFree(buf, 0, MEM_RELEASE);
        CloseHandle(port);
        CloseHandle(h);
        if (flags != -1) break;
    }
    return 0;
}
