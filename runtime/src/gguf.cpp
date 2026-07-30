// Minimal GGUF v3 reader (mmap). Parses header, metadata KV (scalars + integer
// arrays captured; string arrays skipped), and the tensor table; resolves
// tensor data pointers. Split sets (split.count > 1) are opened via the first
// shard and sibling paths are merged into one tensor map.

#include "sparkinfer/gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace sparkinfer {

namespace {
// ggml value types
enum { VT_U8=0, VT_I8=1, VT_U16=2, VT_I16=3, VT_U32=4, VT_I32=5, VT_F32=6,
       VT_BOOL=7, VT_STR=8, VT_ARR=9, VT_U64=10, VT_I64=11, VT_F64=12 };

int scalar_size(uint32_t t) {
    switch (t) { case VT_U8: case VT_I8: case VT_BOOL: return 1;
        case VT_U16: case VT_I16: return 2; case VT_U32: case VT_I32: case VT_F32: return 4;
        case VT_U64: case VT_I64: case VT_F64: return 8; default: return 0; }
}

struct Cursor {
    const uint8_t* p; size_t off, size; bool ok = true;
    template <class T> T rd() { T v{}; if (off + sizeof(T) > size) { ok=false; return v; } memcpy(&v, p+off, sizeof(T)); off += sizeof(T); return v; }
    std::string rd_str() { uint64_t n = rd<uint64_t>(); if (!ok || off+n>size) { ok=false; return {}; } std::string s((const char*)(p+off), n); off += n; return s; }
    void skip(size_t n) { off += n; if (off > size) ok = false; }
};

// block (bytes, elements) per ggml type for n_bytes computation.
//
// COMPLETENESS MATTERS MORE THAN IT LOOKS. A missing entry does not fail — it
// silently yields n_bytes = 0, which reads as "an empty tensor" rather than "a
// tensor I cannot size". On the real Kimi K3 UD-Q2_K_XL that cost 745 GiB: the
// three ffn_*_exps tensors are IQ2_XS (type 17), which was absent, so the whole
// 802 GiB model measured as 57.9 GiB and every expert appeared to be zero bytes.
// Nothing can shard, offset, or upload a tensor whose size is zero, so this
// blocked the multi-GPU path outright while looking like a successful parse.
//
// An unsloth "dynamic" quant deliberately MIXES types per tensor — attention in
// Q8_0, experts in an IQ2 variant, norms in F32 — so a table covering only the
// common K-quants is not enough. The IQ family below is what UD quants actually
// reach for. Sizes are ggml's block layouts (ggml/src/ggml.c type_traits).
void block_info(int t, long& bytes, long& elems) {
    switch (t) {
        case 0:  bytes=4;   elems=1;   break;   // F32
        case 1:  bytes=2;   elems=1;   break;   // F16
        case 2:  bytes=18;  elems=32;  break;   // Q4_0
        case 3:  bytes=20;  elems=32;  break;   // Q4_1
        case 6:  bytes=22;  elems=32;  break;   // Q5_0
        case 7:  bytes=24;  elems=32;  break;   // Q5_1
        case 8:  bytes=34;  elems=32;  break;   // Q8_0
        case 9:  bytes=36;  elems=32;  break;   // Q8_1
        case 10: bytes=84;  elems=256; break;   // Q2_K
        case 11: bytes=110; elems=256; break;   // Q3_K
        case 12: bytes=144; elems=256; break;   // Q4_K
        case 13: bytes=176; elems=256; break;   // Q5_K
        case 14: bytes=210; elems=256; break;   // Q6_K
        case 15: bytes=292; elems=256; break;   // Q8_K
        case 16: bytes=66;  elems=256; break;   // IQ2_XXS
        case 17: bytes=74;  elems=256; break;   // IQ2_XS   <- K3 experts
        case 18: bytes=98;  elems=256; break;   // IQ3_XXS
        case 19: bytes=50;  elems=256; break;   // IQ1_S
        case 20: bytes=18;  elems=32;  break;   // IQ4_NL
        case 21: bytes=110; elems=256; break;   // IQ3_S
        case 22: bytes=82;  elems=256; break;   // IQ2_S
        case 23: bytes=136; elems=256; break;   // IQ4_XS
        case 24: bytes=1;   elems=1;   break;   // I8
        case 25: bytes=2;   elems=1;   break;   // I16
        case 26: bytes=4;   elems=1;   break;   // I32
        case 27: bytes=8;   elems=1;   break;   // I64
        case 28: bytes=8;   elems=1;   break;   // F64
        case 29: bytes=56;  elems=256; break;   // IQ1_M
        case 30: bytes=2;   elems=1;   break;   // BF16
        case 39: bytes=17;  elems=32;  break;   // MXFP4
        default: bytes=0;   elems=0;   break;
    }
}

// True when block_info() knows the type. Callers that need a real size (upload,
// sharding, offset arithmetic) must check this and refuse rather than treat an
// unknown type as a zero-byte tensor — see the comment above.
bool block_known(int t) {
    long b = 0, e = 0;
    block_info(t, b, e);
    return e != 0;
}
} // namespace

bool GGUF::map_file(const std::string& path, MappedFile& mf) {
#ifdef _WIN32
    mf.win_file = (void*)CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (mf.win_file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[gguf] open failed: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx((HANDLE)mf.win_file, &li) || li.QuadPart <= 0) {
        fprintf(stderr, "[gguf] stat failed: %s\n", path.c_str());
        CloseHandle((HANDLE)mf.win_file);
        mf.win_file = INVALID_HANDLE_VALUE;
        return false;
    }
    mf.size = (size_t)li.QuadPart;
    mf.win_map = (void*)CreateFileMappingA((HANDLE)mf.win_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mf.win_map) {
        fprintf(stderr, "[gguf] CreateFileMapping failed: %s\n", path.c_str());
        CloseHandle((HANDLE)mf.win_file);
        mf.win_file = INVALID_HANDLE_VALUE;
        return false;
    }
    mf.base = MapViewOfFile((HANDLE)mf.win_map, FILE_MAP_READ, 0, 0, 0);
    if (!mf.base) {
        fprintf(stderr, "[gguf] MapViewOfFile failed: %s\n", path.c_str());
        CloseHandle((HANDLE)mf.win_map);
        CloseHandle((HANDLE)mf.win_file);
        mf.win_map = nullptr;
        mf.win_file = INVALID_HANDLE_VALUE;
        return false;
    }
#else
    mf.fd = ::open(path.c_str(), O_RDONLY);
    if (mf.fd < 0) { fprintf(stderr, "[gguf] open failed: %s\n", path.c_str()); return false; }
    struct stat st{};
    if (fstat(mf.fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "[gguf] stat failed: %s\n", path.c_str());
        close(mf.fd); mf.fd = -1; return false;
    }
    mf.size = (size_t)st.st_size;
    mf.base = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (mf.base == MAP_FAILED) {
        fprintf(stderr, "[gguf] mmap failed: %s\n", path.c_str());
        close(mf.fd); mf.fd = -1; mf.base = nullptr; return false;
    }
#endif
    return true;
}

void GGUF::unmap_file(MappedFile& mf) {
#ifdef _WIN32
    if (mf.base) UnmapViewOfFile(mf.base);
    if (mf.win_map) CloseHandle((HANDLE)mf.win_map);
    if (mf.win_file && mf.win_file != INVALID_HANDLE_VALUE) CloseHandle((HANDLE)mf.win_file);
    mf.base = nullptr; mf.win_map = nullptr; mf.win_file = INVALID_HANDLE_VALUE; mf.size = 0;
#else
    if (mf.base && mf.base != MAP_FAILED) munmap(mf.base, mf.size);
    if (mf.fd >= 0) close(mf.fd);
    mf.base = nullptr; mf.fd = -1; mf.size = 0;
#endif
}

std::string GGUF::split_path(const std::string& prefix, int split_no_0based, int split_count) {
    char buf[32];
    snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", split_no_0based + 1, split_count);
    return prefix + buf;
}

bool GGUF::split_prefix(const std::string& path, int split_no_0based, int split_count,
                        std::string& out_prefix) {
    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no_0based + 1, split_count);
    const size_t plen = strlen(postfix);
    if (path.size() < plen) return false;
    if (path.compare(path.size() - plen, plen, postfix) != 0) return false;
    out_prefix = path.substr(0, path.size() - plen);
    return true;
}

GGUF::~GGUF() {
    for (auto& m : maps_) unmap_file(m);
    maps_.clear();
}

bool GGUF::parse_mapped(MappedFile& mf, bool capture_meta, int shard_idx) {
    Cursor c{ (const uint8_t*)mf.base, 0, mf.size };
    char magic[4]; memcpy(magic, c.p, 4); c.off = 4;
    if (memcmp(magic, "GGUF", 4) != 0) { fprintf(stderr, "[gguf] bad magic (shard %d)\n", shard_idx); return false; }
    uint32_t version = c.rd<uint32_t>();
    uint64_t n_tensors = c.rd<uint64_t>();
    uint64_t n_kv = c.rd<uint64_t>();
    (void)version;

    // Local copies used when capture_meta=false (still need alignment from this file).
    long alignment = 32;
    for (uint64_t i = 0; i < n_kv && c.ok; i++) {
        std::string key = c.rd_str();
        uint32_t vt = c.rd<uint32_t>();
        if (vt == VT_STR) {
            std::string s = c.rd_str();
            if (capture_meta) strs_[key] = std::move(s);
        } else if (vt == VT_F32) {
            float v = c.rd<float>();
            if (capture_meta) floats_[key] = v;
        } else if (vt == VT_F64) {
            double v = c.rd<double>();
            if (capture_meta) floats_[key] = v;
        } else if (vt == VT_BOOL || vt == VT_U8) {
            long v = c.rd<uint8_t>();
            if (capture_meta) ints_[key] = v;
            if (key == "general.alignment") alignment = v;
        } else if (vt == VT_I8) {
            long v = c.rd<int8_t>();
            if (capture_meta) ints_[key] = v;
        } else if (vt == VT_U16) {
            long v = c.rd<uint16_t>();
            if (capture_meta) ints_[key] = v;
            if (key == "split.no" || key == "split.count") {
                // Always keep split bookkeeping keys, even on later shards —
                // we validate split.no against the expected index.
                ints_[key] = v;
            }
        } else if (vt == VT_I16) {
            long v = c.rd<int16_t>();
            if (capture_meta) ints_[key] = v;
        } else if (vt == VT_U32) {
            long v = c.rd<uint32_t>();
            if (capture_meta) ints_[key] = v;
            if (key == "general.alignment") alignment = v;
        } else if (vt == VT_I32) {
            long v = c.rd<int32_t>();
            if (capture_meta) ints_[key] = v;
            if (key == "split.tensors.count") ints_[key] = v;
        } else if (vt == VT_U64) {
            long v = (long)c.rd<uint64_t>();
            if (capture_meta) ints_[key] = v;
        } else if (vt == VT_I64) {
            long v = c.rd<int64_t>();
            if (capture_meta) ints_[key] = v;
        } else if (vt == VT_ARR) {
            uint32_t et = c.rd<uint32_t>(); uint64_t n = c.rd<uint64_t>();
            if (et == VT_STR) {
                for (uint64_t k = 0; k < n && c.ok; k++) c.rd_str();
            } else if (et == VT_U8 || et == VT_I8 || et == VT_BOOL ||
                       et == VT_U16 || et == VT_I16 ||
                       et == VT_U32 || et == VT_I32 ||
                       et == VT_U64 || et == VT_I64) {
                int es = scalar_size(et);
                if (es == 0 || n > (c.size - c.off) / (size_t)es) {
                    fprintf(stderr, "[gguf] bad metadata array (elem type %u, n=%llu) for %s\n",
                            et, (unsigned long long)n, key.c_str());
                    return false;
                }
                std::vector<long> vals;
                vals.reserve((size_t)n);
                for (uint64_t k = 0; k < n && c.ok; k++) {
                    long v = 0;
                    switch (et) {
                        case VT_U8:   v = (long)c.rd<uint8_t>();  break;
                        case VT_I8:   v = (long)c.rd<int8_t>();   break;
                        case VT_BOOL: v = (long)c.rd<uint8_t>();  break;
                        case VT_U16:  v = (long)c.rd<uint16_t>(); break;
                        case VT_I16:  v = (long)c.rd<int16_t>();  break;
                        case VT_U32:  v = (long)c.rd<uint32_t>(); break;
                        case VT_I32:  v = (long)c.rd<int32_t>();  break;
                        case VT_U64:  v = (long)c.rd<uint64_t>(); break;
                        case VT_I64:  v = (long)c.rd<int64_t>();  break;
                    }
                    vals.push_back(v);
                }
                if (c.ok && capture_meta) int_arrays_[key] = std::move(vals);
            } else {
                int es = scalar_size(et);
                if (es == 0 || n > (c.size - c.off) / (size_t)es) {
                    fprintf(stderr, "[gguf] bad metadata array (elem type %u, n=%llu) for %s\n",
                            et, (unsigned long long)n, key.c_str());
                    return false;
                }
                c.skip((size_t)n * es);
            }
        } else {
            fprintf(stderr, "[gguf] unknown vt %u for %s\n", vt, key.c_str());
            return false;
        }
    }
    if (!c.ok) { fprintf(stderr, "[gguf] metadata parse error (shard %d)\n", shard_idx); return false; }

    if (capture_meta && ints_.count("general.alignment"))
        alignment = ints_["general.alignment"];
    if (alignment <= 0) alignment = 32;

    // Verify split.no on non-first shards (and on the first).
    if (ints_.count("split.no")) {
        long got = ints_["split.no"];
        if (got != shard_idx) {
            fprintf(stderr, "[gguf] split.no=%ld but expected shard index %d\n", got, shard_idx);
            return false;
        }
    }

    struct Info { std::string name; GGUFTensor t; uint64_t offset; };
    std::vector<Info> infos; infos.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors && c.ok; i++) {
        Info in; in.name = c.rd_str();
        uint32_t nd = c.rd<uint32_t>();
        if (!c.ok || nd > 4) {
            fprintf(stderr, "[gguf] tensor %s has invalid n_dims=%u (max 4)\n",
                    in.name.c_str(), nd);
            return false;
        }
        in.t.n_dims = (int)nd;
        long nv = 1;
        for (uint32_t d = 0; d < nd; d++) { long e = (long)c.rd<uint64_t>(); in.t.dims[d] = e; nv *= e; }
        in.t.ggml_type = (int)c.rd<uint32_t>();
        in.offset = c.rd<uint64_t>();
        in.t.n_values = nv;
        in.t.shard = shard_idx;
        long bb, be; block_info(in.t.ggml_type, bb, be);
        in.t.n_bytes = be ? (nv / be) * bb : 0;
        // An unknown type silently sizes to 0, which downstream reads as "empty
        // tensor" rather than "cannot size this". That is how 745 GiB of IQ2_XS
        // experts once measured as zero. Say it loudly, once per type.
        if (!be) {
            static std::vector<int> warned;
            if (std::find(warned.begin(), warned.end(), in.t.ggml_type) == warned.end()) {
                warned.push_back(in.t.ggml_type);
                fprintf(stderr, "[gguf] WARNING: unknown ggml_type %d (first seen on '%s') — "
                                "n_bytes=0 for every tensor of this type. Add it to block_info() "
                                "before sharding or uploading.\n",
                        in.t.ggml_type, in.name.c_str());
            }
        }
        infos.push_back(std::move(in));
    }
    if (!c.ok) { fprintf(stderr, "[gguf] tensor table parse error (shard %d)\n", shard_idx); return false; }

    size_t data_start = (c.off + (size_t)alignment - 1) / (size_t)alignment * (size_t)alignment;
    for (auto& in : infos) {
        if (tensors_.count(in.name)) {
            fprintf(stderr, "[gguf] duplicate tensor '%s' (shard %d)\n", in.name.c_str(), shard_idx);
            return false;
        }
        if (data_start > mf.size || in.offset > mf.size - data_start) {
            fprintf(stderr, "[gguf] tensor %s offset out of bounds (shard %d)\n",
                    in.name.c_str(), shard_idx);
            return false;
        }
        // Strict end-bound only when we know the packed size.
        if (in.t.n_bytes > 0 &&
            (uint64_t)in.t.n_bytes > mf.size - data_start - in.offset) {
            fprintf(stderr, "[gguf] tensor %s data out of bounds (offset=%llu n_bytes=%ld file=%zu shard=%d)\n",
                    in.name.c_str(), (unsigned long long)in.offset, in.t.n_bytes, mf.size, shard_idx);
            return false;
        }
        in.t.data = (const uint8_t*)mf.base + data_start + in.offset;
        tensors_[in.name] = in.t;
    }
    return true;
}

bool GGUF::open(const std::string& path) {
    for (auto& m : maps_) unmap_file(m);
    maps_.clear();
    ints_.clear(); floats_.clear(); strs_.clear(); int_arrays_.clear(); tensors_.clear();

    MappedFile first;
    if (!map_file(path, first)) return false;
    maps_.push_back(first);
    if (!parse_mapped(maps_.back(), /*capture_meta=*/true, /*shard_idx=*/0)) return false;

    long n_split = ints_.count("split.count") ? ints_["split.count"] : 1;
    if (n_split <= 1) return true;

    if (ints_.count("split.no") && ints_["split.no"] != 0) {
        fprintf(stderr, "[gguf] must open the first split (split.no=0), got %ld (%s)\n",
                ints_["split.no"], path.c_str());
        return false;
    }

    std::string prefix;
    if (!split_prefix(path, 0, (int)n_split, prefix)) {
        fprintf(stderr, "[gguf] cannot derive split prefix from %s (expected …-00001-of-%05ld.gguf)\n",
                path.c_str(), n_split);
        return false;
    }

    for (int i = 1; i < (int)n_split; ++i) {
        const std::string sp = split_path(prefix, i, (int)n_split);
        MappedFile mf;
        if (!map_file(sp, mf)) return false;
        maps_.push_back(mf);
        // Clear per-shard split.no before parse so a missing key is distinguishable.
        ints_.erase("split.no");
        if (!parse_mapped(maps_.back(), /*capture_meta=*/false, /*shard_idx=*/i)) return false;
    }

    if (ints_.count("split.tensors.count")) {
        long want = ints_["split.tensors.count"];
        if (want != (long)tensors_.size()) {
            fprintf(stderr, "[gguf] split.tensors.count=%ld but loaded %zu tensors\n",
                    want, tensors_.size());
            return false;
        }
    }
    // Later shards overwrite split.no during validation; restore the canonical
    // "this open was from shard 0" value for callers.
    ints_["split.no"] = 0;
    return true;
}

long GGUF::meta_int(const std::string& k, long d) const {
    auto it = ints_.find(k); return it == ints_.end() ? d : it->second;
}
double GGUF::meta_float(const std::string& k, double d) const {
    auto it = floats_.find(k); return it == floats_.end() ? d : it->second;
}
std::string GGUF::meta_str(const std::string& k, const std::string& d) const {
    auto it = strs_.find(k); return it == strs_.end() ? d : it->second;
}
const std::vector<long>* GGUF::meta_int_array(const std::string& k) const {
    auto it = int_arrays_.find(k);
    return it == int_arrays_.end() ? nullptr : &it->second;
}
const GGUFTensor* GGUF::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it == tensors_.end() ? nullptr : &it->second;
}
std::vector<std::string> GGUF::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace sparkinfer
