#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

struct GGUFTensor {
    int   ggml_type = 0;
    int   n_dims = 0;
    long  dims[4] = {1, 1, 1, 1};   // ggml ne order (dims[0] fastest)
    long  n_values = 0;
    long  n_bytes = 0;
    const void* data = nullptr;     // pointer into one of the mmap'd shards
    int   shard = 0;                // which split file owns the bytes
};

// Block layout of a ggml quantisation type: how many bytes one block occupies and
// how many logical values it encodes. `elems == 0` means the type is NOT KNOWN.
//
// Public because sizing is not only the reader's business. Anything that slices,
// offsets, or uploads tensor bytes needs the same table, and a second copy of it
// would be free to drift — the failure that cost 745 GiB (IQ2_XS absent => experts
// measured zero bytes) is exactly a table that did not cover the real file.
//
// CALLERS MUST CHECK gguf_type_known() before trusting a size. An unknown type
// yields zero, and zero is indistinguishable from "empty tensor" downstream.
void gguf_block_info(int ggml_type, long& bytes, long& elems);
bool gguf_type_known(int ggml_type);

// Minimal read-only GGUF (v3) reader. Mmaps the file (or every split shard),
// parses metadata + tensor table(s), exposes scalar metadata, integer arrays,
// and tensor data pointers.
//
// Options for open(). Default is STRICT: every sibling shard must be present.
struct GGUFOpenOptions {
    // Accept a shard set with holes. Tensors living in absent shards are simply not in
    // the index — open() succeeds and reports what is missing rather than failing.
    //
    // This exists for development on a box that cannot hold the whole model. K3 at
    // UD-IQ1_S is 553 GiB across 14 shards; shard 1 is ~7 MB of pure metadata, so a
    // trivial download yields the entire real config, and one 45 GiB shard yields real
    // tensors for the early layers. That is enough to pin shapes and validate a layer
    // against llama.cpp on a single GPU.
    //
    // DEFAULT IS OFF, deliberately. A partial model that loads silently is a model that
    // generates from uninitialised weights. Callers that opt in MUST check coverage —
    // see is_partial() / missing_shards(), and kimi_k3_layer_coverage() for the
    // per-layer question that actually matters.
    bool allow_missing_shards = false;
    // Log each shard as it is mapped. Useful when a 14-shard open takes minutes.
    bool verbose = false;
};

// Split models: open() the FIRST shard (…-00001-of-NNNNN.gguf). If
// split.count > 1, siblings are discovered via the llama.cpp naming convention
// and their tensor tables are merged. Metadata always comes from shard 0.
class GGUF {
public:
    ~GGUF();

    // Open a single-file GGUF, or the first shard of a split set.
    bool open(const std::string& path, const GGUFOpenOptions& opt = {});

    // What the file claims the shard count is (1 for a single-file model).
    int  split_count()   const { return split_count_; }
    // How many shards were actually mapped.
    int  shards_loaded() const { return (int)maps_.size(); }
    // True when shards are missing — only possible under allow_missing_shards.
    bool is_partial()    const { return !missing_shards_.empty(); }
    // 1-based shard numbers that were not found, ascending.
    const std::vector<int>& missing_shards() const { return missing_shards_; }

    long        meta_int(const std::string& key, long def = 0) const;
    double      meta_float(const std::string& key, double def = 0) const;
    std::string meta_str(const std::string& key, const std::string& def = "") const;
    // Returns nullptr when the key is absent or was not an integer array.
    const std::vector<long>* meta_int_array(const std::string& key) const;
    // Returns nullptr when the key is absent or was not a captured string array.
    // Only tokenizer.ggml.{tokens,merges,token_type} are captured — see gguf.cpp.
    const std::vector<std::string>* meta_str_array(const std::string& key) const;
    const GGUFTensor* tensor(const std::string& name) const;

    size_t n_tensors() const { return tensors_.size(); }
    size_t n_shards() const { return maps_.size(); }
    // Sorted names — useful for manifests / diffs.
    std::vector<std::string> tensor_names() const;

private:
    struct MappedFile {
#ifndef _WIN32
        int    fd = -1;
#else
        void*  win_file = (void*)(intptr_t)-1;
        void*  win_map  = nullptr;
#endif
        void*  base = nullptr;
        size_t size = 0;
    };

    // Drop every tensor that came from one shard — rollback for a shard that parsed
    // partway before failing (truncated or still-downloading file).
    void drop_shard_tensors(int shard_idx);

    // Parse one already-mapped file. capture_meta=true only for shard 0.
    // shard_idx is stored on every tensor resolved from this file.
    bool parse_mapped(MappedFile& mf, bool capture_meta, int shard_idx);

    static bool map_file(const std::string& path, MappedFile& mf);
    static void unmap_file(MappedFile& mf);

    // Build sibling path: prefix + "-%05d-of-%05d.gguf" (1-based index).
    static std::string split_path(const std::string& prefix, int split_no_0based,
                                  int split_count);
    // Strip "-%05d-of-%05d.gguf" from a first-shard path to recover the prefix.
    static bool split_prefix(const std::string& path, int split_no_0based,
                             int split_count, std::string& out_prefix);

    std::vector<MappedFile> maps_;
    int              split_count_ = 1;
    std::vector<int> missing_shards_;   // 1-based, ascending
    std::unordered_map<std::string, long>               ints_;
    std::unordered_map<std::string, double>             floats_;
    std::unordered_map<std::string, std::string>        strs_;
    std::unordered_map<std::string, std::vector<long>>  int_arrays_;
    std::unordered_map<std::string, std::vector<std::string>> str_arrays_;
    std::unordered_map<std::string, GGUFTensor>         tensors_;
};

} // namespace sparkinfer
